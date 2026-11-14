"""Text encoder service: its own thread and request queue, off the generation path, with a cache by prompt text."""

import collections
import itertools
import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future

log = logging.getLogger("engine.encoder")

URGENT, BACKGROUND = 0, 1  # request priorities; lower runs first


class EncoderService:
    """`request(prompt)` returns a Future of (embedding, token ids); `encode` and `encode_with_ids` block on it.

    `encode_with_ids` is the backend's `encode_prompt_with_ids(prompt) -> (tensor, list[int])`; the embedding and
    its ids are cached together.

    A cached prompt returns an already finished Future, so it costs no encoding time.
    Identical prompts requested while one is being encoded share one Future.
    `prefetch(prompts)` makes `prompts` the background work, at background priority behind every command's request:
    background prompts still queued from an earlier call and not in `prompts` are dropped (they were the likely
    prompts of an older scene).
    The cache is a bounded LRU keyed by the exact prompt text.

    `background_turn`, if given, is asked before each background encode and returns True when one may start now.
    The engine uses it to keep prefetch off the GPU while a command is in flight and to pace it into the time
    generation would otherwise sit idle (see PrefetchPacer in engine.py). While it says no, the background item
    goes back on the queue and the thread waits for `wake()` or a new request; urgent requests never wait for it.
    """

    WAIT_S = 0.25  # longest wait for a turn before asking again

    def __init__(self, encode_with_ids: Callable[[str], tuple[object, list[int]]], cache_size: int = 512,
                 background_turn: Callable[[], bool] | None = None) -> None:
        self._encode = encode_with_ids
        self._background_turn = background_turn
        self._wake = threading.Event()  # set by a new request and by wake(): the thread looks at the queue again
        self._cache_size = cache_size
        self._cache: collections.OrderedDict[str, object] = collections.OrderedDict()
        self._inflight: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._requests: queue.PriorityQueue = queue.PriorityQueue()
        self._order = itertools.count()
        self._background: set[str] = set()  # in-flight prompts that were queued at background priority
        self._thread = threading.Thread(target=self._run, name="encoder", daemon=True)
        self.hits = 0
        self.misses = 0
        self.last_encode_ms: float | None = None
        self._recent_ms: collections.deque[float] = collections.deque(maxlen=20)
        self._running: str | None = None  # the prompt being encoded now; its Future cannot be cancelled

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._requests.put((-1, next(self._order), None, None))
        self._wake.set()

    def wake(self) -> None:
        """A background turn may be free now: look at the queue again."""
        self._wake.set()

    def prefetch(self, prompts) -> int:
        """Queue every prompt not cached or in flight at background priority, and drop queued background prompts
        not in `prompts`. Returns how many were queued."""
        prompts = list(prompts)
        queued = 0
        with self._lock:
            keep = set(prompts)
            stale = [p for p in self._background if p not in keep and p != self._running]
            for prompt in stale:
                self._background.discard(prompt)
                self._inflight.pop(prompt).cancel()  # its queue entry is skipped as done
            for prompt in prompts:
                if prompt in self._cache or prompt in self._inflight:
                    continue
                fut = Future()
                self._inflight[prompt] = fut
                self._background.add(prompt)
                self._requests.put((BACKGROUND, next(self._order), prompt, fut))
                queued += 1
        if queued or stale:
            log.info("encoder: %d prompts queued for prefetch, %d stale ones dropped", queued, len(stale))
        return queued

    def request(self, prompt: str) -> Future:
        with self._lock:
            if prompt in self._cache:
                self._cache.move_to_end(prompt)
                self.hits += 1
                fut: Future = Future()
                fut.set_result(self._cache[prompt])
                log.info("encoder: cache hit %r", _short(prompt))
                return fut
            if prompt in self._inflight:
                fut = self._inflight[prompt]
                if prompt in self._background:  # move it ahead of the other background work
                    self._background.discard(prompt)
                    self._requests.put((URGENT, next(self._order), prompt, fut))
                    self._wake.set()
                    log.info("encoder: %r was queued for prefetch, now urgent", _short(prompt))
                else:
                    log.info("encoder: already encoding %r", _short(prompt))
                return fut
            self.misses += 1
            fut = Future()
            self._inflight[prompt] = fut
        log.info("encoder: cache miss %r, queued (queue depth %d)", _short(prompt), self._requests.qsize())
        self._requests.put((URGENT, next(self._order), prompt, fut))
        self._wake.set()
        return fut

    def encode_with_ids(self, prompt: str) -> tuple[object, list[int]]:
        """Block until (embedding, token ids) for `prompt` is ready."""
        return self.request(prompt).result()

    def encode(self, prompt: str) -> object:
        """Block until the embedding for `prompt` is ready."""
        return self.request(prompt).result()[0]

    @property
    def pending(self) -> int:
        """Prompts queued or being encoded, prefetch included; 0 means the encoder is idle."""
        with self._lock:
            return len(self._inflight)

    @property
    def work_ms(self) -> float | None:
        """The fastest of the last 20 encodes: about what one encode costs the GPU when it does not have to share
        it. None before the first encode."""
        with self._lock:
            return min(self._recent_ms) if self._recent_ms else None

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def _run(self) -> None:
        while True:
            item = self._requests.get()
            priority, _, prompt, fut = item
            if prompt is None:
                return
            if fut.done():  # the same request was queued twice (prefetch, then urgent)
                continue
            if priority == BACKGROUND and self._background_turn is not None and not self._background_turn():
                self._requests.put(item)  # keeps its place: same priority and order number
                self._wake.wait(self.WAIT_S)
                self._wake.clear()  # a request that set it is already on the queue, so none is missed
                continue
            with self._lock:
                if fut.done():  # dropped as stale since it was taken off the queue
                    continue
                self._running = prompt
            t0 = time.perf_counter()
            try:
                emb, ids = self._encode(prompt)
                result = (emb, list(ids))
            except Exception as exc:  # report to the waiter, keep the service alive
                log.exception("encoder: failed on %r", _short(prompt))
                with self._lock:
                    self._running = None
                    self._inflight.pop(prompt, None)
                    self._background.discard(prompt)
                fut.set_exception(exc)
                continue
            ms = (time.perf_counter() - t0) * 1e3
            with self._lock:
                self._running = None
                self.last_encode_ms = ms
                self._recent_ms.append(ms)
                self._cache[prompt] = result
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
                self._inflight.pop(prompt, None)
                self._background.discard(prompt)
            log.info("encoder: encoded %r in %.0f ms%s", _short(prompt), ms,
                     " (prefetch)" if priority == BACKGROUND else "")
            fut.set_result(result)


def _short(text: str, n: int = 60) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."
