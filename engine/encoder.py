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
    `prefetch(prompts)` queues prompts at background priority, behind every command's request.
    The cache is a bounded LRU keyed by the exact prompt text.
    """

    def __init__(self, encode_with_ids: Callable[[str], tuple[object, list[int]]], cache_size: int = 512) -> None:
        self._encode = encode_with_ids
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

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._requests.put((-1, next(self._order), None, None))

    def prefetch(self, prompts) -> int:
        """Queue every prompt not cached or in flight at background priority. Returns how many were queued."""
        queued = 0
        with self._lock:
            for prompt in prompts:
                if prompt in self._cache or prompt in self._inflight:
                    continue
                fut = Future()
                self._inflight[prompt] = fut
                self._background.add(prompt)
                self._requests.put((BACKGROUND, next(self._order), prompt, fut))
                queued += 1
        if queued:
            log.info("encoder: %d prompts queued for prefetch", queued)
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
                    log.info("encoder: %r was queued for prefetch, now urgent", _short(prompt))
                else:
                    log.info("encoder: already encoding %r", _short(prompt))
                return fut
            self.misses += 1
            fut = Future()
            self._inflight[prompt] = fut
        log.info("encoder: cache miss %r, queued (queue depth %d)", _short(prompt), self._requests.qsize())
        self._requests.put((URGENT, next(self._order), prompt, fut))
        return fut

    def encode_with_ids(self, prompt: str) -> tuple[object, list[int]]:
        """Block until (embedding, token ids) for `prompt` is ready."""
        return self.request(prompt).result()

    def encode(self, prompt: str) -> object:
        """Block until the embedding for `prompt` is ready."""
        return self.request(prompt).result()[0]

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def _run(self) -> None:
        while True:
            priority, _, prompt, fut = self._requests.get()
            if prompt is None:
                return
            if fut.done():  # the same request was queued twice (prefetch, then urgent)
                continue
            t0 = time.perf_counter()
            try:
                emb, ids = self._encode(prompt)
                result = (emb, list(ids))
            except Exception as exc:  # report to the waiter, keep the service alive
                log.exception("encoder: failed on %r", _short(prompt))
                with self._lock:
                    self._inflight.pop(prompt, None)
                    self._background.discard(prompt)
                fut.set_exception(exc)
                continue
            ms = (time.perf_counter() - t0) * 1e3
            with self._lock:
                self.last_encode_ms = ms
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
