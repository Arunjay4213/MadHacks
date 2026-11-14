"""Measure real T5 text-encoder embeddings the way the steerer will blend them.

Run:  python -m steering.eval.embedding_stats [--model google/umt5-small]
          [--device cpu] [--max-length 512] [--prompts file.json] [--align]

Prompts are encoded like Wan2.1 (see Causal-Forcing utils/wan_wrapper.py):
whitespace collapsed, tokenized with EOS and padded to max_length, encoded,
then every position past the real token count set to 0.

The numeric helpers below take tensors only, so they can be tested without a
model. `transformers` is imported only inside `encode`.
"""

import argparse
import json
from collections.abc import Sequence

import torch
from torch import Tensor

from steering.align import Aligner
from steering.blend import blend

DEFAULT_PROMPTS = {
    "base": "a wide shot of a meadow with a single oak tree, gentle wind, cinematic",
    "changes": {
        "winter": ", in deep winter with snow on the ground",
        "night": ", at night under a full moon",
        "rain": ", in heavy rain",
        "golden": ", at golden hour sunset",
        "fog": ", in thick fog",
    },
    "stack": ["winter", "night", "rain"],
}


# ---------------------------------------------------------------- numerics


def angles_deg(a: Tensor, b: Tensor) -> Tensor:
    """Angle in degrees between matching token vectors (last dim), float64."""
    x, y = a.double(), b.double()
    ux = x / x.norm(dim=-1, keepdim=True)
    uy = y / y.norm(dim=-1, keepdim=True)
    cos = (ux * uy).sum(dim=-1, keepdim=True)
    sin = (uy - cos * ux).norm(dim=-1, keepdim=True)
    return torch.rad2deg(torch.atan2(sin, cos)).squeeze(-1)


def spherical_mask(a: Tensor, b: Tensor, eps: float = 1e-4) -> Tensor:
    """Tokens where `steering.blend.blend` takes the spherical path.

    Copy of blend's rule: both norms >= eps and sin(angle) >= eps.
    """
    work = torch.float64 if a.dtype == torch.float64 else torch.float32
    x, y = a.to(work), b.to(work)
    nx = x.norm(dim=-1, keepdim=True)
    ny = y.norm(dim=-1, keepdim=True)
    ux = x / nx.clamp_min(eps)
    uy = y / ny.clamp_min(eps)
    cos = (ux * uy).sum(dim=-1, keepdim=True)
    sin = (uy - cos * ux).norm(dim=-1, keepdim=True)
    return ((nx >= eps) & (ny >= eps) & (sin >= eps)).squeeze(-1)


def position_groups(len_a: int, len_b: int, total: int) -> dict[str, Tensor]:
    """Boolean masks over `total` positions: real in both, one only, neither."""
    pos = torch.arange(total)
    return real_groups(pos < len_a, pos < len_b)


def real_groups(ra: Tensor, rb: Tensor) -> dict[str, Tensor]:
    """Same as `position_groups`, from masks of the real positions of a and b."""
    return {"both": ra & rb, "one": ra ^ rb, "neither": ~(ra | rb)}


def real_mask(positions: Sequence[int], total: int) -> Tensor:
    """Boolean mask over `total` positions, True at `positions`."""
    m = torch.zeros(total, dtype=torch.bool)
    m[list(positions)] = True
    return m


def align_pair(
    a: Tensor, b: Tensor, ids_a: list[int], ids_b: list[int]
) -> tuple[Tensor, list[int]]:
    """Place `b` with a fresh Aligner whose previous layout is `a` (identity).

    Returns (placed b, position of each of b's real tokens).
    """
    al = Aligner(a.shape[0])
    al.place(a, ids_a)
    b_al = al.place(b, ids_b)
    return b_al, [p for p, _ in al.layout]


def midpoint_norm_ratios(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Norm of blend(a, b, 0.5) and of plain lerp, each / mean(|a|, |b|)."""
    ref = (a.double().norm(dim=-1) + b.double().norm(dim=-1)) / 2
    mid = blend(a, b, 0.5).double().norm(dim=-1)
    lerp = ((a.double() + b.double()) / 2).norm(dim=-1)
    return mid / ref, lerp / ref


def suffix_alignment(ids_a: Sequence[int], ids_b: Sequence[int]) -> tuple[int, int, int]:
    """Locate the shared tail after a phrase was removed from (or added to) `a`.

    Returns (start_a, start_b, n): ids_a[start_a + i] == ids_b[start_b + i]
    for i < n, the longest common suffix that does not overlap the common
    prefix in either sequence.
    """
    p = 0
    while p < min(len(ids_a), len(ids_b)) and ids_a[p] == ids_b[p]:
        p += 1
    n = 0
    limit = min(len(ids_a), len(ids_b)) - p
    while n < limit and ids_a[-1 - n] == ids_b[-1 - n]:
        n += 1
    return len(ids_a) - n, len(ids_b) - n, n


def token_shift(
    a: Tensor,
    b: Tensor,
    ids_a: Sequence[int],
    ids_b: Sequence[int],
    pos_b: Sequence[int] | None = None,
) -> dict:
    """Compare what a per-token blend mixes against the same word's embedding.

    `a`, `b`: [L, D] encodings (padding zeroed); `ids_a`, `ids_b`: the real
    token ids. `pos_b`: where b's real tokens sit (default: the first
    len(ids_b) positions; pass the aligner's positions for an aligned b).
    The shared tail sits at positions idx_a in `a` and idx_b in `b`. A
    per-token blend mixes a[i] with b[i] at every position i where either
    copy of the tail sits; the same word is at a[idx_a], b[idx_b].
    Keys: shift, tail_len, moved (tail tokens at a different position in b),
    positional_angles (positions real in both), positional_same_id (of
    those, how many pair the same token id), tail_vs_padding (tail positions
    paired with zeroed padding), aligned_angles (same word in both).
    """
    if pos_b is None:
        pos_b = range(len(ids_b))
    pos_b = list(pos_b)
    id_at_b = dict(zip(pos_b, ids_b))
    start_a, start_b, n = suffix_alignment(ids_a, ids_b)
    idx_a = torch.arange(start_a, start_a + n)
    idx_b = torch.tensor(pos_b[start_b : start_b + n], dtype=torch.long)
    pos = torch.unique(torch.cat([idx_a, idx_b]))
    both_real = (pos < len(ids_a)) & torch.isin(pos, torch.tensor(pos_b))
    pos_real = pos[both_real]
    same_id = sum(int(ids_a[i] == id_at_b[i]) for i in pos_real.tolist())
    return {
        "shift": start_a - start_b,
        "tail_len": n,
        "moved": int((idx_a != idx_b).sum()),
        "positional_angles": angles_deg(a[pos_real], b[pos_real]),
        "positional_same_id": same_id,
        "tail_vs_padding": int((~both_real).sum()),
        "aligned_angles": angles_deg(a[idx_a], b[idx_b]),
    }


def stats(x: Tensor) -> str:
    """'min / median / max' of a 1-D tensor, or 'n/a' if empty."""
    if x.numel() == 0:
        return "n/a"
    x = x.double()
    return f"{x.min():.2f} / {x.median():.2f} / {x.max():.2f}"


# ---------------------------------------------------------------- encoding


def build_prompts(spec: dict) -> dict[str, str]:
    """Named prompts: base, base+each change, the stack, stack minus middle."""
    base, changes, stack = spec["base"], spec["changes"], spec["stack"]
    if len(stack) != 3:
        raise ValueError("stack must name exactly three changes")
    prompts = {"base": base}
    prompts.update({name: base + suffix for name, suffix in changes.items()})
    prompts["stack"] = base + "".join(changes[n] for n in stack)
    prompts["stack_minus_middle"] = base + changes[stack[0]] + changes[stack[2]]
    return prompts


def build_pairs(spec: dict) -> list[tuple[str, str]]:
    """base->each change, stack->stack minus middle, each change->the next."""
    names = list(spec["changes"])
    pairs = [("base", n) for n in names]
    pairs.append(("stack", "stack_minus_middle"))
    pairs += list(zip(names, names[1:]))
    return pairs


def encode(
    texts: list[str], model_name: str, device: str, max_length: int
) -> tuple[Tensor, list[list[int]]]:
    """Wan2.1-style encoding. Returns ([N, max_length, D] float32 with padding
    zeroed, list of real token ids per text)."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    texts = [" ".join(t.split()) for t in texts]  # Wan's clean='whitespace'
    tok = AutoTokenizer.from_pretrained(model_name)
    model = UMT5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device).eval()
    enc = tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_tensors="pt",
    )
    ids, mask = enc.input_ids.to(device), enc.attention_mask.to(device)
    with torch.no_grad():
        ctx = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
    lens = mask.sum(dim=1).tolist()
    for row, n in zip(ctx, lens):
        row[n:] = 0.0
    real_ids = [enc.input_ids[i, :n].tolist() for i, n in enumerate(lens)]
    return ctx.cpu(), real_ids


# ---------------------------------------------------------------- report


def pair_report(
    name_a: str,
    name_b: str,
    a: Tensor,
    b: Tensor,
    ids_a: list[int],
    ids_b: list[int],
    pos_b: Sequence[int] | None = None,
) -> list[str]:
    """Report lines for one (from, to) pair; items 1, 2, 3 and 5 of the spec.

    `pos_b`: where b's real tokens sit, for an aligned b (header says so).
    """
    total = a.shape[0]
    ra = torch.arange(total) < len(ids_a)
    rb = torch.arange(total) < len(ids_b) if pos_b is None else real_mask(pos_b, total)
    groups = real_groups(ra, rb)
    both = groups["both"]
    sph = spherical_mask(a, b)
    slerp_r, lerp_r = midpoint_norm_ratios(a[both], b[both])
    lines = [
        f"== {name_a} -> {name_b}" + ("" if pos_b is None else "  [aligned]"),
        f"  tokens: from={len(ids_a)} to={len(ids_b)} | positions real in both="
        f"{int(both.sum())}, one only={int(groups['one'].sum())}, "
        f"padding in both={int(groups['neither'].sum())}",
        f"  angle deg (real in both)  min/med/max: {stats(angles_deg(a[both], b[both]))}",
        f"  norm from (real)          min/med/max: {stats(a[: len(ids_a)].norm(dim=-1))}",
        f"  norm to   (real)          min/med/max: {stats(b[rb].norm(dim=-1))}",
    ]
    path = "  blend path (eps=1e-4):"
    for g in ("both", "one", "neither"):
        m = groups[g]
        path += f" {g}: {int((sph & m).sum())} spherical/{int((~sph & m).sum())} linear;"
    lines.append(path.rstrip(";"))
    lines.append(f"  midpoint |blend(a,b,.5)| / mean(|a|,|b|) min/med/max: {stats(slerp_r)}")
    lines.append(f"  midpoint |lerp(a,b,.5)|  / mean(|a|,|b|) min/med/max: {stats(lerp_r)}")
    return lines


def shift_report(a: Tensor, b: Tensor, ids_a: list[int], ids_b: list[int]) -> list[str]:
    """Report lines for item 4: does a per-token blend mix unrelated words?"""
    r = token_shift(a, b, ids_a, ids_b)
    n_pos = r["positional_angles"].numel()
    n_diff = n_pos - r["positional_same_id"]
    lines = [
        "  token shift (stack -> stack_minus_middle):",
        f"    the {r['tail_len']} tokens after the removed phrase move by {r['shift']} positions",
        f"    positions holding the tail in either prompt: {n_pos} real in both "
        f"({n_diff} with different token ids), {r['tail_vs_padding']} paired with padding",
        f"    angle, same position  (what blend mixes) min/med/max: "
        f"{stats(r['positional_angles'])}",
        f"    angle, same word (aligned by offset)     min/med/max: "
        f"{stats(r['aligned_angles'])}",
    ]
    if r["shift"] == 0:
        lines.append("    VERDICT: no shift; a per-token blend pairs the same words.")
    else:
        lines.append(
            "    VERDICT: yes, a per-token blend mixes unrelated words here: "
            f"{n_diff} tail positions pair different words, and "
            f"{r['tail_vs_padding']} pair a word with zeroed padding (that word "
            "fades out at its old position while it fades in at its new one)."
        )
    return lines


def shift_report_aligned(
    a: Tensor, b: Tensor, b_al: Tensor, ids_a: list[int], ids_b: list[int],
    pos_b: Sequence[int],
) -> list[str]:
    """Item 4 before (positional) and after alignment, side by side."""
    r0 = token_shift(a, b, ids_a, ids_b)
    r1 = token_shift(a, b_al, ids_a, ids_b, pos_b)

    def row(label: str, f) -> str:
        return f"    {label:<46} {f(r0):<22} | {f(r1)}"

    def real_diff(r: dict) -> str:
        n = r["positional_angles"].numel()
        return f"{n} / {n - r['positional_same_id']}"

    return [
        "  token shift (stack -> stack_minus_middle), before | aligned:",
        f"    tail = the {r0['tail_len']} tokens after the removed phrase",
        row("tail tokens at a different position", lambda r: str(r["moved"])),
        row("tail positions real in both / diff. token id", real_diff),
        row("tail positions paired with padding", lambda r: str(r["tail_vs_padding"])),
        row("angle, same position (blend mixes) m/med/M", lambda r: stats(r["positional_angles"])),
        row("angle, same word                   m/med/M", lambda r: stats(r["aligned_angles"])),
    ]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="google/umt5-small")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--prompts", help="JSON with base, changes {name: suffix}, stack [3 names]")
    p.add_argument(
        "--align",
        action="store_true",
        help="also report each pair with the 'to' prompt placed by steering.align.Aligner",
    )
    args = p.parse_args(argv)

    spec = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts) as f:
            spec = json.load(f)
    prompts = build_prompts(spec)
    names = list(prompts)
    ctx, ids = encode([prompts[n] for n in names], args.model, args.device, args.max_length)
    emb = dict(zip(names, ctx))
    tid = dict(zip(names, ids))

    print(f"model={args.model} max_length={args.max_length} dim={ctx.shape[-1]}")
    print("prompts:")
    for n in names:
        print(f"  {n:<20} ({len(tid[n])} tok) {prompts[n]}")
    print()
    for a, b in build_pairs(spec):
        for line in pair_report(a, b, emb[a], emb[b], tid[a], tid[b]):
            print(line)
        is_shift = (a, b) == ("stack", "stack_minus_middle")
        if is_shift and not args.align:
            for line in shift_report(emb[a], emb[b], tid[a], tid[b]):
                print(line)
        if args.align:
            b_al, pos_b = align_pair(emb[a], emb[b], tid[a], tid[b])
            for line in pair_report(a, b, emb[a], b_al, tid[a], tid[b], pos_b):
                print(line)
            if is_shift:
                for line in shift_report_aligned(
                    emb[a], emb[b], b_al, tid[a], tid[b], pos_b
                ):
                    print(line)
        print()


if __name__ == "__main__":
    main()
