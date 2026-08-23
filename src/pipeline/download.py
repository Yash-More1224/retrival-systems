"""Q1.1 -- Download raw data files for MIND-small and EB-NeRD demo/small.

Idempotent: skips a file if it's already present and its recorded SHA256 in
MANIFEST.json matches. Designed to run unattended on a fresh checkout (e.g.
on the remote GPU node) as the first step of `make data`.

Usage:
    python -m src.pipeline.download --datasets mind ebnerd
    python -m src.pipeline.download --datasets mind ebnerd --include-testset
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.config import load_config, resolve_path

# URLs verified against the assignment PDF and (for ebnerd_testset.zip) a live
# HEAD request on 2026-08-20 -- see SPEC.md Q1.1.
MIND_FILES = {
    "MINDsmall_train.zip": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip",
    "MINDsmall_dev.zip": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip",
}

EBNERD_FILES = {
    "ebnerd_demo.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
    "ebnerd_small.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
    "Ekstra_Bladet_word2vec.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip",
}

# Only needed for the Q5 Codabench submission, not for offline dev -- opt in
# with --include-testset since it's 1.5GB.
EBNERD_TESTSET_FILES = {
    "ebnerd_testset.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip",
}

CHUNK_SIZE = 1 << 20  # 1 MiB
READ_TIMEOUT_SEC = 30  # a connection that goes idle this long is treated as stalled, not hung forever
MAX_RETRIES = 30  # generous -- a flaky route can need many reconnects to land 1.5GB

# Large files (ebnerd_testset.zip, 1.6GB) get split across N concurrent
# connections instead of one. Found empirically (2026-08-22/23): a single
# stream to ebnerd-dataset.s3.eu-west-1.amazonaws.com crawled at ~15-30KB/s
# with frequent stalls -- reproduced from two independent networks (a home
# connection and the `ada` remote node), while a same-machine test to a
# different host (Cloudflare) hit 1.5MB/s easily. That rules out "the local
# link is slow" and points at a per-connection cap (loss/RTT limiting a
# single TCP stream's window) rather than a saturated pipe -- exactly the
# failure mode multi-connection downloaders exist for.
PARALLEL_THRESHOLD_BYTES = 200_000_000
PARALLEL_CONNECTIONS = 24
MIN_SEGMENT_BYTES = 5_000_000  # don't split a connection's share below ~5MB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_token() -> str | None:
    """MIND (yjw1029/MIND) is a GATED HuggingFace dataset: being logged in via
    `hf auth login` alone isn't enough -- your account also needs to have
    clicked "Agree and access repository" on the dataset's page, AND the
    download request needs an `Authorization: Bearer <token>` header, which
    plain urllib never sends on its own (this is what actually caused the
    401 the first time -- see conversation). Reads the token the same way
    huggingface_hub does: HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars first,
    then the token file `hf auth login` writes to disk.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token
    return None


def _auth_headers(url: str) -> dict[str, str]:
    headers = {}
    if "huggingface.co" in url:
        token = _hf_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def _raise_for_http_error(e: urllib.error.HTTPError, url: str) -> None:
    if e.code == 401 and "huggingface.co" in url:
        raise RuntimeError(
            f"401 Unauthorized downloading {url}. MIND (yjw1029/MIND) is a GATED "
            "HuggingFace dataset -- two separate things are required, not just "
            "`hf auth login`: (1) visit https://huggingface.co/datasets/yjw1029/MIND "
            "and click 'Agree and access repository' with the SAME account you logged "
            "in with; (2) a valid token must be present at ~/.cache/huggingface/token "
            "or in $HF_TOKEN. Retry after (1) -- access approval can take a few minutes."
        ) from e
    raise e


def _content_length(url: str, headers: dict[str, str]) -> int:
    req = urllib.request.Request(url, method="HEAD")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=READ_TIMEOUT_SEC) as resp:
        return int(resp.headers.get("Content-Length", 0))


# ---------------------------------------------------------------------------
# Sequential path -- used for files under PARALLEL_THRESHOLD_BYTES.
# ---------------------------------------------------------------------------

def _download_attempt(url: str, tmp: Path, headers: dict[str, str]) -> bool:
    """One connection attempt, resuming from tmp's current size via Range if it
    already has bytes. Returns True if the whole remaining body was written.
    Raises socket.timeout/URLError if the connection stalls or drops mid-transfer
    (caller retries); re-raises HTTPError for the caller to classify.
    """
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url)
    for k, v in headers.items():
        req.add_header(k, v)
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")

    with urllib.request.urlopen(req, timeout=READ_TIMEOUT_SEC) as resp:
        resuming = resume_from and resp.status == 206
        if resume_from and not resuming:
            # Server ignored the Range request (200 OK, full body) -- can't
            # append a fresh full stream onto existing bytes, start clean.
            resume_from = 0
        total = int(resp.headers.get("Content-Length", 0)) + resume_from
        written = resume_from
        mode = "ab" if resuming else "wb"
        if resuming:
            print(f"    resuming from {resume_from / 1e6:.1f}MB")
        with open(tmp, mode) as out:
            while chunk := resp.read(CHUNK_SIZE):
                out.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r    {written / 1e6:.1f}MB / {total / 1e6:.1f}MB ({pct:.0f}%)", end="")
    print()
    return not total or written >= total


def _sequential_download(url: str, dest: Path, headers: dict[str, str]) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    while True:
        attempt += 1
        try:
            done = _download_attempt(url, tmp, headers)
        except urllib.error.HTTPError as e:
            if e.code == 416:  # Range Not Satisfiable -- .part is already complete or corrupt
                tmp.unlink()
                raise RuntimeError(f"{tmp} rejected on resume (416); deleted it -- rerun to restart the download") from e
            _raise_for_http_error(e, url)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
            if attempt >= MAX_RETRIES:
                raise RuntimeError(
                    f"gave up after {attempt} attempts downloading {url} -- last error: {e}. "
                    f"{tmp} has {tmp.stat().st_size / 1e6 if tmp.exists() else 0:.1f}MB, rerun to resume."
                ) from e
            got = tmp.stat().st_size / 1e6 if tmp.exists() else 0
            print(f"\n    stalled/dropped ({e}) at {got:.1f}MB -- reconnecting (attempt {attempt + 1}/{MAX_RETRIES})")
            continue
        if done:
            break
    tmp.rename(dest)


# ---------------------------------------------------------------------------
# Parallel path -- used for files >= PARALLEL_THRESHOLD_BYTES. Splits the
# not-yet-downloaded portion of the file into PARALLEL_CONNECTIONS byte
# ranges downloaded concurrently, each with its own stall-retry loop (same
# idea as the sequential path, scoped to a range). Progress persists in a
# `.chunks.json` sidecar as a list of "done" byte intervals so that:
#   (a) a full process restart resumes every connection from where it left
#       off, not from zero, and
#   (b) PARALLEL_CONNECTIONS can be changed between runs (e.g. bumped up for
#       more throughput) -- the remaining "holes" are recomputed from the
#       done-intervals and freely re-split into however many connections are
#       requested now, rather than being locked to whatever count the file
#       was originally split into.
# ---------------------------------------------------------------------------

def _merge_intervals(intervals: list[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _holes(done: list[list[int]], total: int) -> list[tuple[int, int]]:
    holes = []
    pos = 0
    for s, e in done:
        if s > pos:
            holes.append((pos, s))
        pos = max(pos, e)
    if pos < total:
        holes.append((pos, total))
    return holes


def _split_holes(holes: list[tuple[int, int]], target_n: int) -> list[list[int]]:
    """Subdivides `holes` into ~target_n pieces total, sized proportionally to
    each hole's share of the remaining bytes, never below MIN_SEGMENT_BYTES.
    """
    total_bytes = sum(e - s for s, e in holes)
    if total_bytes <= 0:
        return []
    segments: list[list[int]] = []
    for s, e in holes:
        size = e - s
        n_here = max(1, round(target_n * size / total_bytes))
        n_here = min(n_here, max(1, size // MIN_SEGMENT_BYTES))
        piece = size // n_here
        for j in range(n_here):
            ps = s + j * piece
            pe = e if j == n_here - 1 else ps + piece
            segments.append([ps, pe, 0])  # [start, end, written]
    return segments


def _download_chunk(
    url: str, fd: int, seg: list[int], headers: dict[str, str], stop_event: threading.Event,
) -> None:
    start, end = seg[0], seg[1]
    attempt = 0
    while start + seg[2] < end:
        if stop_event.is_set():
            return
        req = urllib.request.Request(url)
        for k, v in headers.items():
            req.add_header(k, v)
        req.add_header("Range", f"bytes={start + seg[2]}-{end - 1}")
        try:
            with urllib.request.urlopen(req, timeout=READ_TIMEOUT_SEC) as resp:
                while chunk := resp.read(CHUNK_SIZE):
                    os.pwrite(fd, chunk, start + seg[2])
                    seg[2] += len(chunk)  # single-writer per segment -- GIL makes this safe
            attempt = 0
        except urllib.error.HTTPError as e:
            stop_event.set()
            _raise_for_http_error(e, url)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
            attempt += 1
            if attempt >= MAX_RETRIES:
                stop_event.set()
                raise RuntimeError(f"segment [{start},{end}) gave up after {attempt} attempts: {e}") from e


def _load_done_intervals(state_path: Path, tmp: Path, total: int) -> list[list[int]]:
    if not state_path.exists():
        # No state file. If a .part exists it's either untouched or a leftover
        # from the old sequential path (a valid prefix written from byte 0).
        existing = tmp.stat().st_size if tmp.exists() else 0
        return [[0, existing]] if existing else []

    state = json.loads(state_path.read_text())
    if state.get("total") != total:
        return []  # source file changed size since -- can't trust any of this

    done = [tuple(iv) for iv in state.get("done", [])]
    for seg in state.get("segments", []):
        s, e, w = seg
        if w > 0:
            done.append((s, s + w))
    if "progress" in state:  # migrate from the older fixed-connection-count format
        n = len(state["progress"])
        size = total // n
        for i, w in enumerate(state["progress"]):
            if w > 0:
                s = i * size
                done.append((s, s + w))
    return _merge_intervals(done)


def _parallel_download(url: str, dest: Path, headers: dict[str, str], total: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    state_path = dest.with_suffix(dest.suffix + ".chunks.json")

    done = _load_done_intervals(state_path, tmp, total)
    holes = _holes(done, total)
    segments = _split_holes(holes, PARALLEL_CONNECTIONS)

    done_bytes = sum(e - s for s, e in done)
    print(f"    {done_bytes / 1e6:.1f}MB already done, splitting remaining "
          f"{(total - done_bytes) / 1e6:.1f}MB into {len(segments)} parallel connections")

    with open(tmp, "ab") as f:
        f.truncate(total)

    def flush_state() -> None:
        state_path.write_text(json.dumps({"total": total, "done": done, "segments": segments}))

    fd = os.open(tmp, os.O_RDWR)
    stop_event = threading.Event()
    try:
        if segments:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(segments)) as ex:
                futures = [ex.submit(_download_chunk, url, fd, seg, headers, stop_event) for seg in segments]
                while not all(fut.done() for fut in futures):
                    time.sleep(2)
                    written = done_bytes + sum(seg[2] for seg in segments)
                    pct = 100 * written / total
                    print(f"\r    {written / 1e6:.1f}MB / {total / 1e6:.1f}MB ({pct:.0f}%) "
                          f"across {len(segments)} connections", end="")
                    flush_state()
                print()
                for fut in futures:
                    fut.result()  # re-raises the first fatal segment error, if any
    finally:
        os.close(fd)

    flush_state()
    total_written = done_bytes + sum(seg[2] for seg in segments)
    if total_written < total:
        raise RuntimeError("parallel download exited without completing all segments")
    tmp.rename(dest)
    state_path.unlink(missing_ok=True)


def download_file(url: str, dest: Path) -> None:
    """Streams url -> dest via a .part file. Resumes from wherever a previous
    attempt left off instead of restarting from zero, auto-reconnects on a
    stalled/dropped connection, and uses several parallel connections for
    large files where a single stream is the bottleneck (see
    PARALLEL_THRESHOLD_BYTES above for why).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest}")
    headers = _auth_headers(url)

    total = 0
    try:
        total = _content_length(url, headers)
    except urllib.error.HTTPError as e:
        _raise_for_http_error(e, url)
    except (TimeoutError, urllib.error.URLError, ConnectionError):
        pass  # fall back to the sequential path, which discovers size from the GET itself

    if total >= PARALLEL_THRESHOLD_BYTES:
        _parallel_download(url, dest, headers, total)
    else:
        _sequential_download(url, dest, headers)


def ensure_file(name: str, url: str, raw_dir: Path, manifest: dict) -> None:
    dest = raw_dir / name
    recorded = manifest.get(name, {}).get("sha256")

    if dest.exists() and recorded:
        actual = sha256_of(dest)
        if actual == recorded:
            print(f"  [skip] {name} already present, checksum matches")
            return
        print(f"  [warn] {name} checksum mismatch (recorded={recorded[:12]}.. actual={actual[:12]}..), re-downloading")
    elif dest.exists() and not recorded:
        print(f"  [info] {name} present but unrecorded, computing checksum")
        manifest[name] = {"url": url, "sha256": sha256_of(dest), "bytes": dest.stat().st_size}
        return

    download_file(url, dest)
    manifest[name] = {"url": url, "sha256": sha256_of(dest), "bytes": dest.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"], choices=["mind", "ebnerd"])
    parser.add_argument("--include-testset", action="store_true", help="also fetch ebnerd_testset.zip (1.5GB, needed only for Q5)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    raw_dir = resolve_path(cfg, "raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    files: dict[str, str] = {}
    if "mind" in args.datasets:
        files.update(MIND_FILES)
    if "ebnerd" in args.datasets:
        files.update(EBNERD_FILES)
        if args.include_testset:
            files.update(EBNERD_TESTSET_FILES)

    print(f"Ensuring {len(files)} raw file(s) in {raw_dir}")
    for name, url in files.items():
        ensure_file(name, url, raw_dir, manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
