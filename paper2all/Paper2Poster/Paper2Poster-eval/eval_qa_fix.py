#!/usr/bin/env python3
"""
Run eval_poster_pipeline.py for every sub-folder in poster_sum_100,
using up to 10 threads.  poster_method and fix are now taken from
command-line arguments.

Example:
    python run_eval_threads.py \
        --poster_method poster_sum_50 \
        --fix llama-3-70b-vl

"""
import argparse
import concurrent.futures as cf
import pathlib
import signal
import subprocess
import sys

BASE_DIR     = pathlib.Path("poster_sum_100")   # directory holding the papers                           # number of worker threads

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run eval_poster_pipeline.py concurrently on all papers."
)
parser.add_argument(
    "--poster_method",
    default="poster_sum_100",
    help="Name of the poster-generation method to evaluate (default: %(default)s)",
)
parser.add_argument(
    "--fix",
    default="qwen-2.5-vl-72b",
    help="Value to pass to --fix in eval_poster_pipeline.py (default: %(default)s)",
)

parser.add_argument(
    '--max_workers',
    type=int,
    default=1,
)

parser.add_argument('--del_model_name', type=str)
args = parser.parse_args()
# ───────────────────────────────────────────────────────────────────────────────


MAX_WORKERS = args.max_workers

def run_pipeline(subfolder: str, poster_method: str, fix: str) -> None:
    """Invoke eval_poster_pipeline.py for a single paper."""
    cmd = [
        "python",
        "eval_poster_pipeline.py",
        "--paper_name",
        subfolder,
        "--poster_method",
        poster_method,
        "--poster_image_name",
        "poster.png",
        "--metric",
        "qa",
        "--fix",
        fix,
    ]
    if args.del_model_name:
        cmd += ["--del_model_name", args.del_model_name]
    subprocess.run(cmd, check=True)


MAX_RETRIES = 50

def run_with_retries(folder: str, poster_method, fix) -> None:
    """
    Tries to run_pipeline up to MAX_RETRIES times before giving up.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            run_pipeline(folder, poster_method, fix)
            return
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"⚠️  {folder}: attempt {attempt} failed ({e!r}), retrying…")
            else:
                # Last attempt also failed, re-raise so the pool will catch it
                raise

def main() -> None:
    folders = sorted(p.name for p in BASE_DIR.iterdir() if p.is_dir())

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_with_retries, f, args.poster_method, args.fix): f
            for f in folders
        }
        for fut in cf.as_completed(futures):
            paper = futures[fut]
            try:
                fut.result()
                print(f"✓ {paper} done")
            except Exception as e:
                print(f"✗ {paper} failed after {MAX_RETRIES} attempts: {e}", file=sys.stderr)
# ── Graceful shutdown on Ctrl-C / SIGTERM ──────────────────────────────────────
def _handle_signal(signum, frame):
    print("\nReceived signal, shutting down…", file=sys.stderr)
    sys.exit(1)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()