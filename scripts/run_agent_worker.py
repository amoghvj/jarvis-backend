"""Standalone CLI entrypoint for the J.A.R.V.I.S. durable agent worker.

Usage:
    python -m scripts.run_agent_worker
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.worker import run_worker_loop  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the J.A.R.V.I.S. background investigation agent worker.")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.5,
        help="Seconds to sleep when no eligible agent runs are in the queue (default: 1.5)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional maximum number of runs to execute before exiting",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_worker_loop(
        poll_interval_seconds=args.poll_interval,
        max_runs=args.max_runs,
    )


if __name__ == "__main__":
    main()

