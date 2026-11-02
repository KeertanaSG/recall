"""Compare saved retrieval runs.

    uv run python evals/compare.py              # the latest 20 runs
    uv run python evals/compare.py --config agent
    uv run python evals/compare.py --subjects   # adds pass counts per subject
"""

import argparse
import json
from pathlib import Path

RUNS = Path(__file__).parent / "results" / "runs.jsonl"


def seconds(value) -> str:
    return "-" if value is None else f"{value:.1f}"


def load_runs(path: Path = RUNS) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="Only show configs containing this text")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--subjects", action="store_true")
    args = parser.parse_args()

    runs = [run for run in load_runs() if args.config in run["config"]][-args.limit :]
    if not runs:
        print("No saved runs yet.")
        return

    print(f"{'date':10} {'config':18} {'rows':>5} {'found':>6} {'absent':>6} {'absent s':>8} "
          f"{'median s':>8} {'slowest':>7} {'graph':>5} {'tokens':>8}  label")
    for run in runs:
        tokens = run.get("tokens")
        total = str(tokens["input"] + tokens["output"]) if tokens else "-"
        graph = run.get("graph_used")
        print(
            f"{run['created_at'][:10]:10} {run['config']:18} {run['rows']:>5} "
            f"{run['answerable_passed']}/{run['answerable_total']:<4} "
            f"{run['absent_passed']}/{run['absent_total']:<4} "
            f"{seconds(run.get('absent_mean_s')):>8} {seconds(run.get('median_s')):>8} "
            f"{seconds(run.get('slowest_s')):>7} {'-' if graph is None else graph:>5} "
            f"{total:>8}  {run.get('label', '')}"
        )
    if args.subjects:
        names = list(runs[-1]["subjects"])
        print(f"\n{'config':18} " + " ".join(f"{name:>8}" for name in names))
        for run in runs:
            cells = []
            for name in names:
                stats = run["subjects"].get(name)
                cells.append(f"{stats['passed']}/{stats['total']}" if stats else "-")
            print(f"{run['config']:18} " + " ".join(f"{cell:>8}" for cell in cells))


if __name__ == "__main__":
    main()
