"""
Compute reactive force statistics from push_arc experiment params.

Reactive force = norm_reactive_force * max_force (from params.json).
Stats are computed per experiment class (e.g. 8pt8kg, 12pt5kg) and saved to
the analysis folder.
"""

from pathlib import Path
import json
import statistics

SAVES_DIR = Path(__file__).resolve().parents[1] / "experiment_logs" / "Saves"
ANALYSIS_DIR = Path(__file__).resolve().parent


def load_reactive_forces_by_class() -> dict[str, list[float]]:
    """Load reactive force (norm_reactive_force * max_force) from push_arc params per class."""
    by_class: dict[str, list[float]] = {}

    if not SAVES_DIR.is_dir():
        return by_class

    for class_dir in sorted(SAVES_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        by_class[class_name] = []

        for run_dir in sorted(class_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("push_arc"):
                continue
            params_path = run_dir / "params.json"
            if not params_path.is_file():
                continue
            try:
                with open(params_path) as f:
                    params = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            norm = params.get("norm_reactive_force")
            max_f = params.get("max_force")
            if norm is None or max_f is None:
                continue
            by_class[class_name].append(float(norm) * float(max_f))

    return {k: v for k, v in by_class.items() if v}


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, std, SEM, min, max, median, and quartiles."""
    n = len(values)
    if n == 0:
        return {}
    sorted_vals = sorted(values)
    mean = statistics.mean(values)
    try:
        stdev = statistics.stdev(values)
    except statistics.StatisticsError:
        stdev = 0.0
    sem = stdev / (n ** 0.5) if n else 0.0
    low = min(values)
    high = max(values)
    median = statistics.median(values)
    q1 = sorted_vals[(n - 1) // 4] if n >= 1 else low
    q3 = sorted_vals[3 * (n - 1) // 4] if n >= 1 else high
    iqr = q3 - q1

    return {
        "n": n,
        "mean": mean,
        "std": stdev,
        "sem": sem,
        "min": low,
        "max": high,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
    }


def format_report(class_name: str, stats: dict[str, float]) -> str:
    """Format a single-class stats report as text."""
    lines = [
        f"Reactive force statistics — {class_name}",
        "=" * 50,
        "",
        "Reactive force = norm_reactive_force * max_force (N).",
        "",
        f"  n (samples):     {stats['n']}",
        f"  mean:            {stats['mean']:.2f} N",
        f"  std:             {stats['std']:.2f} N",
        f"  SEM:             {stats['sem']:.2f} N",
        f"  min:             {stats['min']:.2f} N",
        f"  max:             {stats['max']:.2f} N",
        f"  median:          {stats['median']:.2f} N",
        f"  Q1:              {stats['q1']:.2f} N",
        f"  Q3:              {stats['q3']:.2f} N",
        f"  IQR:             {stats['iqr']:.2f} N",
        "",
        "Mean ± SEM and mean ± std are commonly reported.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    by_class = load_reactive_forces_by_class()
    if not by_class:
        print("No push_arc params found under Saves.")
        return

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    all_lines = [
        "Reactive force statistics (norm_reactive_force * max_force) per experiment class",
        "Only push_arc runs from robot/experiment_logs/Saves/<class>/push_arc_*/params.json",
        "=" * 70,
        "",
    ]

    for class_name in sorted(by_class.keys()):
        values = by_class[class_name]
        stats = compute_stats(values)
        report = format_report(class_name, stats)
        all_lines.append(report)

        # Per-class file: safe filename (e.g. 8pt8kg -> reactive_force_8pt8kg.txt)
        safe_name = class_name.replace("/", "_").replace("\\", "_")
        out_path = ANALYSIS_DIR / f"reactive_force_{safe_name}.txt"
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote {out_path}")

    # Summary file with all classes
    summary_path = ANALYSIS_DIR / "reactive_force_summary.txt"
    summary_path.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
