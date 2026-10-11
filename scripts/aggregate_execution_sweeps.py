"""Aggregate paired execution outcomes across independently downloaded market dates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def circular_block_sample(values: list[float], block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Resample a time series with circular blocks, preserving local dependence."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or block_size < 1:
        raise ValueError("values must be nonempty 1-D and block_size positive")
    out: list[float] = []
    while len(out) < len(x):
        start = int(rng.integers(len(x)))
        take = min(block_size, len(x) - len(out))
        out.extend(x[(start + np.arange(take)) % len(x)].tolist())
    return np.asarray(out[:len(x)], dtype=np.float64)


def cross_date_block_bootstrap_ci(
    daily_improvements: list[list[float]],
    seed: int = 7,
    n_boot: int = 5000,
    block_size: int = 5,
) -> tuple[float, float]:
    """Bootstrap dates and adjacent episode blocks without linking different dates."""
    days = [np.asarray(x, dtype=np.float64) for x in daily_improvements if len(x)]
    if not days or any(not np.isfinite(x).all() for x in days):
        raise ValueError("at least one nonempty finite date series is required")
    if n_boot < 100 or block_size < 1:
        raise ValueError("n_boot must be >= 100 and block_size positive")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        selected_days = rng.integers(0, len(days), size=len(days))
        means = []
        for di in selected_days:
            sample = circular_block_sample(days[int(di)].tolist(), block_size, rng)
            means.append(float(sample.mean()))
        estimates[b] = float(np.mean(means))
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_reports(sweep_dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    reports, manifests = [], []
    for sweep_dir in sweep_dirs:
        manifest_path = sweep_dir / "sweep_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"sweep manifest missing: {manifest_path}")
        manifest = load_json(manifest_path)
        manifests.append({
            "directory": str(sweep_dir),
            "dataset_label": manifest.get("dataset_label", "unknown"),
            "l2_sha256": manifest.get("l2_sha256"),
            "trades_sha256": manifest.get("trades_sha256"),
            "fi_sha256": manifest.get("fi_sha256"),
            "source_commit": manifest.get("source_commit"),
            "workflow_commit": manifest.get("workflow_commit"),
        })
        for run in manifest.get("runs", []):
            if run.get("evaluation", {}).get("return_code") != 0:
                continue
            path = Path(run.get("evaluation_report", ""))
            if not path.exists():
                path = sweep_dir / path.name
            if not path.exists():
                continue
            data = load_json(path)
            if data.get("status") != "passed":
                continue
            if data.get("dataset_label", "unknown") == "unknown":
                data["dataset_label"] = manifest.get("dataset_label", "unknown")
            reports.append(data)
    return reports, manifests


def aggregate_cross_date(reports: list[dict], target_bps: float = 2.4,
                         min_dates: int = 3, n_boot: int = 5000) -> dict:
    """Average seeds within a market window, then compare paired windows to TWAP."""
    seed_maps: dict[tuple[str, str, str], dict[int, dict[str, dict]]] = {}
    seen_labels = set()
    for report in reports:
        label = str(report.get("dataset_label", "unknown"))
        if label == "unknown":
            label = str(report.get("l2_window_start", "unknown"))[:10]
        seen_labels.add(label)
        side = str(report["side"])
        seed = int(report["seed"])
        for episode in report.get("episode_results", []):
            start, end = episode.get("l2_start"), episode.get("l2_end")
            if start is None or end is None:
                continue
            key = f"{start}|{end}"
            baseline = float(episode["twap_slippage_bps"])
            for method, result in episode.get("methods", {}).items():
                group = seed_maps.setdefault((label, side, method), {}).setdefault(seed, {})
                group[key] = {
                    "baseline": baseline,
                    "slippage": float(result["slippage_bps"]),
                    "completion": float(result.get("completion", 0.0)),
                }

    daily_groups: dict[tuple[str, str], list[dict]] = {}
    for (label, side, method), seeds in seed_maps.items():
        valid = {seed: data for seed, data in seeds.items() if data}
        if not valid:
            continue
        common = set.intersection(*(set(data) for data in valid.values()))
        if not common:
            continue
        keys = sorted(common)
        improvements, completion = [], []
        for key in keys:
            baseline = float(np.mean([valid[seed][key]["baseline"] for seed in valid]))
            policy = float(np.mean([valid[seed][key]["slippage"] for seed in valid]))
            completion.append(float(np.mean([valid[seed][key]["completion"] for seed in valid])))
            improvements.append(baseline - policy)
        daily_groups.setdefault((side, method), []).append({
            "dataset_label": label,
            "seed_count": len(valid),
            "episodes": len(keys),
            "mean_improvement_bps": float(np.mean(improvements)),
            "median_improvement_bps": float(np.median(improvements)),
            "mean_completion": float(np.mean(completion)),
            "improvements": improvements,
        })

    summary = {}
    for (side, method), days in sorted(daily_groups.items()):
        days.sort(key=lambda x: x["dataset_label"])
        daily_means = [d["mean_improvement_bps"] for d in days]
        daily_series = [d["improvements"] for d in days]
        equal_day_mean = float(np.mean(daily_means))
        episode_weighted = float(
            sum(sum(d["improvements"]) for d in days) /
            max(1, sum(len(d["improvements"]) for d in days))
        )
        ci = cross_date_block_bootstrap_ci(daily_series, n_boot=n_boot)
        summary.setdefault(side, {})[method] = {
            "independent_dates": len(days),
            "dates": [{k: v for k, v in d.items() if k != "improvements"} for d in days],
            "mean_improvement_equal_date_bps": equal_day_mean,
            "mean_improvement_episode_weighted_bps": episode_weighted,
            "cross_date_paired_bootstrap_ci95_bps": [ci[0], ci[1]],
            "mean_completion_across_dates": float(np.mean([d["mean_completion"] for d in days])),
            "target_bps": float(target_bps),
            "target_2_4_bps_achieved": bool(len(days) >= min_dates and equal_day_mean >= target_bps),
            "within_10pct_of_target": bool(len(days) >= min_dates and equal_day_mean >= 0.9 * target_bps),
            "positive_lower_confidence_bound": bool(ci[0] > 0),
            "accept_as_target_evidence": bool(
                len(days) >= min_dates and equal_day_mean >= target_bps and ci[0] > 0
                and all(d["mean_completion"] >= 0.99 and d["seed_count"] >= 3 for d in days)
            ),
        }
    return {
        "status": "completed",
        "method": "paired TWAP minus policy slippage; model-seed results averaged per market window",
        "uncertainty": "outer bootstrap over dates plus circular moving-block bootstrap within each date",
        "required_independent_dates": int(min_dates),
        "observed_independent_dates": len(seen_labels),
        "target_bps": float(target_bps),
        "aggregate": summary,
    }


def render_markdown(result: dict, manifests: list[dict]) -> str:
    lines = [
        "# FlowOE Cross-Date Execution Sweep",
        "",
        f"- Independent data labels observed: {result['observed_independent_dates']}",
        f"- Required independent dates: {result['required_independent_dates']}",
        f"- Target improvement: {result['target_bps']:.2f} bps",
        f"- Uncertainty: {result['uncertainty']}",
        "",
        "## Per-date results",
        "",
        "| Side | Method | Date | Episodes | Mean improvement (bps) | Completion |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for side, methods in result["aggregate"].items():
        for method, data in methods.items():
            for day in data["dates"]:
                lines.append(
                    f"| {side} | {method} | {day['dataset_label']} | {day['episodes']} | "
                    f"{day['mean_improvement_bps']:.6f} | {day['mean_completion']:.4%} |"
                )
    lines.extend([
        "",
        "## Cross-date summary",
        "",
        "| Side | Method | Independent dates | Equal-date improvement (bps) | 95% CI (bps) | Completion | Target evidence accepted? |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for side, methods in result["aggregate"].items():
        for method, data in methods.items():
            ci = data["cross_date_paired_bootstrap_ci95_bps"]
            lines.append(
                f"| {side} | {method} | {data['independent_dates']} | "
                f"{data['mean_improvement_equal_date_bps']:.6f} | "
                f"[{ci[0]:.6f}, {ci[1]:.6f}] | {data['mean_completion_across_dates']:.4%} | "
                f"{data['accept_as_target_evidence']} |"
            )
    lines.extend(["", "## Provenance", ""])
    for m in manifests:
        lines.append(
            f"- {m['dataset_label']}: L2 SHA-256 {m['l2_sha256']}, trades SHA-256 {m['trades_sha256']}, "
            f"FI-2010 SHA-256 {m['fi_sha256']}, source commit {m['source_commit']}."
        )
    lines.extend([
        "",
        "A positive point estimate on one window is not sufficient. The 2.4-bps claim is accepted only with at least the required independent dates, a positive lower confidence bound, and mean completion of at least 99% on every date.",
        "",
    ])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dirs", nargs="+", required=True)
    ap.add_argument("--output", default="results/cross_date_execution.json")
    ap.add_argument("--markdown", default="results/CROSS_DATE_EXECUTION.md")
    ap.add_argument("--target-bps", type=float, default=2.4)
    ap.add_argument("--min-dates", type=int, default=3)
    ap.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = ap.parse_args()
    if not np.isfinite(args.target_bps) or args.target_bps <= 0 or args.min_dates < 2 or args.bootstrap_resamples < 100:
        ap.error("target-bps must be positive, min-dates >= 2, bootstrap-resamples >= 100")
    sweep_dirs = [Path(p).resolve() for p in args.sweep_dirs]
    reports, manifests = load_reports(sweep_dirs)
    result = aggregate_cross_date(reports, args.target_bps, args.min_dates, args.bootstrap_resamples)
    result["inputs"] = manifests
    out = Path(args.output)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = Path(args.markdown)
    if not md.is_absolute():
        md = Path.cwd() / md
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_markdown(result, manifests), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "independent_dates": result["observed_independent_dates"],
        "output": str(out),
        "markdown": str(md),
    }, indent=2))
    for side, methods in result["aggregate"].items():
        for method, metrics in methods.items():
            if metrics["accept_as_target_evidence"]:
                print(f"TARGET EVIDENCE ACCEPTED: {side}/{method}")


if __name__ == "__main__":
    main()
