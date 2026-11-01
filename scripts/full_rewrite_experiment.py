#!/usr/bin/env python
"""
Does mixing full rewrites into diff-based evolution help? (issue #116)

Runs a grid of examples x full_rewrite_probability x seeds, then compares:
  - best score versus iteration, per probability
  - parse failures (no usable diff / code), evaluation failures and new-best
    rate, per evolution mode

Usage:
  # See what would run, with no LLM calls
  python scripts/full_rewrite_experiment.py run --dry-run

  # Run the grid (skips runs that already finished), then summarise
  python scripts/full_rewrite_experiment.py run --llm-config my_llm.yaml --iterations 100
  python scripts/full_rewrite_experiment.py analyze

--llm-config is a YAML file whose `llm:` section replaces each example's own, so
every example runs on the same model(s). API keys can stay in the environment
(OPENAI_API_KEY); keys in that file are never written to the output directory.

Each run happens in its own subprocess, since the controller sets up root-logger
handlers and worker globals that should not leak between runs. Outputs:
  <output-dir>/<example>/p<prob>_s<seed>/run.json   per-iteration records
  <output-dir>/report.md, curves.csv, best_score_vs_iteration.png (needs matplotlib)
"""

import argparse
import copy
import csv
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXAMPLES = {
    "function_minimization": REPO_ROOT / "examples/function_minimization/config.yaml",
    "circle_packing": REPO_ROOT / "examples/circle_packing/config_phase_1.yaml",
}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/full_rewrite_experiment"
MODES = ("diff", "full_rewrite")

# Iteration outcomes, in pipeline order
OK = "ok"  # evaluated without an error metric
EVAL_FAILURE = "eval_failure"  # parsed, but the evaluator reported an error
PARSE_FAILURE = "parse_failure"  # rejected by the worker before evaluation
REJECTED = "rejected"  # evaluated, then rejected by the database (e.g. novelty)
MISSING = "missing"  # no record (timeout, worker crash, shutdown)

# Logged by ProcessParallelController; rejections are not in the trace or the failures file
_REJECTED_RE = re.compile(r"^Database rejected program (\S+)")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def resolve_example(name_or_path: str) -> Tuple[str, Path]:
    """Map a known example name, or a path to an example's config, to (name, config path)"""
    if name_or_path in EXAMPLES:
        return name_or_path, EXAMPLES[name_or_path]
    path = Path(name_or_path).resolve()
    if not path.is_file():
        raise SystemExit(
            f"Unknown example {name_or_path!r}: not one of {sorted(EXAMPLES)} or a file"
        )
    return f"{path.parent.name}_{path.stem}", path


def run_dir_for(output_dir: Path, example: str, probability: float, seed: int) -> Path:
    return output_dir / example / f"p{probability:g}_s{seed}"


def build_run_config_dict(
    example_config: Path,
    probability: float,
    seed: int,
    iterations: int,
    run_dir: Path,
    llm_section: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The example's config with the experiment's settings applied"""
    import yaml

    with open(example_config) as f:
        raw = yaml.safe_load(f) or {}
    if llm_section is not None:
        raw["llm"] = copy.deepcopy(llm_section)

    raw["diff_based_evolution"] = True
    raw["full_rewrite_probability"] = probability
    raw["random_seed"] = seed
    raw["max_iterations"] = iterations
    raw["early_stopping_patience"] = None  # every run gets the same budget
    raw["evolution_trace"] = {
        "enabled": True,
        "format": "jsonl",
        "include_code": False,
        # Prompts and responses, so failures can be read later (they are in
        # evolution_trace_failures.jsonl; successful children's are in the trace)
        "include_prompts": True,
        "include_failures": True,
        "output_path": str(run_dir / "evolution_trace.jsonl"),
    }

    # Paths in the example config are relative to its own directory
    base = example_config.parent
    prompt = raw.get("prompt") or {}
    if prompt.get("template_dir") and not Path(prompt["template_dir"]).is_absolute():
        prompt["template_dir"] = str((base / prompt["template_dir"]).resolve())
    database = raw.get("database") or {}
    if database.get("island_initial_programs"):
        database["island_initial_programs"] = [
            str((base / p).resolve()) if p is not None and not Path(p).is_absolute() else p
            for p in database["island_initial_programs"]
        ]
    return raw


def redact(obj: Any) -> Any:
    """Copy of `obj` with api keys blanked, for writing to disk"""
    if isinstance(obj, dict):
        return {
            k: ("<redacted>" if "api_key" in str(k).lower() and v else redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


class _RejectionCollector(logging.Handler):
    """Collects children the database rejected, from the controller's log lines"""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.rejected_ids: List[str] = []

    def emit(self, record):
        match = _REJECTED_RE.match(record.getMessage())
        if match:
            self.rejected_ids.append(match.group(1))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _score(metrics: Optional[Dict[str, Any]], metric: str) -> Optional[float]:
    value = (metrics or {}).get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def run_one(
    example: str,
    example_config: Path,
    probability: float,
    seed: int,
    iterations: int,
    run_dir: Path,
    metric: str,
    llm_config: Optional[Path],
) -> Dict[str, Any]:
    """Run one evolution and write <run_dir>/run.json"""
    import asyncio

    import yaml

    from openevolve.config import Config
    from openevolve.controller import OpenEvolve
    from openevolve.process_parallel import _choose_evolution_mode

    llm_section = None
    if llm_config:
        with open(llm_config) as f:
            llm_section = (yaml.safe_load(f) or {}).get("llm")
        if not llm_section:
            raise SystemExit(f"{llm_config} has no 'llm:' section")

    run_dir.mkdir(parents=True, exist_ok=True)
    raw = build_run_config_dict(example_config, probability, seed, iterations, run_dir, llm_section)
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(redact(raw), f, sort_keys=False)
    trace_path = run_dir / "evolution_trace.jsonl"
    failures_path = run_dir / "evolution_trace_failures.jsonl"
    for path in (trace_path, failures_path):
        if path.exists():
            path.unlink()  # a rerun must not append to an old trace

    config = Config.from_dict(raw)
    collector = _RejectionCollector()
    logging.getLogger("openevolve.process_parallel").addHandler(collector)

    controller = OpenEvolve(
        str(example_config.parent / "initial_program.py"),
        str(example_config.parent / "evaluator.py"),
        config=config,
        output_dir=str(run_dir / "openevolve_output"),
    )
    started = time.time()
    asyncio.run(controller.run(iterations=iterations))
    wall_time = time.time() - started

    initial_scores = [
        s
        for p in controller.database.programs.values()
        if p.parent_id is None and (s := _score(p.metrics, metric)) is not None
    ]
    initial_score = max(initial_scores) if initial_scores else None

    traced = {int(e["iteration"]): e for e in _read_jsonl(trace_path)}
    failures = {int(e["iteration"]): e for e in _read_jsonl(failures_path)}
    rejected = set(collector.rejected_ids)

    records = []
    mode_mismatches = 0
    for iteration in range(1, iterations + 1):
        # Seeded runs pick the mode from (seed, iteration), so it is known even without a record
        expected_mode = _choose_evolution_mode(config, iteration)
        record = {"iteration": iteration, "mode": expected_mode, "score": None, "error": None}
        record["completion_tokens"] = None
        if iteration in traced:
            entry = traced[iteration]
            metadata = entry.get("metadata") or {}
            recorded_mode = metadata.get("evolution_mode") or expected_mode
            child_metrics = entry.get("child_metrics") or {}
            record["mode"] = recorded_mode
            record["score"] = _score(child_metrics, metric)
            if "error" in child_metrics or record["score"] is None:
                record["outcome"] = EVAL_FAILURE
                record["error"] = str(child_metrics.get("error", f"no finite {metric}"))
            else:
                record["outcome"] = OK
            usage = metadata.get("token_usage") or {}
            record["completion_tokens"] = usage.get("completion_tokens")
        elif iteration in failures:
            failure = failures[iteration]
            record["mode"] = failure.get("evolution_mode") or expected_mode
            record["outcome"] = PARSE_FAILURE
            record["error"] = failure.get("error")
            record["completion_tokens"] = (failure.get("token_usage") or {}).get(
                "completion_tokens"
            )
        else:
            record["outcome"] = MISSING
        if record["mode"] != expected_mode:
            mode_mismatches += 1
        records.append(record)
    # Database rejections are logged by program id only; count them without an iteration
    summary = {
        "example": example,
        "example_config": str(example_config),
        "full_rewrite_probability": probability,
        "seed": seed,
        "iterations": iterations,
        "metric": metric,
        "initial_score": initial_score,
        "wall_time_s": wall_time,
        "database_rejections": len(rejected),
        "mode_mismatches": mode_mismatches,
        "records": records,
    }
    with open(run_dir / "run.json", "w") as f:
        json.dump(summary, f, indent=2)
    if mode_mismatches:
        print(
            f"WARNING: {mode_mismatches} iteration(s) ran in a different mode than the seed "
            "predicts; check that random_seed reached the workers",
            file=sys.stderr,
        )
    return summary


def cmd_run(args) -> int:
    output_dir = Path(args.output_dir).resolve()
    grid = [
        (name, path, p, seed)
        for name, path in (resolve_example(e) for e in args.examples)
        for p in args.probabilities
        for seed in args.seeds
    ]
    for p in args.probabilities:
        if not 0.0 <= p <= 1.0:
            raise SystemExit(f"probability {p} is not between 0 and 1")

    todo, done = [], []
    for item in grid:
        run_json = run_dir_for(output_dir, item[0], item[2], item[3]) / "run.json"
        (done if run_json.exists() and not args.force else todo).append(item)

    print(
        f"{len(grid)} runs in the grid ({len(done)} already done, {len(todo)} to run), "
        f"{args.iterations} iterations each: up to {len(todo) * args.iterations} LLM calls"
    )
    for name, path, p, seed in todo:
        print(
            f"  {name:<24} p={p:<5g} seed={seed}  ({path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path})"
        )
    if args.dry_run or not todo:
        return 0

    failed = []
    for i, (name, path, p, seed) in enumerate(todo, 1):
        run_dir = run_dir_for(output_dir, name, p, seed)
        print(f"\n[{i}/{len(todo)}] {name} p={p:g} seed={seed} -> {run_dir}", flush=True)
        cmd = [
            sys.executable,
            __file__,
            "_run-one",
            "--example-name",
            name,
            "--example-config",
            str(path),
            "--probability",
            repr(p),
            "--seed",
            str(seed),
            "--iterations",
            str(args.iterations),
            "--run-dir",
            str(run_dir),
            "--metric",
            args.metric,
        ]
        if args.llm_config:
            cmd += ["--llm-config", str(Path(args.llm_config).resolve())]
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "stdout.log", "w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        if proc.returncode != 0:
            failed.append(run_dir)
            print(f"  FAILED (exit {proc.returncode}), see {run_dir / 'stdout.log'}")
    if failed:
        print(f"\n{len(failed)} run(s) failed; rerun the same command to retry them")
        return 1
    print(f"\nDone. Summarise with: python {Path(__file__).name} analyze --output-dir {output_dir}")
    return 0


def cmd_run_one(args) -> int:
    run_one(
        example=args.example_name,
        example_config=Path(args.example_config),
        probability=float(args.probability),
        seed=args.seed,
        iterations=args.iterations,
        run_dir=Path(args.run_dir),
        metric=args.metric,
        llm_config=Path(args.llm_config) if args.llm_config else None,
    )
    return 0


# ---------------------------------------------------------------------------
# Analysis (pure functions over run.json records, so they can be tested)
# ---------------------------------------------------------------------------


def best_so_far(run: Dict[str, Any]) -> List[Optional[float]]:
    """Best score after each iteration 0..N; index 0 is the initial program"""
    best = run.get("initial_score")
    curve = [best]
    for record in sorted(run["records"], key=lambda r: r["iteration"]):
        if record["outcome"] == OK and record["score"] is not None:
            best = record["score"] if best is None else max(best, record["score"])
        curve.append(best)
    return curve


def new_best_iterations(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Records that strictly beat everything before them, initial program included"""
    best = run.get("initial_score")
    found = []
    for record in sorted(run["records"], key=lambda r: r["iteration"]):
        if record["outcome"] == OK and record["score"] is not None:
            if best is None or record["score"] > best:
                found.append(record)
                best = record["score"]
    return found


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _empty_mode_stats() -> Dict[str, Any]:
    counts = {outcome: 0 for outcome in (OK, EVAL_FAILURE, PARSE_FAILURE, MISSING)}
    return {"attempts": 0, **counts, "new_best": 0, "completion_tokens": []}


def mode_stats(runs: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Outcome counts per evolution mode, pooled over `runs`"""
    stats = defaultdict(_empty_mode_stats, {mode: _empty_mode_stats() for mode in MODES})
    for run in runs:
        for record in run["records"]:
            s = stats[record["mode"]]
            s["attempts"] += 1
            s[record["outcome"]] += 1
            if record.get("completion_tokens") is not None:
                s["completion_tokens"].append(record["completion_tokens"])
        for record in new_best_iterations(run):
            stats[record["mode"]]["new_best"] += 1
    return stats


def failure_reason(error: Optional[str]) -> str:
    """Group error messages that differ only in numbers or quoted details"""
    reason = re.sub(r"\d+", "N", error or "unknown")
    return reason.split(":")[0].strip()[:90]


def failure_reasons(runs: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
    """(mode, reason) -> count, over parse and evaluation failures"""
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for run in runs:
        for record in run["records"]:
            if record["outcome"] in (PARSE_FAILURE, EVAL_FAILURE):
                counts[(record["mode"], failure_reason(record.get("error")))] += 1
    return dict(counts)


def summarise_curves(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Final best and mean best-so-far (area under the curve) across seeds"""
    curves = [best_so_far(r) for r in runs]
    finals = [c[-1] for c in curves if c[-1] is not None]
    aucs = [mean(v for v in c if v is not None) for c in curves if any(v is not None for v in c)]
    return {
        "n_runs": len(runs),
        "final_best": finals,
        "final_mean": mean(finals) if finals else None,
        "final_sd": stdev(finals) if len(finals) > 1 else None,
        "auc_mean": mean(aucs) if aucs else None,
        "curves": curves,
    }


def load_runs(output_dir: Path) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(output_dir.glob("*/*/run.json")):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return (
        "-"
        if value is None or (isinstance(value, float) and math.isnan(value))
        else f"{value:.{digits}f}"
    )


def _rate(count: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = wilson_interval(count, n)
    return f"{count}/{n} = {count / n:.0%} [{lo:.0%}, {hi:.0%}]"


def build_report(runs: List[Dict[str, Any]]) -> str:
    by_example: Dict[str, Dict[float, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        by_example[run["example"]][run["full_rewrite_probability"]].append(run)

    lines = ["# full_rewrite_probability experiment (issue #116)", ""]
    lines += [
        "Outcomes per iteration: **parse failure** = rejected before evaluation (no usable "
        "diff / code, diff did not apply, no change); **eval failure** = evaluated but the "
        "evaluator returned an error; **new best** = beat every earlier program in that run. "
        "Rates show a 95% Wilson interval. With few seeds, treat score differences smaller "
        "than the SD as noise.",
        "",
    ]
    mismatches = sum(r.get("mode_mismatches", 0) for r in runs)
    if mismatches:
        lines += [f"> **Warning:** {mismatches} iteration(s) ran in an unexpected mode.", ""]

    for example in sorted(by_example):
        cells = by_example[example]
        metric = next(iter(cells.values()))[0]["metric"]
        iterations = sorted({r["iterations"] for rs in cells.values() for r in rs})
        lines += [
            f"## {example}",
            "",
            f"Metric `{metric}`, {', '.join(map(str, iterations))} iterations per run.",
            "",
        ]
        lines += [
            "### Best score",
            "",
            "| p(full rewrite) | runs | final best, mean ± SD | per seed | mean best-so-far (AUC) |",
            "|---|---|---|---|---|",
        ]
        for p in sorted(cells):
            s = summarise_curves(cells[p])
            per_seed = ", ".join(_fmt(v) for v in s["final_best"])
            sd = f" ± {_fmt(s['final_sd'])}" if s["final_sd"] is not None else ""
            lines.append(
                f"| {p:g} | {s['n_runs']} | {_fmt(s['final_mean'])}{sd} | {per_seed} | {_fmt(s['auc_mean'])} |"
            )
        lines.append("")

        lines += [
            "### Outcomes per mode",
            "",
            "| p | mode | attempts | parse failure | eval failure | new best | mean completion tokens |",
            "|---|---|---|---|---|---|---|",
        ]
        for p in sorted(cells):
            stats = mode_stats(cells[p])
            for mode in MODES:
                st = stats[mode]
                n = st["attempts"]
                if n == 0:
                    continue
                evaluated = st[OK] + st[EVAL_FAILURE]
                tokens = mean(st["completion_tokens"]) if st["completion_tokens"] else None
                lines.append(
                    f"| {p:g} | {mode} | {n} | {_rate(st[PARSE_FAILURE], n)} | "
                    f"{_rate(st[EVAL_FAILURE], evaluated)} | {_rate(st['new_best'], n)} | {_fmt(tokens, 0)} |"
                )
        lines.append("")

        reasons = failure_reasons(r for rs in cells.values() for r in rs)
        if reasons:
            lines += [
                "### Why iterations failed (all runs)",
                "",
                "Responses are in each run's `evolution_trace_failures.jsonl`.",
                "",
                "| mode | reason | count |",
                "|---|---|---|",
            ]
            for (mode, reason), count in sorted(reasons.items(), key=lambda kv: (kv[0][0], -kv[1])):
                lines.append(f"| {mode} | {reason} | {count} |")
            lines.append("")

        mixed = [r for p, rs in cells.items() if 0 < p < 1 for r in rs]
        if mixed:
            stats = mode_stats(mixed)
            d, f = stats["diff"], stats["full_rewrite"]
            lines += [
                "### Within mixed runs",
                "",
                "Same runs, same populations, so this is the most direct diff-vs-rewrite comparison "
                f"(all runs with 0 < p < 1 pooled): new best per attempt, diff "
                f"{_rate(d['new_best'], d['attempts'])}, full rewrite {_rate(f['new_best'], f['attempts'])}.",
                "",
            ]
        missing = sum(st[MISSING] for p in cells for st in mode_stats(cells[p]).values())
        rejected = sum(r.get("database_rejections", 0) for rs in cells.values() for r in rs)
        if missing or rejected:
            lines += [
                f"{missing} iteration(s) had no record (timeout/crash); {rejected} child program(s) rejected by the database.",
                "",
            ]
    return "\n".join(lines)


def write_curves_csv(runs: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["example", "full_rewrite_probability", "seed", "iteration", "best_so_far"])
        for run in runs:
            for i, value in enumerate(best_so_far(run)):
                writer.writerow(
                    [run["example"], run["full_rewrite_probability"], run["seed"], i, value]
                )


# Categorical slots 1-3 of the dataviz reference palette (light surface)
_SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]


def plot_curves(runs: List[Dict[str, Any]], path: Path) -> bool:
    """Small multiples, one per example: mean best-so-far per p, band = min..max over seeds"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    by_example: Dict[str, Dict[float, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        by_example[run["example"]][run["full_rewrite_probability"]].append(run)
    probabilities = sorted({r["full_rewrite_probability"] for r in runs})
    color = {p: _SERIES_COLORS[i % len(_SERIES_COLORS)] for i, p in enumerate(probabilities)}

    examples = sorted(by_example)
    fig, axes = plt.subplots(1, len(examples), figsize=(5.5 * len(examples), 4), squeeze=False)
    for ax, example in zip(axes[0], examples):
        for p in sorted(by_example[example]):
            curves = [best_so_far(r) for r in by_example[example][p]]
            length = min(len(c) for c in curves)
            xs, means, lows, highs = [], [], [], []
            for i in range(length):
                values = [c[i] for c in curves if c[i] is not None]
                if values:
                    xs.append(i)
                    means.append(mean(values))
                    lows.append(min(values))
                    highs.append(max(values))
            if not xs:
                continue
            # Best-so-far only changes at an iteration, so draw it as steps. No direct labels:
            # curves often coincide (no improvement yet), and labels would collide
            label = f"p={p:g} (n={len(curves)})"
            ax.plot(xs, means, color=color[p], linewidth=2, drawstyle="steps-post", label=label)
            ax.fill_between(xs, lows, highs, color=color[p], alpha=0.12, linewidth=0, step="post")
        ax.set_title(example, fontsize=11, color="#1a1a19", loc="left")
        ax.set_xlabel("iteration", color="#5f5e5a")
        ax.set_ylabel(
            f"best {by_example[example][next(iter(by_example[example]))][0]['metric']} so far",
            color="#5f5e5a",
        )
        ax.grid(True, color="#e5e4df", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors="#5f5e5a", labelsize=8)
        ax.legend(
            frameon=False,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(0, -0.16),
            ncol=len(by_example[example]),
        )
    fig.suptitle(
        "Best score vs iteration, mean over seeds (band: min-max)", fontsize=11, x=0.01, ha="left"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def cmd_analyze(args) -> int:
    output_dir = Path(args.output_dir).resolve()
    runs = load_runs(output_dir)
    if not runs:
        print(f"No run.json files under {output_dir}")
        return 1
    report = build_report(runs)
    (output_dir / "report.md").write_text(report)
    write_curves_csv(runs, output_dir / "curves.csv")
    plotted = plot_curves(runs, output_dir / "best_score_vs_iteration.png")
    print(report)
    print(f"\nWrote {output_dir / 'report.md'} and {output_dir / 'curves.csv'}")
    if plotted:
        print(f"Wrote {output_dir / 'best_score_vs_iteration.png'}")
    else:
        print("matplotlib not installed: skipped the plot (pip install matplotlib)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the grid (skips finished runs)")
    run.add_argument(
        "--examples",
        nargs="+",
        default=list(EXAMPLES),
        help=f"example names ({', '.join(EXAMPLES)}) or paths to example config files",
    )
    run.add_argument("--probabilities", nargs="+", type=float, default=[0.0, 0.2, 0.5])
    run.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    run.add_argument("--iterations", type=int, default=100)
    run.add_argument("--llm-config", help="YAML file whose llm: section replaces each example's")
    run.add_argument("--metric", default="combined_score")
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run.add_argument("--dry-run", action="store_true", help="print the grid without running")
    run.add_argument("--force", action="store_true", help="rerun runs that already finished")
    run.set_defaults(func=cmd_run)

    one = sub.add_parser("_run-one", help=argparse.SUPPRESS)
    one.add_argument("--example-name", required=True)
    one.add_argument("--example-config", required=True)
    one.add_argument("--probability", required=True)
    one.add_argument("--seed", type=int, required=True)
    one.add_argument("--iterations", type=int, required=True)
    one.add_argument("--run-dir", required=True)
    one.add_argument("--metric", default="combined_score")
    one.add_argument("--llm-config")
    one.set_defaults(func=cmd_run_one)

    analyze = sub.add_parser("analyze", help="summarise finished runs")
    analyze.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
