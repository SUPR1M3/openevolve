"""
Tests for the analysis half of scripts/full_rewrite_experiment.py (issue #116),
and for the config it builds for each run
"""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "full_rewrite_experiment.py"
_spec = importlib.util.spec_from_file_location("full_rewrite_experiment", SCRIPT)
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)


def _record(iteration, mode, outcome, score=None, tokens=None):
    return {
        "iteration": iteration,
        "mode": mode,
        "outcome": outcome,
        "score": score,
        "error": None,
        "completion_tokens": tokens,
    }


def _run(records, initial=0.5, p=0.5, seed=1, example="ex"):
    return {
        "example": example,
        "full_rewrite_probability": p,
        "seed": seed,
        "iterations": len(records),
        "metric": "combined_score",
        "initial_score": initial,
        "database_rejections": 0,
        "mode_mismatches": 0,
        "records": records,
    }


RUN = _run(
    [
        _record(1, "diff", exp.OK, 0.4, 10),  # worse than initial
        _record(2, "full_rewrite", exp.OK, 0.7, 300),  # new best
        _record(3, "diff", exp.PARSE_FAILURE, tokens=12),
        _record(4, "full_rewrite", exp.EVAL_FAILURE, 0.0, 280),  # error metrics never count
        _record(5, "diff", exp.OK, 0.7, 14),  # ties are not a new best
        _record(6, "diff", exp.OK, 0.9, 16),  # new best
        _record(7, "full_rewrite", exp.MISSING),
    ]
)


class TestAnalysis(unittest.TestCase):
    def test_best_so_far(self):
        self.assertEqual(exp.best_so_far(RUN), [0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.9, 0.9])

    def test_best_so_far_without_initial_score(self):
        run = _run([_record(1, "diff", exp.PARSE_FAILURE), _record(2, "diff", exp.OK, 0.3)], None)
        self.assertEqual(exp.best_so_far(run), [None, None, 0.3])

    def test_new_best_iterations(self):
        self.assertEqual([r["iteration"] for r in exp.new_best_iterations(RUN)], [2, 6])

    def test_records_sorted_before_use(self):
        shuffled = dict(RUN, records=list(reversed(RUN["records"])))
        self.assertEqual(exp.best_so_far(shuffled), exp.best_so_far(RUN))

    def test_mode_stats(self):
        stats = exp.mode_stats([RUN])
        diff, rewrite = stats["diff"], stats["full_rewrite"]
        self.assertEqual(
            (diff["attempts"], diff[exp.OK], diff[exp.PARSE_FAILURE], diff["new_best"]),
            (4, 3, 1, 1),
        )
        self.assertEqual(
            (rewrite["attempts"], rewrite[exp.EVAL_FAILURE], rewrite[exp.MISSING]),
            (3, 1, 1),
        )
        self.assertEqual(rewrite["new_best"], 1)
        self.assertEqual(diff["completion_tokens"], [10, 12, 14, 16])

    def test_wilson_interval(self):
        lo, hi = exp.wilson_interval(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        self.assertEqual(exp.wilson_interval(0, 10)[0], 0.0)

    def test_summarise_curves(self):
        other = _run([_record(1, "diff", exp.OK, 0.6)], seed=2)
        summary = exp.summarise_curves([RUN, other])
        self.assertEqual(summary["final_best"], [0.9, 0.6])
        self.assertAlmostEqual(summary["final_mean"], 0.75)

    def test_report_and_csv(self):
        runs = [RUN, _run([_record(1, "diff", exp.OK, 0.6)], p=0.0)]
        report = exp.build_report(runs)
        self.assertIn("## ex", report)
        self.assertIn("| 0.5 | full_rewrite | 3 |", report)
        self.assertIn("Within mixed runs", report)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curves.csv"
            exp.write_curves_csv(runs, path)
            lines = path.read_text().splitlines()
        self.assertEqual(lines[0], "example,full_rewrite_probability,seed,iteration,best_so_far")
        self.assertEqual(len(lines), 1 + 8 + 2)


class TestFailureCapture(unittest.TestCase):
    def test_rejection_line(self):
        import logging

        collector = exp._RejectionCollector()
        record = logging.makeLogRecord({"msg": "Database rejected program %s", "args": ("abc-1",)})
        collector.emit(record)
        self.assertEqual(collector.rejected_ids, ["abc-1"])

    def test_failure_reasons_group_numbers(self):
        run = _run(
            [
                _record(1, "diff", exp.PARSE_FAILURE),
                _record(2, "diff", exp.PARSE_FAILURE),
                _record(3, "full_rewrite", exp.EVAL_FAILURE),
            ]
        )
        run["records"][0]["error"] = "None of the 3 SEARCH block(s) matched the parent program"
        run["records"][1]["error"] = "None of the 5 SEARCH block(s) matched the parent program"
        run["records"][2]["error"] = "Program execution failed: index 26 is out of bounds"
        self.assertEqual(
            exp.failure_reasons([run]),
            {
                ("diff", "None of the N SEARCH block(s) matched the parent program"): 2,
                ("full_rewrite", "Program execution failed"): 1,
            },
        )


class TestRunConfig(unittest.TestCase):
    def test_overrides_and_validates(self):
        from openevolve.config import Config

        with tempfile.TemporaryDirectory() as tmp:
            raw = exp.build_run_config_dict(
                exp.EXAMPLES["circle_packing"],  # ships with diff_based_evolution: false
                probability=0.2,
                seed=7,
                iterations=30,
                run_dir=Path(tmp),
                llm_section={"api_base": "http://localhost:1/v1", "api_key": "secret"},
            )
            config = Config.from_dict(raw)
        self.assertTrue(config.diff_based_evolution)
        self.assertEqual(config.full_rewrite_probability, 0.2)
        self.assertEqual(config.random_seed, 7)
        self.assertEqual(config.max_iterations, 30)
        self.assertIsNone(config.early_stopping_patience)
        self.assertTrue(config.evolution_trace.enabled)
        self.assertTrue(config.evolution_trace.include_failures)
        self.assertEqual(exp.redact(raw)["llm"]["api_key"], "<redacted>")
        self.assertEqual(raw["llm"]["api_key"], "secret")


if __name__ == "__main__":
    unittest.main()
