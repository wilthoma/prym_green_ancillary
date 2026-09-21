"""Failure-path and fixed-input checks for the end-to-end orchestrator."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.reproduce import Pipeline, ROOT, read_rank, sha256, validate_kernel_report


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / "data/cases.json").read_text())

    def pipeline(self, genus=20):
        case = next(c for c in self.config["cases"] if c["genus"] == genus)
        return Pipeline(case, self.config["settings"], self.root / f"g{genus}", [0], 2)

    def test_paired_inputs_are_reconstructed_and_actual_hash_recorded(self):
        for genus in (20, 24, 28):
            run = self.pipeline(genus)
            run.prepare()
            self.assertTrue(run.summary["small_input_checks_passed"])
            self.assertEqual(run.summary["input_sha256"], sha256(run.instance))
            self.assertEqual(run.summary["frozen_input_sha256"], run.case["sha256"])
            self.assertEqual(json.loads(run.instance.read_text())["deformation"]["mode"], "paired")

    def test_rank_must_match_dimensions_and_required_bound(self):
        path = self.root / "rank.txt"
        path.write_text("Matrix size: 100 x 90\nRank: 89\n")
        with self.assertRaisesRegex(ValueError, "expected 90"):
            read_rank(path, 100, 90, 90)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            read_rank(path, 101, 90, 89)
        self.assertEqual(read_rank(path, 100, 90, 89), 89)

    def test_nonzero_kernel_residual_is_rejected(self):
        job = dict(operator="eliminated", sector=0, rows=100, columns=90)
        report = dict(job, prime=661, candidate_rank=2, product_nonzeros=[0, 1, 0, 0])
        path = self.root / "kernel.json"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "residual"):
            validate_kernel_report(path, job, 661)

    def test_sequential_interrupt_terminates_and_reaps_child(self):
        run = self.pipeline()
        original_wait = subprocess.Popen.wait
        children = []

        def interrupted_wait(process, timeout=None):
            if not children:
                children.append(process)
                raise KeyboardInterrupt()
            return original_wait(process, timeout=timeout)

        with patch.object(subprocess.Popen, "wait", interrupted_wait):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    run.command("interrupted", [sys.executable, "-c", "import time; time.sleep(30)"])
        self.assertIsNotNone(children[0].poll())
        self.assertFalse(run.processes)

    def test_failed_command_is_logged_and_never_accepted(self):
        run = self.pipeline()
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "exited with 7"):
                run.command("failure", [sys.executable, "-c", "raise SystemExit(7)"])
        record = json.loads((run.directory / "summary.json").read_text())["commands"][-1]
        self.assertEqual(record["exit_code"], 7)
        self.assertTrue((run.directory / record["log"]).is_file())
        self.assertFalse(run.processes)


if __name__ == "__main__":
    unittest.main()
