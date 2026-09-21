"""Failure-path and fixed-input checks for the end-to-end orchestrator."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.reproduce import Pipeline, ROOT, read_rank, result_path, sha256, validate_kernel_report


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / "data/cases.json").read_text())

    def pipeline(self, genus=20, gpus=None, threads=2):
        case = next(c for c in self.config["cases"] if c["genus"] == genus)
        return Pipeline(case, self.config["settings"], self.root / f"g{genus}",
                        [0] if gpus is None else gpus, threads)

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

    def test_gpu_work_overlaps_bounded_cpu_rank_work(self):
        run = self.pipeline(threads=4)
        jobs = [run.job("eliminated", c, run.case["expected_base_ranks"][c]) for c in range(3)]
        by_name = {job["run_id"]: job for job in jobs}
        lock = threading.Lock()
        first_rank = threading.Event()
        second_rank = threading.Event()
        counts = dict(sequences=0, ranks=0, cpu=0, peak_cpu=0, gpu=0, overlap=False)

        def command(name, args, threads=None):
            job = by_name[name.rsplit("-", 1)[0]]
            wdm = Path(job["wdm_path"])
            if name.endswith("-sequence"):
                with lock:
                    counts["gpu"] += 1
                    self.assertEqual(counts["gpu"], 1)
                    counts["sequences"] += 1
                    first = counts["sequences"] == 1
                try:
                    if not first:
                        self.assertTrue(first_rank.wait(5), "GPU stalled behind CPU rank work")
                        with lock:
                            counts["overlap"] |= counts["cpu"] > 0
                    wdm.touch()
                    result_path(wdm, ".rowmix.json").write_text("{}")
                finally:
                    with lock:
                        counts["gpu"] -= 1
            else:
                with lock:
                    counts["ranks"] += 1
                    index = counts["ranks"]
                    counts["cpu"] += threads
                    counts["peak_cpu"] = max(counts["peak_cpu"], counts["cpu"])
                try:
                    if index == 1:
                        first_rank.set()
                        self.assertTrue(second_rank.wait(5), "Independent CPU rank jobs did not overlap")
                    elif index == 2:
                        second_rank.set()
                    result_path(wdm, "_result.txt").write_text(
                        f"Matrix size: {job['rows']} x {job['columns']}\nRank: {job['expected_rank']}\n")
                finally:
                    with lock:
                        counts["cpu"] -= threads

        with patch.object(run, "command", command):
            run.execute_jobs(jobs)
        self.assertTrue(counts["overlap"])
        self.assertEqual(counts["peak_cpu"], run.threads)
        self.assertEqual(len(run.summary["ranks"]), 3)

    def test_parallel_failure_terminates_children_and_releases_gpus(self):
        run = self.pipeline(gpus=[0, 1])
        jobs = [run.job("eliminated", c, run.case["expected_base_ranks"][c]) for c in range(2)]
        marker = self.root / "other-child-started"
        first = ("import pathlib,sys,time\n"
                 "deadline=time.monotonic()+5\n"
                 "while not pathlib.Path(sys.argv[1]).exists() and time.monotonic()<deadline:\n"
                 "    time.sleep(0.01)\n"
                 "raise SystemExit(7)\n")
        second = "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(30)"
        command = run.command
        popen = subprocess.Popen
        children = []

        def spawn(*args, **kwargs):
            process = popen(*args, **kwargs)
            children.append(process)
            return process

        def simulate(name, args, threads=None):
            script = first if name == jobs[0]["run_id"] + "-sequence" else second
            command(name, [sys.executable, "-c", script, marker], threads)

        with patch.object(run, "command", simulate), patch.object(subprocess, "Popen", spawn):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "exited with"):
                    run.execute_jobs(jobs)
        self.assertTrue(marker.exists())
        self.assertEqual(len(children), 2)
        self.assertTrue(all(p.poll() is not None for p in children))
        self.assertTrue(run.cancelled.is_set())
        self.assertFalse(run.processes)

    def test_deformation_starts_before_background_base_jobs_finish(self):
        run = self.pipeline()
        background_started = threading.Event()
        deformation_finished = threading.Event()
        jobs = [dict(sector=c) for c in range(2)]

        def job(item):
            if item["sector"] == 1:
                background_started.set()
                self.assertTrue(deformation_finished.wait(5), "Deformation waited for all base sectors")

        def deform(base):
            self.assertEqual(base, jobs)
            self.assertTrue(background_started.wait(5))
            deformation_finished.set()

        with patch.object(run, "execute_job", job), patch.object(run, "deform", deform):
            run.execute_jobs(jobs, deform=True)
        self.assertTrue(deformation_finished.is_set())


if __name__ == "__main__":
    unittest.main()
