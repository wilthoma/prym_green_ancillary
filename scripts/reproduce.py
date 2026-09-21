"""The complete, fixed paper pipeline. WDM files are internal working data.

GPU sequence/kernel jobs and CPU rank jobs overlap with separate resource
limits. A verified sector-zero kernel unlocks the deformation stages.
No historical output is reused. A failed stage cannot produce a successful
summary, and child processes are terminated on failure or interruption.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import heapq
from itertools import count
import json
import os
import platform
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
GENERA = (20, 22, 24, 26, 28, 30)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Replace a small summary atomically so failed writes are not successes."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def result_path(wdm, suffix):
    """BCW strips .wdm.zst; CUDA kernel outputs instead append to that name."""
    return Path(str(wdm).removesuffix(".wdm.zst") + suffix)


def read_rank(path, rows, columns, expected):
    text = Path(path).read_text()
    match = re.fullmatch(r"\s*Matrix size:\s*(\d+)\s*x\s*(\d+)\s*Rank:\s*(\d+)\s*", text)
    if match is None:
        raise ValueError(f"Invalid rank result: {path}")
    actual_rows, actual_columns, rank = map(int, match.groups())
    if (actual_rows, actual_columns) != (rows, columns):
        raise ValueError(f"Wrong matrix dimensions in {path}")
    if rank != expected:
        raise ValueError(f"Rank lower bound {rank}; expected {expected}: {path}")
    return rank


def validate_kernel_report(path, job, prime):
    report = read_json(path)
    for key, value in dict(operator=job["operator"], sector=job["sector"],
                           rows=job["rows"], columns=job["columns"], prime=prime).items():
        if report.get(key) != value:
            raise ValueError(f"Kernel report {key} mismatch: {path}")
    products = report.get("product_nonzeros")
    if report.get("candidate_rank") != 2 or not products or any(x != 0 for x in products):
        raise ValueError(f"Kernel independence or residual check failed: {path}")
    return report


class Resources:
    """Allocate devices or CPU slots, giving deformation work first choice.

    Jobs already running finish normally. Among waiting jobs, priority zero
    runs before priority one, with arrival order breaking ties.
    """

    def __init__(self, values):
        self.available = list(values)
        self.waiting = []
        self.tickets = count()
        self.condition = threading.Condition()

    @contextmanager
    def reserve(self, priority=0):
        with self.condition:
            ticket = (priority, next(self.tickets))
            heapq.heappush(self.waiting, ticket)
            try:
                self.condition.wait_for(lambda: self.available and self.waiting[0] == ticket)
            except BaseException:
                self.waiting.remove(ticket)
                heapq.heapify(self.waiting)
                self.condition.notify_all()
                raise
            heapq.heappop(self.waiting)
            value = self.available.pop(0)
            self.condition.notify_all()
        try:
            yield value
        finally:
            with self.condition:
                self.available.append(value)
                self.condition.notify_all()


class Pipeline:
    """One fresh genus run, using the fixed numerical recipe in cases.json."""

    def __init__(self, case, settings, directory, gpus, threads):
        self.case, self.settings = case, settings
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "logs").mkdir()
        self.artifacts = self.directory / "artifacts"
        self.artifacts.mkdir()
        self.gpus, self.threads = gpus, threads
        # A small fixed number of rank workers overlaps CPU and GPU work.
        self.rank_slots = min(2 * len(gpus), threads)
        self.worker_threads = max(1, threads // self.rank_slots)
        self.devices = Resources(gpus)
        self.cpu_slots = Resources(range(self.rank_slots))
        self.instance = self.directory / "input.json"
        self.geometry = ROOT / "target/release/prym-phi"
        self.rank_binary = ROOT / "target/release/prym-rank"
        self.lock = threading.Lock()
        self.processes = set()
        self.cancelled = threading.Event()
        self.summary = dict(genus=case["genus"], prime=case["prime"], status="running",
                            started_at=datetime.now(timezone.utc).isoformat(),
                            settings=settings, gpus=gpus, cpu_threads=threads, ranks=[], commands=[],
                            python=platform.python_version(), platform=platform.platform())

    def save(self):
        write_json(self.directory / "summary.json", self.summary)

    def command(self, name, args, threads=None):
        """Send verbose algebra/GPU output to a per-stage log, never a pipe."""
        if self.cancelled.is_set():
            raise RuntimeError("Pipeline cancelled")
        args = list(map(str, args))
        log_path = self.directory / "logs" / f"{name}.log"
        env = dict(os.environ, RAYON_NUM_THREADS=str(threads or self.threads))
        started = time.monotonic()
        print(f"g={self.case['genus']}: {name}", flush=True)
        with log_path.open("w") as log:
            with self.lock:
                if self.cancelled.is_set():
                    raise RuntimeError("Pipeline cancelled")
                process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                self.processes.add(process)
            try:
                returncode = process.wait()
            except BaseException:
                # A main-thread interrupt can arrive during a CPU stage too.
                # Keep the child registered until it has been terminated/reaped.
                self.abort()
                raise
            finally:
                with self.lock:
                    self.processes.discard(process)
        record = dict(stage=name, command=args, exit_code=returncode,
                      seconds=round(time.monotonic() - started, 3),
                      log=str(log_path.relative_to(self.directory)))
        with self.lock:
            self.summary["commands"].append(record)
            self.save()
        if returncode:
            raise RuntimeError(f"{name} exited with {returncode}; see {log_path}")

    def abort(self):
        self.cancelled.set()
        with self.lock:
            processes = list(self.processes)
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

    def prepare(self):
        from scripts.check_geometry import validate_input
        source = ROOT / "data" / self.case["input"]
        if sha256(source) != self.case["sha256"]:
            raise ValueError(f"Fixed paper input checksum mismatch: {source}")
        fixture = read_json(source)
        self.summary["frozen_input_sha256"] = sha256(source)
        self.summary["geometry"] = validate_input(self.case, ROOT / "data")
        if self.case["deformation"]:
            from scripts.prepare_deformation import build_instance
            rebuilt = build_instance(self.case["genus"], self.case["prime"])
            comparison = json.loads(json.dumps(rebuilt))  # tuples become JSON arrays
            original = json.loads(json.dumps(fixture))
            original["deformation"].pop("generator_sha256", None)
            if comparison != original:
                raise ValueError("Rebuilt Taylor tensors differ from the fixed paper input")
            write_json(self.instance, rebuilt)
        else:
            self.instance.write_bytes(source.read_bytes())
            self.command("verify-input", [self.geometry, "verify-cyclic", "--instance", self.instance])
        self.summary["input_sha256"] = sha256(self.instance)
        self.summary["small_input_checks_passed"] = True
        self.save()

    def job(self, operator, sector, expected_rank, dense=None, drop=None, kernel=False):
        columns = self.case["sector_columns"][sector] + (2 if operator == "augmented" else 0)
        label = f"{operator}-{sector:02}"
        out = self.directory / label
        out.mkdir()
        s = self.settings
        filename = (f"{operator}-sector{sector:02d}-v{s['vector_count']}"
                    f"-rowmix-r{s['rowmix_rounds']}-s{s['rowmix_seed']}.wdm.zst")
        return dict(run_id=label, operator=operator, sector=sector,
                    rows=self.case["sector_rows"][sector], columns=columns,
                    expected_rank=expected_rank, wdm_path=str(out / filename),
                    dense=dense, drop=drop, kernel=kernel)

    def gpu_command(self, name, args, priority):
        """Hold a GPU only while a CUDA executable is actually running."""
        with self.devices.reserve(priority) as device:
            args = [*args, "--device", device]
            self.command(name, args, self.worker_threads)
            return list(map(str, args))

    def cpu_command(self, name, args, priority=0):
        with self.cpu_slots.reserve(priority):
            self.command(name, args, self.worker_threads)

    def execute_job(self, job):
        s = self.settings
        operator, sector = job["operator"], job["sector"]
        priority = 0 if job["kernel"] or operator == "replacement" else 1
        program = "cuprym_cyclic" if operator == "eliminated" else "cuprym_deformation"
        args = [ROOT / "cuda" / program, self.instance, "--operator", operator,
                "--sector", sector, "-v", s["vector_count"],
                "--seed", s["numerical_seed"], "--rowmix-rounds", s["rowmix_rounds"],
                "--rowmix-seed", s["rowmix_seed"], "--out", Path(job["wdm_path"]).parent]
        if job["dense"]:
            args += ["--dense-file", job["dense"]]
        if job["drop"]:
            args += ["--drop-columns", job["drop"]]
        job["args"] = self.gpu_command(job["run_id"] + "-sequence", args, priority)
        wdm = Path(job["wdm_path"])
        if not wdm.is_file() or not result_path(wdm, ".rowmix.json").is_file():
            raise RuntimeError(f"Missing sequence or row-mix metadata for {wdm}")
        args = [self.rank_binary, wdm, "--threads", self.worker_threads]
        if job["kernel"]:
            args += ["--generator"]
        self.cpu_command(job["run_id"] + "-rank", args, priority)
        job["rank"] = read_rank(result_path(wdm, "_result.txt"), job["rows"],
                                job["columns"], job["expected_rank"])
        if job["kernel"]:
            args = [ROOT / "cuda" / (program + "_kernel_vectors"), self.instance,
                    "-f", wdm, "-g", result_path(wdm, "_generators.txt"),
                    "--operator", operator, "--sector", sector]
            if job["dense"]:
                args += ["--dense-file", job["dense"]]
            self.gpu_command(job["run_id"] + "-kernel", args, priority)
            validate_kernel_report(str(wdm) + "_nullvectors_report.json", job, self.case["prime"])
        with self.lock:
            self.summary["ranks"].append({k: job[k] for k in ("operator", "sector", "rows", "columns", "rank")})
            self.save()

    def guarded_job(self, job):
        try:
            self.execute_job(job)
        except BaseException:
            # A background sector failure must also stop the deformation lane.
            self.abort()
            raise

    def execute_jobs(self, jobs, deform=False):
        if not jobs:
            return
        # Each fixed paper stage has at most fifteen lightweight coordinators;
        # shared resource pools bound the actual CPU and GPU subprocesses.
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(self.guarded_job, job) for job in jobs]
            try:
                if deform:
                    futures[0].result()  # sector-zero kernel is now verified
                    self.deform(jobs)   # other base sectors continue meanwhile
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                self.abort()
                raise

    def deform(self, base):
        zero = base[0]
        wdm = zero["wdm_path"]
        self.cpu_command("extract-kernel", [self.geometry, "deformation-extract-base-kernel",
                     "--instance", self.instance, "--kernel-vectors", wdm + "_nullvectors_2.txt",
                     "--kernel-report", wdm + "_nullvectors_report.json", "--out-dir", self.artifacts])
        report = read_json(self.artifacts / "base_kernel_report.json")
        if report.get("rank") != 2 or report.get("F0_residual_nonzeros") != 0:
            raise ValueError("Base kernel verification failed")
        kernel = self.artifacts / "K_sector0.json"
        corrections = self.case["correction_sectors"]
        self.cpu_command("derive-rhs", [self.geometry, "deformation-derive-rhs", "--instance", self.instance,
                     "--kernel", kernel, "--out-dir", self.artifacts, "--expected-corrections",
                     ",".join(map(str, corrections))])
        rhs = read_json(self.artifacts / "rhs_manifest.json")["rhs"]
        if sorted(r["sector"] for r in rhs) != sorted(corrections):
            raise ValueError("Wrong first-order correction sectors")
        jobs = [self.job("augmented", r["sector"],
                         self.case["expected_augmented_ranks"][str(r["sector"])],
                         dense=r["path"], kernel=True) for r in rhs]
        self.execute_jobs(jobs)
        manifest = self.artifacts / "augmented.json"
        write_json(manifest, {"runs": jobs})
        self.cpu_command("normalize-corrections", [self.geometry, "deformation-normalize-augmented",
                     "--instance", self.instance, "--manifest", manifest, "--out-dir", self.artifacts])
        solutions = self.artifacts / "solutions_manifest.json"
        checked = read_json(solutions)
        if (checked.get("first_order_equations_verified") is not True
                or sorted(r["sector"] for r in checked["solutions"]) != sorted(corrections)
                or any(r.get("residual_nonzeros") != 0 for r in checked["solutions"])):
            raise ValueError("First-order correction verification failed")
        self.cpu_command("derive-obstruction", [self.geometry, "deformation-derive-quadratic",
                     "--instance", self.instance, "--kernel", kernel,
                     "--solutions-manifest", solutions, "--out-dir", self.artifacts])
        quadratic = read_json(self.artifacts / "quadratic_manifest.json")
        indices = quadratic["drop_columns"]["indices"]
        if indices != self.case["replacement_drop_columns"]:
            raise ValueError(f"Unexpected kernel pivot rows: {indices}")
        replacement = self.job("replacement", 0, self.case["expected_replacement_rank"],
                               dense=quadratic["Z0"]["path"], drop=quadratic["drop_columns"]["path"])
        self.execute_jobs([replacement])
        with self.lock:
            self.summary["deformation"] = dict(kernel_rank=2, kernel_residual_nonzeros=0,
                                               correction_sectors=corrections,
                                               correction_residual_nonzeros=0,
                                               replacement_drop_columns=indices,
                                               replacement_rank=replacement["rank"])

    def run(self):
        started = time.monotonic()
        self.save()
        try:
            binaries = [self.geometry, self.rank_binary] + [ROOT / "cuda" / name for name in (
                "cuprym_cyclic", "cuprym_cyclic_kernel_vectors", "cuprym_deformation",
                "cuprym_deformation_kernel_vectors")]
            self.summary["executable_sha256"] = {
                str(path.relative_to(ROOT)): sha256(path) for path in binaries}
            self.prepare()
            base = [self.job("eliminated", c, rank, kernel=bool(self.case["deformation"] and c == 0))
                    for c, rank in enumerate(self.case["expected_base_ranks"])]
            self.execute_jobs(base, deform=bool(self.case["deformation"]))
            self.summary["status"] = "passed"
        except BaseException as exc:
            self.abort()
            self.summary.update(status="failed", error=str(exc) or type(exc).__name__)
            raise
        finally:
            self.summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
            self.summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.save()
        print(f"PASS g={self.case['genus']}: all rank and residual checks passed.\n"
              f"Summary: {self.directory / 'summary.json'}", flush=True)


def parse_gpus(text):
    try:
        devices = [int(x) for x in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use GPU indices such as 0 or 0,1") from exc
    if not devices or min(devices) < 0 or len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError("GPU indices must be distinct nonnegative integers")
    return devices


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce complete Prym–Green paper computations (CUDA required).")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--genus", type=int, choices=GENERA)
    selection.add_argument("--all", action="store_true", help="run all six genera, including the multi-day genus-30 case")
    parser.add_argument("--gpus", type=parse_gpus, default=[0], help="GPU indices, default: 0; use 0,1 for two GPUs")
    parser.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1), help="total CPU thread budget")
    parser.add_argument("--output", type=Path, default=ROOT / "runs", help="working-directory parent (default: runs/)")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    from scripts.check_geometry import load_cases
    config = read_json(ROOT / "data/cases.json")
    cases = [c for c in load_cases() if args.all or c["genus"] == args.genus]
    required = [ROOT / "target/release" / n for n in ("prym-phi", "prym-rank")]
    required += [ROOT / "cuda" / n for n in ("cuprym_cyclic", "cuprym_cyclic_kernel_vectors",
                                             "cuprym_deformation", "cuprym_deformation_kernel_vectors")]
    missing = [str(p.relative_to(ROOT)) for p in required if not os.access(p, os.X_OK)]
    if missing:
        parser.error("Run make on a CUDA machine first. Missing executables: " + ", ".join(missing))
    try:
        for case in cases:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            Pipeline(case, config["settings"], args.output / f"g{case['genus']}-{stamp}",
                     args.gpus, args.threads).run()
    except KeyboardInterrupt:
        print("Interrupted; working files and logs retained.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
