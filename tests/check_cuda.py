#!/usr/bin/env python3
"""Small CUDA integration test against independently assembled exact matrices.

Run after `make`, with the Python requirements installed. This developer test
uses genus 12 so every reference matrix is small. The paper runner still offers
only the six paper genera and always computes complete sequences from scratch.
"""
from datetime import datetime, timezone
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.reproduce import Pipeline, ROOT, read_json, sha256
from deformation_reference import check, inverse


class ReferencePipeline(Pipeline):
    def prepare(self):
        source = ROOT / "tests/fixtures/g12.json"
        self.instance.write_bytes(source.read_bytes())
        self.summary["input_sha256"] = sha256(self.instance)
        self.reference = check(read_json(self.instance), include_vectors=True)
        self.summary["independent_reference_passed"] = True

    def command(self, name, args, threads=None):
        # Compare CPU and CUDA Gram applications on three deterministic blocks
        # in every base sector before computing the complete sequence.
        if Path(args[0]).name == "cuprym_cyclic":
            args = [*args, "--validate-rowmix"]
        super().command(name, args, threads)

    def deform(self, base):
        super().deform(base)
        certificate = self.reference["certificate"]
        prime = self.case["prime"]

        def dense(path):
            block = read_json(path)
            return np.asarray(block["values"], dtype=np.int64).reshape(
                block["rows"], block["columns"])

        # CUDA may choose a different kernel basis. Compare all subsequent
        # vectors after the same invertible two-dimensional basis change.
        kernel = dense(self.artifacts / "K_sector0.json")
        expected = np.asarray(certificate["K0"], dtype=np.int64)
        pivots = certificate["J"]
        change = inverse(expected[pivots], prime) @ kernel[pivots] % prime
        inverse(change, prime)  # also proves that the recovered span has rank 2
        np.testing.assert_array_equal(kernel, expected @ change % prime)
        solutions = read_json(self.artifacts / "solutions_manifest.json")
        for solution in solutions["solutions"]:
            expected = np.asarray(certificate["S_by_sector"][str(solution["sector"])],
                                  dtype=np.int64)
            np.testing.assert_array_equal(dense(solution["path"]), expected @ change % prime)
        quadratic = read_json(self.artifacts / "quadratic_manifest.json")
        expected = np.asarray(certificate["Z0"], dtype=np.int64)
        np.testing.assert_array_equal(dense(quadratic["Z0"]["path"]), expected @ change % prime)
        self.summary["recovered_vectors_match_independent_reference"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/cuda-tests")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")
    # Expected ranks come from the independent exact elimination certificate.
    reference = read_json(ROOT / "tests/fixtures/g12-reference.json")
    profiles = reference["sector_profiles"]
    corrections = reference["first_order_corrections"]
    case = dict(genus=12, prime=109, deformation="paired",
                sector_rows=[p["rows"] for p in profiles],
                sector_columns=[p["columns"] for p in profiles],
                expected_base_ranks=[p["rank"] for p in profiles],
                correction_sectors=[c["sector"] for c in corrections],
                expected_augmented_ranks={str(c["sector"]): c["augmented_shape"][1] - 2
                                          for c in corrections},
                expected_replacement_rank=reference["completion_rank"],
                replacement_drop_columns=reference["kernel_pivot_rows_sector0"])
    settings = read_json(ROOT / "data/cases.json")["settings"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    ReferencePipeline(case, settings, args.output / f"g12-{stamp}", [args.device], 2).run()


if __name__ == "__main__":
    main()
