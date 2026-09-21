"""Regression checks that malformed paper inputs and incomplete results fail."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from check_geometry import ROOT, check_input, load_cases, load_input


class PaperInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()

    def test_original_hashes_geometry_and_elimination(self):
        for case in self.cases:
            with self.subTest(genus=case["genus"]):
                check_input(case, load_input(case), products=False)

    def test_changed_node_is_rejected(self):
        case = self.cases[0]
        instance = load_input(case)
        instance["cyclic"]["points"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "point recipe"):
            check_input(case, instance, products=False)

    def test_alternating_g20_is_not_the_paper_case(self):
        case = self.cases[0]
        instance = load_input(case)
        instance["deformation"]["point_velocities"][1] = [case["prime"]-case["zeta"], 0]
        with self.assertRaisesRegex(ValueError, "paired velocities"):
            check_input(case, instance, products=False)

    def test_wrong_taylor_inverse_is_rejected(self):
        case = self.cases[0]
        instance = load_input(case)
        instance["deformation"]["right_inverse_coefficients"][2][0] += 1
        with self.assertRaisesRegex(ValueError, "mu_w R"):
            check_input(case, instance, products=False)

    def test_historical_results_cover_each_required_operator(self):
        records = json.loads((ROOT/"data/recorded-results.json").read_text())["cases"]
        self.assertEqual([r["genus"] for r in records], [c["genus"] for c in self.cases])
        for case, record in zip(self.cases, records):
            with self.subTest(genus=case["genus"]):
                observed = {}
                for run in record["runs"]:
                    key = (run["operator"], run["sector"])
                    self.assertNotIn(key, observed)
                    observed[key] = (run["rows"], run["columns"], run["rank"])
                expected = {("eliminated", sector): (case["sector_rows"][sector], case["sector_columns"][sector], rank)
                            for sector, rank in enumerate(case["expected_base_ranks"])}
                for sector, rank in case["expected_augmented_ranks"].items():
                    sector = int(sector)
                    expected[("augmented", sector)] = (case["sector_rows"][sector], case["sector_columns"][sector]+2, rank)
                if case["expected_replacement_rank"] is not None:
                    expected[("replacement", 0)] = (case["sector_rows"][0], case["sector_columns"][0], case["expected_replacement_rank"])
                    self.assertEqual(record["deformation_residuals"]["base_kernel_residual_nonzeros"], 0)
                    self.assertTrue(record["deformation_residuals"]["first_order_equations_verified"])
                self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
