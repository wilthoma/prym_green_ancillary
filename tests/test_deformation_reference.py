"""An explicit small matrix independently checks the factored deformation."""
import unittest
from scripts.prepare_deformation import build_instance
from deformation_reference import check


class DeformationReferenceTests(unittest.TestCase):
    def test_paired_genus12_obstruction(self):
        # This tiny case uses explicit matrices and Gaussian elimination,
        # independently of Rust factored applications and GPU/BCW ranks.
        result = check(build_instance(12, 109), include_vectors=False)
        self.assertEqual(result["original_kernel_dimension"], 2)
        self.assertEqual(result["completion_rank"], 84)
        self.assertTrue(result["first_order_residual_zero"])
        self.assertTrue(result["factored_actions_and_transposes_verified"])
        self.assertTrue(result["generic_injectivity_verified"])


if __name__ == "__main__":
    unittest.main()
