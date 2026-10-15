
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_picker_v3.model import diff, sig, cl


class TestV41Features(unittest.TestCase):

    def test_equal_statistics(self):
        self.assertEqual(diff(0.10, 0.10, 0.05), 0.0)

    def test_missing_statistics(self):
        self.assertEqual(diff(float("nan"), float("nan"), 0.05), 0.0)

    def test_probability(self):
        self.assertAlmostEqual(sig(0), 0.5)

    def test_confidence(self):
        self.assertEqual(cl(0.75), 5)


if __name__ == "__main__":
    unittest.main()
