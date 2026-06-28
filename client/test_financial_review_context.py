import unittest
from unittest.mock import patch
from datetime import datetime

from finance.query_tools import month_coverage_metadata


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 6, 25)


class FinancialReviewContextTests(unittest.TestCase):
    def test_month_coverage_marks_current_month_partial(self):
        with patch("finance.query_tools.datetime", FixedDateTime):
            coverage = month_coverage_metadata("2026-06")

        self.assertEqual(coverage["month"], "2026-06")
        self.assertFalse(coverage["is_complete_month"])
        self.assertEqual(coverage["coverage_days"], 25)
        self.assertEqual(coverage["expected_days"], 30)

    def test_month_coverage_marks_historical_month_complete(self):
        with patch("finance.query_tools.datetime", FixedDateTime):
            coverage = month_coverage_metadata("2026-05")

        self.assertTrue(coverage["is_complete_month"])
        self.assertEqual(coverage["coverage_days"], 31)
        self.assertEqual(coverage["expected_days"], 31)


if __name__ == "__main__":
    unittest.main()
