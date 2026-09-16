import unittest
from nrsm_run_budget import RunBudget


class RunBudgetTests(unittest.TestCase):
    def test_timed_default_preserved(self):
        budget = RunBudget.from_args()
        self.assertEqual(budget.seconds, 3600)
        self.assertFalse(budget.reached(3599, 300001))
        self.assertTrue(budget.reached(3600, 0))
        self.assertEqual(budget.stop_reason, 'duration')

    def test_full_budget_counts_restored_steps_and_has_no_hour_limit(self):
        budget = RunBudget.from_args(steps=300000)
        self.assertEqual(budget.seconds, 0)
        self.assertFalse(budget.reached(360000, 8400))
        self.assertFalse(budget.reached(360000, 299999))
        self.assertTrue(budget.reached(1, 300000))
        self.assertEqual(budget.stop_reason, 'environment_step_target')

    def test_invalid_or_ambiguous_budgets_rejected(self):
        for seconds, steps in ((0, 0), (-1, 0), (0, -1), (3600, 300000), (3601, 0)):
            with self.subTest(seconds=seconds, steps=steps), self.assertRaises(ValueError):
                RunBudget(seconds, steps)


if __name__ == '__main__':
    unittest.main()
