import unittest

from measurement.synthetic_sessions import synthetic_sessions


class SyntheticSessionTests(unittest.TestCase):
    def test_contexts_grow_and_agents_share_no_prefix(self):
        sessions = synthetic_sessions(n_agents=3, n_steps=4, words_per_step=20, pre_gap_s=0.5)
        self.assertEqual(len(sessions), 3)
        first_messages = {rows[0]["input"][0]["content"] for rows in sessions.values()}
        self.assertEqual(len(first_messages), 3)
        for rows in sessions.values():
            self.assertEqual([r["step"] for r in rows], [1, 2, 3, 4])
            self.assertEqual([r["pre_gap"] for r in rows], [0.0, 0.5, 0.5, 0.5])
            for earlier, later in zip(rows, rows[1:]):
                # Each step resends the previous step's context as its prefix.
                self.assertEqual(later["input"][:len(earlier["input"])], earlier["input"])
                self.assertGreater(len(later["input"]), len(earlier["input"]))

    def test_deterministic(self):
        self.assertEqual(synthetic_sessions(seed=1, n_steps=2), synthetic_sessions(seed=1, n_steps=2))
        self.assertNotEqual(synthetic_sessions(seed=1, n_steps=2), synthetic_sessions(seed=2, n_steps=2))


if __name__ == "__main__":
    unittest.main()
