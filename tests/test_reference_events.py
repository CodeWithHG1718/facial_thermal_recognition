"""Protect the event interpretation that previously doubled Joao's rate."""

import unittest
from datetime import datetime, timedelta

from thermal_face.data import breath_events, recording_rate


class ReferenceEventTests(unittest.TestCase):
    def test_alternating_phase_markers_count_one_breath_each(self):
        start = datetime(2020, 1, 1)
        times = [start + timedelta(seconds=second) for second in range(10)]
        markers = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        self.assertEqual(len(breath_events(times, markers)), 3)
        self.assertAlmostEqual(recording_rate(times, markers), 20.0)


if __name__ == "__main__":
    unittest.main()
