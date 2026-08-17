from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from evaluate.visualization import radar_plot


class TestViz(TestCase):
    def test_invert_range(self):
        data = [{"accuracy": 0.9, "precision": 0.8}, {"accuracy": 0.7, "precision": 0.6}]
        model_names = ["model1", "model2"]
        wrong_invert_range = ["latency_in_seconds"]  # Value not present in data
        with self.assertRaises(ValueError):
            radar_plot(data, model_names, wrong_invert_range)

    def test_output_is_plot(self):
        data = [
            {"accuracy": 0.9, "precision": 0.8, "latency_in_seconds": 48.1},
            {"accuracy": 0.7, "precision": 0.6, "latency_in_seconds": 51.4},
        ]
        model_names = ["model1", "model2"]
        invert_range = ["latency_in_seconds"]
        out_plt = radar_plot(data, model_names, invert_range)
        self.assertIsInstance(out_plt, plt.Figure)

    def test_range_padding_is_symmetric(self):
        data = [{"accuracy": 0.0, "precision": 1.0}, {"accuracy": 10.0, "precision": 2.0}]
        model_names = ["model1", "model2"]
        out_plt = radar_plot(data, model_names)
        # the accuracy values span 10, so both ends of its axis are padded by 1
        low, high = out_plt.axes[0].get_ylim()
        self.assertAlmostEqual(low, -1.0)
        self.assertAlmostEqual(high, 11.0)

    def test_metric_with_identical_values(self):
        data = [{"accuracy": 0.9, "precision": 0.8}, {"accuracy": 0.9, "precision": 0.6}]
        model_names = ["model1", "model2"]
        out_plt = radar_plot(data, model_names)
        # accuracy is identical for both models, but precision is not, so the two
        # plotted shapes must still differ from each other
        first, second = (line.get_ydata() for line in out_plt.axes[1].lines)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.isfinite(second).all())
        self.assertFalse(np.array_equal(first, second))
