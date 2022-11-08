from unittest import TestCase

import matplotlib.pyplot as plt

from evaluate.visualization import radar_plot


class TestViz(TestCase):
    def test_invert_range_bad_input(self):
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

    def test_input(self):
        data = [{"accuracy": 0.9, "precision": 0.8}, {"accuracy": 0.7, "precision": 0.6}]
        self.assertIsInstance(type(data), list)
