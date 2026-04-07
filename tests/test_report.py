from __future__ import annotations

import matplotlib
import numpy as np

from errorquake.analyze import PredictionResult
from errorquake.report import plot_magnitude_frequency, plot_prediction_calibration, set_errorquake_style

matplotlib.use("Agg")


def test_set_errorquake_style_modifies_rcparams() -> None:
    set_errorquake_style()
    assert matplotlib.pyplot.rcParams["font.size"] == 10


def test_plot_magnitude_frequency_returns_figure() -> None:
    fig = plot_magnitude_frequency(np.array([0.5, 1.0, 1.5, 2.0]), "model-x")
    assert fig.__class__.__name__ == "Figure"


def test_plot_prediction_calibration_single_model() -> None:
    fig = plot_prediction_calibration(
        [
            PredictionResult(
                model_name="m",
                b_easy=1.0,
                predicted_catastrophic=1.0,
                observed_catastrophic=1.2,
                ratio=0.83,
                within_1_5x=True,
            )
        ]
    )
    assert fig.__class__.__name__ == "Figure"
