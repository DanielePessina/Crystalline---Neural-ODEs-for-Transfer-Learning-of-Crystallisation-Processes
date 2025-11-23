import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("diffrax")

spec = importlib.util.spec_from_file_location("error", Path(__file__).resolve().parents[1] / "metrics" / "error.py")
error = importlib.util.module_from_spec(spec)
spec.loader.exec_module(error)
make_failure_metrics = error.make_failure_metrics


def test_make_failure_metrics_structure():
    metrics = make_failure_metrics([1.0, 2.0], "Train", 50, 0.1)
    assert len(metrics) == 2
    expected_keys = {
        "Training_Experiments",
        "Training_Timepoints",
        "Experiment_Tag",
        "Initial_Concentration",
        "Measurement_Noise",
        "MAE_Total",
        "MSE_Total",
        "RMSE_Total",
        "MAPE_Total",
        "MAE_Concentration",
        "MSE_Concentration",
        "RMSE_Concentration",
        "MAPE_Concentration",
        "MAE_D43",
        "MSE_D43",
        "RMSE_D43",
        "MAPE_D43",
        "Final_Train_Loss",
        "Final_Test_Loss",
    }
    for m in metrics:
        assert set(m.keys()) == expected_keys
        assert m["Training_Experiments"] == 2
        assert m["Training_Timepoints"] == 50
        assert m["Experiment_Tag"] == "Train"
        assert m["Measurement_Noise"] == 0.1
        for key in expected_keys - {
            "Training_Experiments",
            "Training_Timepoints",
            "Experiment_Tag",
            "Initial_Concentration",
            "Measurement_Noise",
        }:
            assert m[key] == 999.0
