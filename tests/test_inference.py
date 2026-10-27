from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from water_quality.predict import predict_file


@pytest.mark.skipif(not Path("artifacts/metadata.json").exists(), reason="Train artifacts first")
@pytest.mark.parametrize("family", ["XGBoost", "MLP", "Blend"])
def test_saved_models_reproduce_recorded_test_predictions(tmp_path, family):
    fixture = pd.read_csv("examples/measurements.csv")
    recorded = pd.read_csv("examples/expected_predictions.csv")
    # Input and expected-output rows retain the same order from the holdout sample.
    expected = recorded[family].to_numpy()
    result = predict_file("examples/measurements.csv", "artifacts", tmp_path / "out.csv", family)
    assert len(result) == len(fixture)
    np.testing.assert_allclose(result.predicted_source_index, expected, atol=1e-4, rtol=1e-6)
    assert (result.lower_90 <= result.predicted_source_index).all()
    assert (result.upper_90 >= result.predicted_source_index).all()


@pytest.mark.skipif(not Path("artifacts/metadata.json").exists(), reason="Train artifacts first")
def test_inference_refuses_a_fully_invalid_sensor_row(tmp_path):
    fixture = pd.read_csv("examples/measurements.csv").iloc[:1].copy()
    fixture[:] = np.nan
    fixture.to_csv(tmp_path / "empty.csv", index=False)
    with pytest.raises(ValueError, match="at least one valid measurement"):
        predict_file(tmp_path / "empty.csv", "artifacts", tmp_path / "out.csv")
