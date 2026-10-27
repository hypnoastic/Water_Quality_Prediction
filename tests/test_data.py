import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from water_quality.data import (
    FEATURES,
    TARGET,
    prepare,
    quality_features,
    source_formula,
    temporal_split,
)


def measurements():
    return pd.DataFrame([[0.1, 2, 9, 0.02, 7.2, 15, 3, 5]], columns=FEATURES)


def test_source_formula_preserves_source_peculiarities():
    frame = measurements()
    assert source_formula(frame)[0] == 100
    frame.loc[0, "Dissolved Oxygen (mg/l)"] = 1
    # Source only checks upper bounds: low oxygen is not penalized.
    assert source_formula(frame)[0] == 100
    frame.loc[0, FEATURES[0]] = 1
    f1 = 100 / 12
    nse = 1 / 12
    f3 = nse / (0.01 * nse + 0.01)
    expected = 100 - np.sqrt(2 * f1**2 + f3**2) / 1.732
    assert source_formula(frame)[0] == pytest.approx(expected)
    frame.loc[0, FEATURES[0]] = np.nan
    assert np.isnan(source_formula(frame)[0])


def test_quality_flags_mask_invalid_but_preserve_pollution_extremes():
    frame = measurements()
    frame.loc[0, FEATURES[0]] = -1
    frame.loc[0, FEATURES[1]] = 255
    frame.loc[0, FEATURES[4]] = 30
    frame.loc[0, FEATURES[5]] = 63
    result = quality_features(frame)
    assert result.shape[1] == 16
    assert result.iloc[0, :8].isna().sum() == 3
    assert result.iloc[0, 8:].sum() == 3
    assert result.loc[0, FEATURES[1]] == 255


def test_detection_limit_is_not_silently_read_as_exact_value():
    frame = measurements().astype(object)
    frame.loc[0, FEATURES[0]] = "<0.01"
    result = quality_features(frame)
    assert np.isnan(result.loc[0, FEATURES[0]])
    assert result.loc[0, f"{FEATURES[0]}__missing"] == 1


def test_missing_schema_has_actionable_error():
    with pytest.raises(ValueError, match="Missing required measurement"):
        quality_features(pd.DataFrame({"pH": [7]}))


def test_test_values_never_change_training_imputation():
    train = pd.concat([measurements()] * 3, ignore_index=True)
    train.loc[2, FEATURES[0]] = np.nan
    pre = SimpleImputer(strategy="median", keep_empty_features=True)
    pre.fit(quality_features(train))
    before = pre.statistics_.copy()
    test = measurements()
    test.loc[0, FEATURES[0]] = 100000
    pre.transform(quality_features(test))
    np.testing.assert_array_equal(before, pre.statistics_)
    assert pre.statistics_[0] == 0.1


def test_temporal_partitions_do_not_overlap_and_dedup_is_traceable():
    frame = pd.concat([measurements()] * 6, ignore_index=True)
    frame["Country"], frame["Area"] = "Example", "Station A"
    frame["Date"] = [
        "31-12-2018",
        "01-01-2019",
        "01-01-2020",
        "01-01-2021",
        "01-01-2021",
        "invalid",
    ]
    frame[TARGET], frame["CCME_WQI"] = 100, "Excellent"
    clean, counts = prepare(frame)
    assert counts == {"exact_duplicates_removed": 1, "invalid_metadata_or_target": 1}
    parts = temporal_split(
        clean,
        {
            "train_end": "2018-12-31",
            "validation_end": "2019-12-31",
            "calibration_end": "2020-12-31",
        },
    )
    assert [len(p) for p in parts.values()] == [1, 1, 1, 1]
    ids = [set(p.source_row) for p in parts.values()]
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            assert not left & right


def test_invalid_split_boundaries_fail():
    with pytest.raises(ValueError, match="increase strictly"):
        temporal_split(
            pd.DataFrame(),
            {"train_end": "2020", "validation_end": "2019", "calibration_end": "2021"},
        )
