import numpy as np

from app.calibration import calibrate_minmax, calibrate_zscore, compute_entropy


def test_entropy_preserves_normalization_effects():
    scores = np.array([0.0, 1.0, 10.0, 25.0])

    raw_entropy = compute_entropy(scores)
    minmax_entropy = compute_entropy(calibrate_minmax(scores))
    zscore_entropy = compute_entropy(calibrate_zscore(scores))

    assert raw_entropy != minmax_entropy
    assert zscore_entropy != raw_entropy
    assert zscore_entropy != minmax_entropy
