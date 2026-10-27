"""Exercise both native runtimes together: incompatible OpenMP builds can crash."""

import subprocess
import sys


def test_tree_and_neural_runtimes_coexist():
    code = """
import numpy as np
import torch
from xgboost import XGBRegressor
from water_quality.models import MLP
torch.set_num_threads(4)
x = np.random.default_rng(42).normal(size=(256, 16)).astype('float32')
y = x[:, 0] * 2 + x[:, 1]
tree = XGBRegressor(n_estimators=5, max_depth=2, n_jobs=4, tree_method='hist').fit(x, y)
assert np.isfinite(tree.predict(x)).all()
net = MLP()
loss = (net(torch.tensor(x)) - torch.tensor(y)).square().mean()
loss.backward()
assert torch.isfinite(loss)
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
