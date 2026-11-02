"""Checks that exploratory.lopo_curves.fast_run_policy reproduces the onsets and Insufficient Evidence states of the
frozen core.run_policy (accumulate=False)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evialert import core
import lopo_curves as LC
rng = np.random.default_rng(1)
for trial in range(20):
    n, T = rng.integers(1, 30), rng.integers(5, 400)
    E = rng.normal(0, 3, (n, T)); use = rng.random((n, T)) > rng.uniform(0, 0.6)
    grid = np.sort(rng.normal(0, 3, rng.integers(1, 40)))
    for gate in (True, False):
        on1, st1 = core.run_policy(E, use, grid, window=6, rho=0.5, refractory=24, gate=gate, accumulate=False)
        on2, ins2 = LC.fast_run_policy(E, use, grid, window=6, rho=0.5, refractory=24, gate=gate, accumulate=False)
        assert np.array_equal(on1, on2)
        assert np.array_equal(st1[0] == core.INSUFF, ins2)
print('fast_run_policy identical to core.run_policy in 40 random cases')
