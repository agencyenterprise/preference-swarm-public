"""The block hook's arithmetic on a toy module (needs torch: skipped on the Mac, runs in the box smoke)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from preference_swarm.steering import BlockHook, Steer  # noqa: E402


def test_prefill_steering_and_readout_with_left_padding():
    B, T, d = 2, 5, 4
    h = torch.zeros(B, T, d)
    D = np.eye(d)[:2]                                         # read directions e0, e1
    hook = BlockHook(D)
    v = np.array([1.0, 0.0, 0.0, 0.0])
    # row 0: length 3 (offset 2), steered on its tokens 1..3 by 2.0; row 1: length 5, steered everywhere by 1.0
    hook.prepare([3, 5], [Steer(v, 2.0, [(1, 3)]), Steer(np.array([0.0, 1.0, 0.0, 0.0]), 1.0, None)],
                 [[(1, 3)], [(0, 5)]])
    out = hook(None, None, (h,))[0]
    assert torch.allclose(out[0, :, 0], torch.tensor([0, 0, 0, 2.0, 2.0]))
    assert torch.allclose(out[1, :, 1], torch.ones(5))
    assert np.allclose(hook.readings[0, 0], [2.0, 0.0])      # read AFTER steering
    assert np.allclose(hook.readings[1, 0], [0.0, 1.0])


def test_decode_steps_keep_only_everywhere_rows():
    hook = BlockHook(None)
    hook.prepare([3, 3], [Steer(np.array([1.0, 0.0]), 1.0, [(0, 3)]), Steer(np.array([0.0, 1.0]), 3.0, None)])
    hook(None, None, torch.zeros(2, 3, 2))                    # prefill
    step = hook(None, None, torch.zeros(2, 1, 2))             # one decode step
    assert torch.allclose(step[0, 0], torch.zeros(2)) and torch.allclose(step[1, 0], torch.tensor([0.0, 3.0]))


def test_unsteered_batch_is_identity():
    hook = BlockHook(None)
    hook.prepare([2], [None])
    x = torch.randn(1, 2, 3)
    assert torch.equal(hook(None, None, x), x)
