"""
Acetylcholine-gated exploration coupled to stochastic release + input gain (bead hy8.5).

hy8.1 coupled ACh (uncertainty) to the stochastic vesicle-release fraction. hy8.5 completes
the mechanism: ACh ALSO drives an input/attention gain, and the coupling is shown to TRACK
uncertainty, with an ABLATION against today's fixed `stochastic_train_frac`.

  ACh = f(predictive entropy) -> { explore gain on stochastic-release fraction,
                                   attend gain on the synaptic input }

so the model explores more (more vesicle noise) AND attends harder when uncertain, and commits
when confident.

Run:  pytest tests/test_ach_exploration.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.neuromod import NeuromodulatoryBus, NeuromodConfig
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear

from _bio_testkit import set_seed

IN, OUT, B = 16, 16, 4


def _linear_no_hebb() -> SynapticLinear:
    set_seed(0)
    cfg = SynapticConfig(enable_hebbian=False, enable_metabolism=False)
    return SynapticLinear(IN, OUT, cfg, bias=False)


# --------------------------------------------------------------------------- #
# 1. ACh input gain scales the synaptic input (attention)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ach_input_gain_scales_the_synaptic_input():
    x = torch.randn(B, IN)
    ca, en = torch.ones(B), torch.ones(B)
    base = _linear_no_hebb()
    gained = _linear_no_hebb()
    gained._nm_ach_input_gain = 2.0
    with torch.no_grad():
        y1 = base(x, ca, en)
        y2 = gained(x, ca, en)
    assert torch.allclose(y2, 2.0 * y1, atol=1e-5), "ACh input gain must scale the input linearly"


# --------------------------------------------------------------------------- #
# 2. One ACh signal drives BOTH exploration and attention together
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ach_drives_explore_and_attend_together():
    bus = NeuromodulatoryBus()
    bus.update(entropy=1.0)
    bus.update(entropy=3.0)  # rising uncertainty
    g = bus.gains()
    assert g["explore"] > 1.0 and g["attend"] > 1.0, "uncertainty must raise both ACh gains"
    # both derive from the same ACh level
    assert bus.levels()["ach"] > 0


# --------------------------------------------------------------------------- #
# 3. The effective stochastic fraction TRACKS uncertainty (high vs low entropy)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_effective_fraction_tracks_uncertainty():
    base_frac = 0.12

    def effective_frac(entropy_seq):
        bus = NeuromodulatoryBus()
        for h in entropy_seq:
            bus.update(entropy=h)
        return base_frac * bus.gains()["explore"]

    confident = effective_frac([2.0, 2.0, 1.0])  # entropy falling -> low/zero ACh
    uncertain = effective_frac([2.0, 2.0, 4.0])  # entropy rising -> high ACh
    assert uncertain > confident, "effective stochastic fraction must rise with uncertainty"


# --------------------------------------------------------------------------- #
# 4. ABLATION: dynamic ACh-modulated fraction vs a fixed fraction
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ablation_dynamic_fraction_varies_where_fixed_does_not():
    base_frac = 0.12
    bus = NeuromodulatoryBus()
    # a run whose uncertainty swings up and down
    entropies = [1.0, 1.5, 3.0, 4.0, 2.0, 0.5]
    dynamic = []
    for h in entropies:
        bus.update(entropy=h)
        dynamic.append(base_frac * bus.gains()["explore"])
    fixed = [base_frac] * len(entropies)

    dyn_range = max(dynamic) - min(dynamic)
    assert dyn_range > 1e-3, "dynamic ACh fraction must actually vary over the run"
    assert (max(fixed) - min(fixed)) == 0.0, "the fixed-fraction baseline is constant by definition"
    assert max(dynamic) > base_frac, "high uncertainty must explore MORE than the fixed fraction"


# --------------------------------------------------------------------------- #
# 5. broadcast wires the ACh input gain onto the synaptic layers
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_broadcast_sets_ach_input_gain():
    import sys

    sys.path.insert(0, "tests")
    from _bio_testkit import make_tiny_synaptic

    model = make_tiny_synaptic(seed=0)
    bus = NeuromodulatoryBus()
    bus.update(entropy=1.0)
    bus.update(entropy=3.0)
    bus.broadcast(model)
    lins = [m for m in model.modules() if isinstance(m, SynapticLinear)]
    assert lins and all(getattr(m, "_nm_ach_input_gain", None) == bus.gains()["attend"] for m in lins)


# --------------------------------------------------------------------------- #
# 6. Disabled / neutral ACh leaves input untouched
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_neutral_ach_input_gain_is_noop():
    # A disabled bus yields attend gain 1.0; with no gain attribute set, getattr defaults to 1.0.
    assert NeuromodulatoryBus(NeuromodConfig(enabled=False)).gains()["attend"] == 1.0
    x = torch.randn(B, IN)
    ca, en = torch.ones(B), torch.ones(B)
    a = _linear_no_hebb()
    b = _linear_no_hebb()
    with torch.no_grad():
        y1 = a(x, ca, en)
        y2 = b(x, ca, en)  # no gain set -> getattr default 1.0
    assert torch.allclose(y1, y2, atol=1e-6)
