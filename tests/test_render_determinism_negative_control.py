"""The cross-implementation comparison must stay disclosed, not just recorded.

A negative control is only worth anything if the reader can see what differed
between the two runs. The sampler state is the one thing that could not be made
identical — llvmpipe does not expose anisotropic filtering at all — so these
tests fail if a future run drops that disclosure or quietly turns the guard off
on the device where it does apply.
"""

from __future__ import annotations

import json
from pathlib import Path

VALIDATION = Path("docs/validation/render_determinism_v0")
INTEL = VALIDATION / "snapshot_replay_substituted_intel_arc_140v.json"
MESA = VALIDATION / "negative_control_mesa_llvmpipe_21_1_3.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_both_runs_used_the_same_substituted_bundle() -> None:
    intel, mesa = _load(INTEL), _load(MESA)

    assert intel["snapshot_sha256"] == mesa["snapshot_sha256"]
    assert intel["iterations_per_case"] == mesa["iterations_per_case"] == 1024
    # Neither run may have reached into the project.
    assert intel["simulator_modules_loaded"] == []
    assert mesa["simulator_modules_loaded"] == []


def test_the_implementations_are_actually_different() -> None:
    intel, mesa = _load(INTEL), _load(MESA)

    assert "Intel" in intel["gpu"]["GL_RENDERER"]
    assert "llvmpipe" in mesa["gpu"]["GL_RENDERER"]
    assert mesa["gpu"]["GL_VENDOR"] != intel["gpu"]["GL_VENDOR"]


def test_intel_observes_the_trigger_and_llvmpipe_does_not() -> None:
    intel, mesa = _load(INTEL), _load(MESA)

    for program, result in intel["results"].items():
        assert result["target_controls_stable"], program
        assert result["trigger_effect_observed"] is True, program
    for program, result in mesa["results"].items():
        assert result["target_controls_stable"], program
        assert result["trigger_effect_observed"] is False, program

    def outside(report: dict) -> int:
        return sum(
            case["samples_outside_target_control"]
            for result in report["results"].values()
            for case in result["cases"].values()
        )

    assert outside(mesa) == 0
    assert outside(intel) > 1000


def test_llvmpipe_holds_a_single_state_everywhere() -> None:
    mesa = _load(MESA)
    states = {
        state["sha256"]
        for result in mesa["results"].values()
        for case in result["cases"].values()
        for state in case["states"]
    }

    assert len(states) == 1
    for result in mesa["results"].values():
        for case in result["cases"].values():
            assert case["unique_states"] == 1
            assert case["max_abs_rgb_from_target"] == 0.0


def test_the_sampler_difference_is_disclosed_not_hidden() -> None:
    intel, mesa = _load(INTEL), _load(MESA)

    # Where the captured sampler state is reproducible, the guard stays on.
    assert intel.get("sampler_mismatch_allowed") in (False, None)
    # Where it is not, the run must say so and record what it actually got.
    assert mesa["sampler_mismatch_allowed"] is True
    entry = next(s for s in mesa["sampler_state"] if s["texture"] == "scanned")
    assert entry["matched"] is False
    assert entry["anisotropy_requested"] == 8.0
    assert entry["anisotropy_achieved"] != entry["anisotropy_requested"]
