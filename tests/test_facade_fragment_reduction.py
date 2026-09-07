from __future__ import annotations

import moderngl
import pytest

from simulator import shaders
from tools.probe_facade_fragment_reduction import (
    DRAW_CASES, FACADE_START, GENERIC_START, SCAN_CALL, VARIANTS,
    fragment_source, replace_once, replace_span, summarize_program_controls,
)


@pytest.mark.parametrize("variant", VARIANTS)
def test_reductions_preserve_entire_production_facade_path(variant):
    original = shaders.source("scene.frag")
    reduced = fragment_source(variant)
    assert reduced[reduced.index(FACADE_START):] == original[original.index(FACADE_START):]
    assert (reduced == original) == (variant == "recompiled")


def test_reductions_remove_intended_non_facade_work():
    assert SCAN_CALL not in fragment_source("no_scan_call")
    assert "    vec3 tangent_normal = texture(" not in fragment_source("no_scan_normal")
    assert "if (relief.y > 0.0)" not in fragment_source("no_pattern_relief")
    assert "float pattern_value = .5;" in fragment_source("constant_pattern")
    for name in ("non_facade_brdf", "non_facade_constant", "non_facade_discard"):
        reduced = fragment_source(name)
        branch = reduced[reduced.index(GENERIC_START):reduced.index(FACADE_START)]
        assert "apply_scanned_material" not in branch
        assert "dFdx" not in branch
        assert "surface_pattern(" not in branch
    assert "discard;" in fragment_source("non_facade_discard")


def test_reduction_anchor_drift_fails_closed():
    with pytest.raises(ValueError):
        fragment_source("typo")
    for variant in VARIANTS[1:]:
        with pytest.raises(RuntimeError):
            fragment_source(variant, "#version 330\nvoid main() {}")
    with pytest.raises(RuntimeError):
        replace_once("aa", "a", "b")
    with pytest.raises(RuntimeError):
        replace_span("end start", "start", "end", "")


def test_target_controls_bookend_both_draw_orders():
    assert list(DRAW_CASES) == ["target_only", "trigger_then_target", "target_then_trigger", "target_only_after"]
    assert DRAW_CASES["target_only"] == DRAW_CASES["target_only_after"] == ((3, 6),)
    assert DRAW_CASES["trigger_then_target"] == tuple(reversed(DRAW_CASES["target_then_trigger"]))


def test_drifting_target_controls_do_not_support_trigger_verdict():
    stages = {
        f"probe/full_frame/{case}": {
            "states": [{"sha256": "a", "count": 8}],
            "samples_outside_target_control_states": 0 if "only" in case else 8,
        } for case in DRAW_CASES
    }
    report = {"programs": {"probe": {}}, "stages": stages}
    assert summarize_program_controls(report)["probe"]["trigger_effect_observed"] is True
    stages["probe/full_frame/target_only_after"]["states"][0]["sha256"] = "b"
    assert summarize_program_controls(report)["probe"]["trigger_effect_observed"] is None
    del stages["probe/full_frame/target_only_after"]
    assert summarize_program_controls(report) == {}


@pytest.mark.opengl
def test_all_reduced_programs_link_against_production_vertex_shader():
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"OpenGL unavailable: {error}")
    try:
        for variant in VARIANTS:
            program = ctx.program(vertex_shader=shaders.source("scene.vert"), fragment_shader=fragment_source(variant))
            try:
                assert "view_projection" in program
                assert "material_pattern" in program
                assert ctx.error == "GL_NO_ERROR"
            finally:
                program.release()
    finally:
        ctx.release()
