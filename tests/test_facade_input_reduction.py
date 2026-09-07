from __future__ import annotations

import moderngl
import numpy as np
import pytest

from tools.probe_facade_input_reduction import (
    CASES, add_control_comparisons, draw_ranges, reduced_sources, variant_vertices,
)
from tools.probe_facade_pass_ladder import FRAME_SIZE, REGION_GL, _measure_draws
from tools.probe_facade_shader_reproducer import _VERTEX_SHADER


@pytest.mark.parametrize("case", CASES)
def test_variants_preserve_target_and_change_only_named_trigger_fields(case):
    vertices = np.arange(90, dtype=np.float32).reshape(9, 10)
    original = vertices.copy()
    result = variant_vertices(vertices, case)
    np.testing.assert_array_equal(vertices, original)
    np.testing.assert_array_equal(result[3:], original[3:])
    expected = original.copy()
    if case == "normal_target":
        expected[:3, 3:6] = original[3, 3:6]
    elif case == "surface_target":
        expected[:3, 6] = original[3, 6]
    elif case == "uv_zero":
        expected[:3, 7:9] = 0
    elif case == "position_degenerate":
        expected[:3, :3] = original[0, :3]
    np.testing.assert_array_equal(result, expected)
    assert not np.shares_memory(result, vertices)


def test_draw_controls_cover_correct_geometry_and_order():
    assert draw_ranges("target_only") == ((3, 6),)
    assert draw_ranges("target_then_trigger") == ((3, 6), (0, 3))
    assert draw_ranges("baseline_before") == draw_ranges("baseline_after") == ((0, 3), (3, 6))
    with pytest.raises(ValueError):
        draw_ranges("typo")
    with pytest.raises(ValueError):
        variant_vertices(np.zeros((8, 10), dtype=np.float32), "target_only")


def test_control_comparison_detects_stable_but_different_output():
    report = {"stages": {
        "shader/full/target_only": {"states": [{"sha256": "a", "count": 8}]},
        "shader/full/baseline": {"states": [{"sha256": "b", "count": 8}]},
        "shader/full/mixed": {"states": [{"sha256": "a", "count": 5}, {"sha256": "b", "count": 3}]},
        "shader/region/target_only": {"states": [{"sha256": "b", "count": 8}]},
    }}
    add_control_comparisons(report)
    assert [stage["samples_outside_target_control_states"] for stage in report["stages"].values()] == [0, 8, 3, 0]


@pytest.fixture
def gl_context():
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"OpenGL unavailable: {error}")
    yield ctx
    ctx.release()


@pytest.mark.opengl
@pytest.mark.parametrize("scissor_region", [True, False])
def test_measure_draws_applies_scissor_to_bound_fbo_and_restores_previous(gl_context, scissor_region):
    ctx = gl_context
    texture = ctx.texture((16, 16), 4)
    previous = ctx.framebuffer([texture])
    previous.use()
    ctx.viewport = (0, 0, 12, 12)
    ctx.scissor = (1, 2, 3, 4)
    program = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader="""#version 330
        out vec4 frag_color;
        void main() { frag_color = vec4(1.0); }
    """)
    vao = ctx.vertex_array(program, [])
    samples = []

    def draw():
        assert ctx.fbo is not previous
        query = ctx.query(samples=True)
        with query:
            vao.render(moderngl.TRIANGLES, vertices=3)
        samples.append(query.samples)

    try:
        result = _measure_draws(ctx, 2, draw, depth=False, scissor_region=scissor_region)
        assert samples == [8 if scissor_region else FRAME_SIZE[0] * FRAME_SIZE[1]] * 2
        assert result["effective_scissor_xywh"] == list(REGION_GL if scissor_region else (0, 0, *FRAME_SIZE))
        assert result["bit_deterministic"]
        assert ctx.fbo is previous
        assert ctx.viewport == (0, 0, 12, 12)
        assert ctx.scissor == (1, 2, 3, 4)
    finally:
        vao.release()
        program.release()
        previous.release()
        texture.release()


@pytest.mark.opengl
def test_reduced_program_links_actual_vertex_varyings(gl_context):
    vertex, fragment = reduced_sources()
    program = gl_context.program(vertex_shader=vertex, fragment_shader=fragment)
    try:
        assert "terrain_height" in program
        assert "view_projection" in program
        assert "camera_position" in program
        assert "static_light_count" not in program
    finally:
        program.release()
