"""V0-15: audit raster scope and reduce the nine-vertex facade input.

This remains an in-simulator diagnostic, not a standalone driver reproducer.
The reduced BRDF uses V0-11 synthetic material inputs; its output is not expected
to match production. Zero-hit samples do not establish a fix or necessity.

    python -m tools.probe_facade_input_reduction --iterations 1024 --output report.json
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import moderngl
import numpy as np

from tools.probe_facade_draw_threshold import (
    ORIGINAL_BUILDING_VERTICES, TARGET_FIRST, TARGET_END, VERTEX_BYTES,
    SimulatorApp, SimulationConfig, DEFAULT_SCENARIO_PATH, VERTEX_LAYOUT,
    _bind_opaque_textures, _zero_light_counts, load_visual_regression_suite, pygame,
)
from tools.probe_facade_pass_ladder import (
    REGION_HALF_OPEN, _measure_draws, ladder_fragment_source, locate_residual_quad,
)
from tools.probe_facade_shader_reproducer import _context_metadata
from simulator import shaders


TRIGGER_FIRST = ORIGINAL_BUILDING_VERTICES + 11_328
CASES = (
    "baseline_before", "target_only", "target_then_trigger",
    "normal_target", "surface_target", "uv_zero", "position_degenerate",
    "baseline_after",
)


def variant_vertices(vertices: np.ndarray, case: str) -> np.ndarray:
    """Copy the packed input, changing only one trigger attribute group."""
    if case not in CASES:
        raise ValueError(f"unknown input reduction case: {case}")
    if vertices.shape != (9, 10) or vertices.dtype != np.float32:
        raise ValueError("expected nine float32 vertices with ten attributes")
    result = vertices.copy()
    if case == "normal_target":
        result[:3, 3:6] = vertices[3, 3:6]
    elif case == "surface_target":
        result[:3, 6] = vertices[3, 6]
    elif case == "uv_zero":
        result[:3, 7:9] = 0.0
    elif case == "position_degenerate":
        result[:3, :3] = vertices[0, :3]
    return result


def draw_ranges(case: str) -> tuple[tuple[int, int], ...]:
    """Return (first, vertex count) pairs in execution order."""
    if case not in CASES:
        raise ValueError(f"unknown input reduction case: {case}")
    if case == "target_only":
        return ((3, 6),)
    if case == "target_then_trigger":
        return ((3, 6), (0, 3))
    return ((0, 3), (3, 6))


def reduced_sources() -> tuple[str, str]:
    vertex = shaders.source("scene.vert")
    # Preserve production vertex maths/terrain sampling; rename only the two
    # varyings so V0-12 can consume actual geometry instead of captured bounds.
    vertex = vertex.replace("world_position", "ladder_world_position")
    vertex = vertex.replace("world_normal", "ladder_world_normal")
    fragment = ladder_fragment_source(interpolated=True, stage="environment_helper")
    return vertex, fragment


def copy_active_uniforms(source: moderngl.Program, target: moderngl.Program) -> None:
    for name in target:
        member = target[name]
        if not isinstance(member, moderngl.Uniform):
            continue
        if name not in source or len(member.read()) != len(source[name].read()):
            raise RuntimeError(f"cannot copy exact production uniform {name}")
        member.write(source[name].read())


def _save(report: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def add_control_comparisons(report: dict[str, Any]) -> None:
    """A stable stage may still differ from the target-only control in every draw."""
    for name, result in report["stages"].items():
        control_name = name.rsplit("/", 1)[0] + "/target_only"
        control = report["stages"].get(control_name)
        if control is None:
            continue
        control_hashes = {state["sha256"] for state in control["states"]}
        result["target_control"] = control_name
        result["samples_outside_target_control_states"] = sum(
            state["count"] for state in result["states"]
            if state["sha256"] not in control_hashes
        )


def probe(iterations: int, *, output: Path | None = None) -> dict[str, Any]:
    if iterations < 2:
        raise ValueError("iterations must be at least 2")
    if locate_residual_quad().vertex_range != (TARGET_FIRST, TARGET_END):
        raise RuntimeError("residual quad vertex offset changed")
    base = SimulationConfig()
    config = replace(base, render=replace(base.render, vsync=False, target_fps=0))
    app = SimulatorApp(config, scenario_path=DEFAULT_SCENARIO_PATH)
    reduced = None
    try:
        app.world.shells.clear()
        app.world.stars.count = 0
        load_visual_regression_suite().view("water_reflection").apply(app.camera)
        app.renderer.render(app.world, app.camera, app.celestial, 0.0, None)
        app.ctx.finish()
        _zero_light_counts(app)
        original_buffer = app.renderer.scene.buffers[0]
        raw = original_buffer.read(3 * VERTEX_BYTES, TRIGGER_FIRST * VERTEX_BYTES)
        raw += original_buffer.read(6 * VERTEX_BYTES, TARGET_FIRST * VERTEX_BYTES)
        vertices = np.frombuffer(raw, dtype=np.float32).reshape(9, 10)
        vertex_source, fragment_source = reduced_sources()
        reduced = app.ctx.program(vertex_shader=vertex_source, fragment_shader=fragment_source)
        production = app.renderer.scene.program
        copy_active_uniforms(production, reduced)
        report: dict[str, Any] = {
            "schema_version": 1,
            "probe": "facade_input_reduction_v0_15",
            "complete": False,
            "iterations_per_stage": iterations,
            "region_gl_half_open": list(REGION_HALF_OPEN),
            "gpu": _context_metadata(app.ctx),
            "input_sha256": sha256(raw).hexdigest(),
            "source_vertex_ranges": [[TRIGGER_FIRST, TRIGGER_FIRST + 3], [TARGET_FIRST, TARGET_END]],
            "packed_vertices": vertices.tolist(),
            "shader_sha256": {
                "production_vertex": sha256(shaders.source("scene.vert").encode()).hexdigest(),
                "production_fragment": sha256(shaders.source("scene.frag").encode()).hexdigest(),
                "reduced_vertex": sha256(vertex_source.encode()).hexdigest(),
                "reduced_fragment": sha256(fragment_source.encode()).hexdigest(),
            },
            "limitations": [
                "Full simulator initialisation, production uniforms and textures are retained.",
                "Reduced BRDF has synthetic material inputs, not production-equivalent colours.",
                "Sequential finite samples; zero hits do not prove stability or causation.",
                "baseline_before/after use trigger_then_target; only trigger attributes mutate.",
            ],
            "stages": {},
        }
        _save(report, output)
        for shader_name, program in (("production", production), ("environment_brdf", reduced)):
            for scissor_region in (False, True):
                scope = "region" if scissor_region else "full_frame"
                for case in CASES:
                    packed = variant_vertices(vertices, case).tobytes()
                    buffer = app.ctx.buffer(packed)
                    vao = app.ctx.vertex_array(program, [(buffer, *VERTEX_LAYOUT)], skip_errors=True)
                    ranges = draw_ranges(case)

                    def draw() -> None:
                        # Rebind after framebuffer/attachment creation as well.
                        _bind_opaque_textures(app)
                        for first, count in ranges:
                            vao.render(moderngl.TRIANGLES, first=first, vertices=count)

                    name = f"{shader_name}/{scope}/{case}"
                    print(f"measuring {name}", flush=True)
                    try:
                        result = _measure_draws(app.ctx, iterations, draw, depth=True, scissor_region=scissor_region)
                        result["gl_error"] = app.ctx.error
                        if result["gl_error"] != "GL_NO_ERROR":
                            raise RuntimeError(f"{name}: {result['gl_error']}")
                        result["input_sha256"] = sha256(packed).hexdigest()
                        result["draw_ranges_first_count"] = [list(pair) for pair in ranges]
                        report["stages"][name] = result
                        _save(report, output)
                        print(f"  states={result['unique_states']} different={result['differing_iterations']}", flush=True)
                    finally:
                        vao.release()
                        buffer.release()
        report["complete"] = True
        report["varying_stages"] = [name for name, result in report["stages"].items() if not result["bit_deterministic"]]
        add_control_comparisons(report)
        _save(report, output)
        return report
    finally:
        if reduced is not None:
            reduced.release()
        app.audio_executor.shutdown(wait=True, cancel_futures=True)
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("iterations must be at least 2")
    report = probe(args.iterations, output=args.output)
    print(f"varying stages: {report['varying_stages']}")


if __name__ == "__main__":
    main()
