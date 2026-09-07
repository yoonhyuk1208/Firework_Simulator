"""Add real scene-pass state to the V0-11 facade reproducer one step at a time.

The V0-11 2x4 shader stayed bit-identical while the full scene continued to
vary.  This probe locates the two source triangles under the known residual,
then compares five cumulative conditions at the original 1280x720 pixel
location: large viewport, real triangle interpolation, depth, the complete
production scene program, and production program plus depth.

Example::

    python -m tools.probe_facade_pass_ladder --iterations 1024
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Callable

import moderngl
import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from simulator.app import SimulatorApp
from simulator.camera_optics import vertical_fov_deg
from simulator.config import PhysicalCameraConfig, SimulationConfig
from simulator.passes.scene import VERTEX_LAYOUT
from simulator.renderer import (
    FAR_PLANE_M,
    NEAR_PLANE_M,
    SCENE_ASSET,
    TERRAIN_UNIT,
    _look_at,
    _perspective,
)
from simulator.scene import load_scene
from simulator.scenario import DEFAULT_SCENARIO_PATH
from simulator.terrain import sample_heightmap_array
from simulator.validation.views import (
    DEFAULT_VISUAL_VIEWS_PATH,
    load_visual_regression_suite,
)
from tools.probe_facade_shader_reproducer import (
    _VERTEX_SHADER,
    _context_metadata,
    _summarize_states,
    fragment_source,
)


FRAME_SIZE = (1280, 720)
REGION_GL = (598, 380, 2, 4)
REGION_HALF_OPEN = (598, 380, 600, 384)
STAGES = (
    "viewport_constants",
    "triangle_interpolation",
    "triangle_depth",
    "production_program",
    "production_program_depth",
)

_LOCAL_PIXEL_INDEX = """return (gl_FragCoord.x - .5) + (gl_FragCoord.y - .5) * 2.0;"""
_FRAME_PIXEL_INDEX = """return (gl_FragCoord.x - 598.5)
         + (gl_FragCoord.y - 380.5) * 2.0;"""
_CAPTURED_NORMAL = """vec3 captured_normal() {
    return vec3(-.98124694, 0.0, -.19287110);
}"""
_INTERPOLATED_NORMAL = """vec3 captured_normal() {
    return ladder_world_normal;
}"""
_CAPTURED_POSITION = """vec3 captured_position() {
    return mix(vec3(814.0, 20.0, -749.0),
               vec3(814.0, 26.0, -746.0), pixel_mix());
}"""
_INTERPOLATED_POSITION = """vec3 captured_position() {
    return ladder_world_position;
}"""

_INTERPOLATING_VERTEX_SHADER = """#version 330
in vec3 in_position;
in vec3 in_normal;
uniform mat4 view_projection;
out vec3 ladder_world_position;
out vec3 ladder_world_normal;
void main() {
    ladder_world_position = in_position;
    ladder_world_normal = in_normal;
    gl_Position = view_projection * vec4(in_position, 1.0);
}
"""


@dataclass(frozen=True, slots=True)
class ResidualQuad:
    """The two frontmost facade triangles covering the known residual."""

    triangle_indices: tuple[int, int]
    vertex_range: tuple[int, int]
    source_vertices: np.ndarray
    world_vertices: np.ndarray
    screen_xy: np.ndarray
    view_projection: np.ndarray


def _view_projection() -> tuple[np.ndarray, np.ndarray]:
    suite = load_visual_regression_suite(DEFAULT_VISUAL_VIEWS_PATH)
    view = suite.view("water_reflection")
    camera_position = np.asarray(view.position_eus_m, dtype=np.float32)
    target = np.asarray(view.target_eus_m, dtype=np.float32)
    projection = _perspective(
        vertical_fov_deg(PhysicalCameraConfig()),
        FRAME_SIZE[0] / FRAME_SIZE[1],
        NEAR_PLANE_M,
        FAR_PLANE_M,
    )
    return projection @ _look_at(camera_position, target), camera_position


def _overlaps_region(screen_xy: np.ndarray) -> np.ndarray:
    left, bottom, right, top = REGION_HALF_OPEN
    return (
        (screen_xy[:, :, 0].max(axis=1) >= left + 0.5)
        & (screen_xy[:, :, 0].min(axis=1) <= right - 0.5)
        & (screen_xy[:, :, 1].max(axis=1) >= bottom + 0.5)
        & (screen_xy[:, :, 1].min(axis=1) <= top - 0.5)
    )


def locate_residual_quad(scene_path: Path = SCENE_ASSET) -> ResidualQuad:
    """Find the nearest coplanar facade pair over the known eight pixels."""

    scene = load_scene(scene_path)
    triangles = scene.building_vertices.reshape(-1, 3, 10)
    world = triangles[:, :, :3].astype(np.float64, copy=True)
    flat_world = world.reshape(-1, 3)
    flat_world[:, 1] += sample_heightmap_array(
        scene.terrain_height_m,
        scene.terrain_bounds,
        flat_world[:, [0, 2]],
    )
    view_projection, _ = _view_projection()
    homogeneous = np.concatenate(
        (world, np.ones((*world.shape[:2], 1), dtype=np.float64)), axis=2
    )
    clip = np.einsum(
        "ij,tkj->tki", view_projection.astype(np.float64), homogeneous
    )
    ndc = clip[:, :, :3] / clip[:, :, 3:4]
    screen_xy = np.stack(
        (
            (ndc[:, :, 0] * 0.5 + 0.5) * FRAME_SIZE[0],
            (ndc[:, :, 1] * 0.5 + 0.5) * FRAME_SIZE[1],
        ),
        axis=2,
    )
    candidates = np.nonzero(
        _overlaps_region(screen_xy)
        & (clip[:, :, 3].min(axis=1) > 0.0)
        & np.all(np.isclose(triangles[:, :, 6], 0.0), axis=1)
    )[0]
    if len(candidates) < 2:
        raise RuntimeError("fewer than two facade triangles cover the residual")
    depths = ndc[candidates, :, 2].mean(axis=1)
    nearest = candidates[np.argsort(depths)[:2]]
    nearest.sort()
    normals = triangles[nearest, :, 3:6]
    if nearest[1] != nearest[0] + 1 or not np.allclose(
        normals, normals[0, 0], atol=2e-6
    ):
        raise RuntimeError("frontmost residual triangles are no longer one quad")
    expected_normal = np.array(
        [-0.98124694, 0.0, -0.19287110], dtype=np.float32
    )
    if float(np.dot(normals[0, 0], expected_normal)) < 0.99999:
        raise RuntimeError("residual quad normal no longer matches V0-10 capture")
    start = int(nearest[0] * 3)
    end = int((nearest[1] + 1) * 3)
    return ResidualQuad(
        triangle_indices=(int(nearest[0]), int(nearest[1])),
        vertex_range=(start, end),
        source_vertices=scene.building_vertices[start:end].copy(),
        world_vertices=world[nearest].reshape(-1, 3).astype(np.float32),
        screen_xy=screen_xy[nearest].copy(),
        view_projection=view_projection,
    )


def ladder_fragment_source(*, interpolated: bool, stage: str = "final_facade") -> str:
    """Adapt a V0-11 shader stage to the original framebuffer coordinates."""

    source = fragment_source(stage)
    replacements = [(_LOCAL_PIXEL_INDEX, _FRAME_PIXEL_INDEX)]
    if interpolated:
        source = source.replace(
            "out vec4 frag_color;",
            "in vec3 ladder_world_position;\n"
            "in vec3 ladder_world_normal;\n"
            "out vec4 frag_color;",
            1,
        )
        replacements.extend(
            (
                (_CAPTURED_NORMAL, _INTERPOLATED_NORMAL),
                (_CAPTURED_POSITION, _INTERPOLATED_POSITION),
            )
        )
    for original, replacement in replacements:
        if source.count(original) != 1:
            raise RuntimeError("V0-11 standalone shader source contract changed")
        source = source.replace(original, replacement, 1)
    return source


def _set_uniform(program: moderngl.Program, name: str, value: Any) -> None:
    if name in program:
        program[name].value = value


def _configure_simplified_program(
    program: moderngl.Program, view_projection: np.ndarray, camera_position: np.ndarray
) -> None:
    _set_uniform(program, "camera_position", tuple(camera_position))
    _set_uniform(program, "ambient_irradiance_w_m2", 0.012)
    _set_uniform(program, "static_light_count", 0)
    _set_uniform(program, "static_light_color", (1.0, 0.82, 0.62))
    _set_uniform(program, "static_light_power_w", 0.0)
    _set_uniform(program, "dynamic_light_count", 0)
    if "view_projection" in program:
        program["view_projection"].write(
            view_projection.T.astype(np.float32).tobytes()
        )


def _read_region(framebuffer: moderngl.Framebuffer) -> bytes:
    return framebuffer.read(
        viewport=REGION_GL,
        components=4,
        alignment=1,
        dtype="f2",
    )


def _state_differences(states: list[bytes]) -> list[dict[str, Any]]:
    """Describe every minority fp16 state relative to the dominant state."""

    counts = Counter(states)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) <= 1:
        return []
    reference = np.frombuffer(ordered[0][0], dtype=np.float16).reshape(4, 2, 4)
    details: list[dict[str, Any]] = []
    for raw, count in ordered[1:]:
        candidate = np.frombuffer(raw, dtype=np.float16).reshape(4, 2, 4)
        delta = np.abs(
            candidate[:, :, :3].astype(np.float32)
            - reference[:, :, :3].astype(np.float32)
        )
        mask = delta.max(axis=2) > 0.0
        rows, columns = np.nonzero(mask)
        details.append(
            {
                "sha256": sha256(raw).hexdigest(),
                "count": count,
                "differing_pixels": int(mask.sum()),
                "max_abs_rgb": float(delta.max()),
                "local_rows": [int(rows.min()), int(rows.max())],
                "local_columns": [int(columns.min()), int(columns.max())],
            }
        )
    return details


def _measure_draws(
    ctx: moderngl.Context,
    iterations: int,
    draw: Callable[[], None],
    *,
    depth: bool,
    scissor_region: bool = True,
) -> dict[str, Any]:
    if iterations < 2:
        raise ValueError("iterations must be at least 2")
    previous_framebuffer = ctx.fbo
    previous_viewport = ctx.viewport
    previous_scissor = ctx.scissor
    colour = ctx.texture(FRAME_SIZE, components=4, dtype="f2")
    depth_texture = ctx.depth_texture(FRAME_SIZE) if depth else None
    framebuffer = ctx.framebuffer(
        [colour], depth_attachment=depth_texture
    )
    try:
        # ModernGL stores viewport/scissor per framebuffer. Setting ctx.scissor
        # before use() configures the previous target and is lost at the bind.
        framebuffer.use()
        framebuffer.viewport = (0, 0, *FRAME_SIZE)
        framebuffer.scissor = REGION_GL if scissor_region else None
        ctx.disable(moderngl.BLEND)
        if depth:
            ctx.enable(moderngl.DEPTH_TEST)
        else:
            ctx.disable(moderngl.DEPTH_TEST)
        states: list[bytes] = []
        first: np.ndarray | None = None
        for _ in range(iterations):
            framebuffer.use()
            if ctx.scissor != (REGION_GL if scissor_region else (0, 0, *FRAME_SIZE)):
                raise RuntimeError("probe framebuffer scissor changed")
            framebuffer.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
            draw()
            ctx.finish()
            raw = _read_region(framebuffer)
            states.append(raw)
            if first is None:
                first = np.frombuffer(raw, dtype=np.float16).reshape(4, 2, 4)
        summary = _summarize_states(states, iterations)
        summary["raster_scope"] = "region" if scissor_region else "full_frame"
        summary["effective_scissor_xywh"] = list(ctx.scissor)
        summary["differences_from_dominant"] = _state_differences(states)
        assert first is not None
        summary["first_rgb_min"] = (
            first[:, :, :3].min(axis=(0, 1)).astype(float).tolist()
        )
        summary["first_rgb_max"] = (
            first[:, :, :3].max(axis=(0, 1)).astype(float).tolist()
        )
        return summary
    finally:
        if previous_framebuffer is not None:
            previous_framebuffer.use()
        ctx.viewport = previous_viewport
        ctx.scissor = previous_scissor
        framebuffer.release()
        if depth_texture is not None:
            depth_texture.release()
        colour.release()


def _simplified_stage(
    ctx: moderngl.Context,
    quad: ResidualQuad,
    camera_position: np.ndarray,
    iterations: int,
    *,
    interpolated: bool,
    depth: bool,
) -> dict[str, Any]:
    vertex_shader = _INTERPOLATING_VERTEX_SHADER if interpolated else _VERTEX_SHADER
    program = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=ladder_fragment_source(interpolated=interpolated),
    )
    buffer: moderngl.Buffer | None = None
    if interpolated:
        packed = np.concatenate(
            (quad.world_vertices, quad.source_vertices[:, 3:6]), axis=1
        ).astype(np.float32)
        buffer = ctx.buffer(packed.tobytes())
        vao = ctx.vertex_array(
            program, [(buffer, "3f 3f", "in_position", "in_normal")]
        )
    else:
        vao = ctx.vertex_array(program, [])
    try:
        _configure_simplified_program(
            program, quad.view_projection, camera_position
        )
        return _measure_draws(
            ctx,
            iterations,
            lambda: vao.render(moderngl.TRIANGLES, vertices=6 if interpolated else 3),
            depth=depth,
        )
    finally:
        vao.release()
        if buffer is not None:
            buffer.release()
        program.release()


def _production_stage(
    app: SimulatorApp,
    quad: ResidualQuad,
    iterations: int,
    *,
    depth: bool,
) -> dict[str, Any]:
    renderer = app.renderer
    program = renderer.scene.program
    buffer = app.ctx.buffer(quad.source_vertices.astype(np.float32).tobytes())
    vao = app.ctx.vertex_array(program, [(buffer, *VERTEX_LAYOUT)])
    try:
        program["static_light_count"] = 0
        program["dynamic_light_count"] = 0
        renderer.terrain_texture.use(TERRAIN_UNIT)
        renderer.scene.scanned_materials.bind()
        return _measure_draws(
            app.ctx,
            iterations,
            lambda: vao.render(moderngl.TRIANGLES, vertices=6),
            depth=depth,
        )
    finally:
        vao.release()
        buffer.release()


def probe(iterations: int) -> dict[str, Any]:
    """Execute the cumulative pass-state ladder in one production GL context."""

    if iterations < 2:
        raise ValueError("iterations must be at least 2")
    quad = locate_residual_quad()
    base = SimulationConfig()
    config = replace(base, render=replace(base.render, vsync=False, target_fps=0))
    app = SimulatorApp(config, scenario_path=DEFAULT_SCENARIO_PATH)
    try:
        app.world.shells.clear()
        app.world.stars.count = 0
        view = load_visual_regression_suite().view("water_reflection")
        view.apply(app.camera)
        # Initialise the exact production uniform and texture state once. The
        # ladder draws below target their own framebuffer and never alter the
        # shipped render path.
        app.renderer.render(app.world, app.camera, app.celestial, 0.0, None)
        app.ctx.finish()
        camera_position = app.camera.position_m.copy()
        results = {
            "viewport_constants": _simplified_stage(
                app.ctx,
                quad,
                camera_position,
                iterations,
                interpolated=False,
                depth=False,
            ),
            "triangle_interpolation": _simplified_stage(
                app.ctx,
                quad,
                camera_position,
                iterations,
                interpolated=True,
                depth=False,
            ),
            "triangle_depth": _simplified_stage(
                app.ctx,
                quad,
                camera_position,
                iterations,
                interpolated=True,
                depth=True,
            ),
            "production_program": _production_stage(
                app, quad, iterations, depth=False
            ),
            "production_program_depth": _production_stage(
                app, quad, iterations, depth=True
            ),
        }
        varying = [stage for stage in STAGES if not results[stage]["bit_deterministic"]]
        return {
            "schema_version": 1,
            "probe": "facade_pass_state_ladder_v0_12",
            "iterations_per_stage": iterations,
            "frame_size": list(FRAME_SIZE),
            "region_gl": list(REGION_HALF_OPEN),
            "gpu": _context_metadata(app.ctx),
            "quad": {
                "triangle_indices": list(quad.triangle_indices),
                "vertex_range": list(quad.vertex_range),
                "source_normal": quad.source_vertices[0, 3:6].astype(float).tolist(),
                "screen_xy": quad.screen_xy.astype(float).tolist(),
            },
            "stages": results,
            "verdict": {
                "first_varying_stage": varying[0] if varying else None,
                "all_stages_bit_deterministic": not varying,
            },
        }
    finally:
        app.audio_executor.shutdown(wait=True, cancel_futures=True)
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    try:
        report = probe(arguments.iterations)
    except ValueError as error:
        parser.error(str(error))
    print(f"{report['gpu']['GL_VENDOR']} / {report['gpu']['GL_RENDERER']}")
    print(
        f"quad triangles={report['quad']['triangle_indices']} "
        f"vertices={report['quad']['vertex_range']}"
    )
    for stage, result in report["stages"].items():
        state = "stable" if result["bit_deterministic"] else "VARYING"
        print(
            f"{stage:26s} {state:7s}  "
            f"states={result['unique_states']}  "
            f"different={result['differing_iterations']}"
        )
    first = report["verdict"]["first_varying_stage"]
    print(f"verdict: first varying stage={first or 'none'}")
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
