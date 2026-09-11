"""Capture local GPU resources and compare an in-context snapshot round trip."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import moderngl

from simulator import shaders
from simulator.renderer import TERRAIN_UNIT
from simulator.material_textures import SCANNED_MATERIAL_UNIT
from tools.probe_facade_draw_threshold import (
    SimulatorApp, SimulationConfig, DEFAULT_SCENARIO_PATH, TARGET_FIRST, TARGET_END,
    VERTEX_BYTES, load_visual_regression_suite, pygame, _zero_light_counts, _bind_opaque_textures,
)
from tools.probe_facade_input_reduction import TRIGGER_FIRST, copy_active_uniforms
from tools.probe_facade_fragment_reduction import fragment_source
from tools.probe_facade_pass_ladder import locate_residual_quad
from tools.facade_snapshot import (
    capture_texture, gpu_info, measure, read_bundle, replay, save_report, write_bundle,
)


def capture(path: Path, output: Path, iterations: int) -> dict:
    if path.exists():
        raise FileExistsError(f"snapshot already exists: {path}")
    if iterations < 2:
        raise ValueError("iterations must be at least 2")
    if locate_residual_quad().vertex_range != (TARGET_FIRST, TARGET_END):
        raise RuntimeError("target geometry changed")
    base = SimulationConfig()
    app = SimulatorApp(replace(base, render=replace(base.render, vsync=False, target_fps=0)),
                       scenario_path=DEFAULT_SCENARIO_PATH)
    programs = {}
    try:
        app.world.shells.clear()
        app.world.stars.count = 0
        load_visual_regression_suite().view("water_reflection").apply(app.camera)
        app.renderer.render(app.world, app.camera, app.celestial, 0.0, None)
        app.ctx.finish()
        _zero_light_counts(app)
        original = app.renderer.scene.program
        buffer = app.renderer.scene.buffers[0]
        raw = buffer.read(3 * VERTEX_BYTES, TRIGGER_FIRST * VERTEX_BYTES)
        raw += buffer.read(6 * VERTEX_BYTES, TARGET_FIRST * VERTEX_BYTES)
        concrete = fragment_source("concrete_scan")
        sources = {"original": shaders.source("scene.frag"), "concrete_scan": concrete,
                   "concrete_combined": fragment_source("combined_no_normal_relief_pattern", concrete)}
        blobs = {"vertices.bin": raw, "scene.vert": shaders.source("scene.vert").encode()}
        manifest = {"schema_version": 1, "gpu": gpu_info(app.ctx),
                    "context_require": 430, "programs": {}, "textures": [],
                    "measurement": {"frame_size": [1280, 720], "region_xywh": [598, 380, 2, 4],
                                    "full_frame_raster": True, "depth_func": "<", "colour_dtype": "f2"}}
        for name, source in sources.items():
            program = original if name == "original" else app.ctx.program(
                vertex_shader=blobs["scene.vert"].decode(), fragment_shader=source)
            programs[name] = program
            if name != "original":
                copy_active_uniforms(original, program)
            key = f"shaders/{name}.frag"
            blobs[key] = source.encode()
            manifest["programs"][name] = {"fragment": key, "uniforms": {
                uniform: program[uniform].read().hex() for uniform in program
                if isinstance(program[uniform], moderngl.Uniform)}}
            samplers = {uniform for uniform in program if isinstance(program[uniform], moderngl.Uniform)
                        and program[uniform].gl_type in (0x8B5E, 0x8DC1)}
            if samplers != {"terrain_height", "scanned_material_texture"}:
                raise RuntimeError(f"unexpected active texture dependencies: {samplers}")
            if program["terrain_height"].value != TERRAIN_UNIT or program["scanned_material_texture"].value != SCANNED_MATERIAL_UNIT:
                raise RuntimeError("sampler units changed")
        manifest["textures"] = [
            capture_texture("terrain", app.renderer.terrain_texture, TERRAIN_UNIT, blobs),
            capture_texture("scanned", app.renderer.scene.scanned_materials.texture, SCANNED_MATERIAL_UNIT, blobs, array=True),
        ]
        if app.ctx.error != "GL_NO_ERROR":
            raise RuntimeError("OpenGL error capturing snapshot")
        write_bundle(path, manifest, blobs)
        report = {"schema_version": 1, "probe": "facade_snapshot_capture_v0_15c", "complete": False,
                  "gpu": gpu_info(app.ctx), "iterations_per_case": iterations,
                  "snapshot_sha256": sha256(path.read_bytes()).hexdigest(),
                  "snapshot_bytes": path.stat().st_size,
                  "manifest": manifest, "live": {}, "roundtrip": {},
                  "limitations": ["Local snapshot includes source texture pixels; not committed or externally submitted.",
                                  "Explicit depth-only state is shared by live, round-trip and independent replay."]}
        save_report(output, report)
        for name, program in programs.items():
            print(f"live {name}", flush=True)
            report["live"][name] = measure(app.ctx, program, raw, lambda: _bind_opaque_textures(app), iterations)
            save_report(output, report)
        manifest, blobs = read_bundle(path)
        report["roundtrip"] = replay(app.ctx, manifest, blobs, iterations)
        report["complete"] = True
        save_report(output, report)
        return report
    finally:
        for name, program in programs.items():
            if name != "original":
                program.release()
        app.audio_executor.shutdown(wait=True, cancel_futures=True)
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1024)
    args = parser.parse_args()
    capture(args.snapshot, args.output, args.iterations)


if __name__ == "__main__":
    main()
