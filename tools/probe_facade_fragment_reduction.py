"""V0-15B: reduce non-facade fragment work while keeping the facade source.

All shader changes are confined to diagnostic programs. Each program has its
own target-only controls, since recompilation itself can change numeric output.
This probe retains production initialisation, uniforms, textures and vertex GLSL.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import moderngl

from simulator import shaders
from tools.probe_facade_draw_threshold import (
    SimulatorApp, SimulationConfig, DEFAULT_SCENARIO_PATH, VERTEX_LAYOUT,
    TARGET_FIRST, TARGET_END, VERTEX_BYTES, load_visual_regression_suite, pygame,
    _bind_opaque_textures, _zero_light_counts,
)
from tools.probe_facade_input_reduction import (
    TRIGGER_FIRST, _save, add_control_comparisons, copy_active_uniforms,
)
from tools.probe_facade_pass_ladder import (
    REGION_HALF_OPEN, _measure_draws, locate_residual_quad,
)
from tools.probe_facade_shader_reproducer import _context_metadata


VARIANTS = (
    "recompiled", "no_scan_call", "no_scan_normal", "no_pattern_relief",
    "constant_pattern", "non_facade_brdf", "non_facade_constant",
    "non_facade_discard",
    "scan_brdf", "scan_brdf_no_normal", "scan_brdf_albedo_only",
    "scan_brdf_arm_only", "scan_brdf_fixed_layer",
    "combined_no_normal_relief_pattern", "concrete_scan",
)
GENERIC_START = "    if (kind != PATTERN_FACADE) {"
FACADE_START = "    if (surface > .5) {"
SCAN_CALL = """        apply_scanned_material(
            scanned_layer, micro_detail, scanned_colour_strength,
            n, albedo, reflectance
        );"""
DRAW_CASES = {
    "target_only": ((3, 6),),
    "trigger_then_target": ((0, 3), (3, 6)),
    "target_then_trigger": ((3, 6), (0, 3)),
    "target_only_after": ((3, 6),),
}


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"fragment reduction source contract changed: {old[:60]}")
    return source.replace(old, new, 1)


def replace_span(source: str, start: str, end: str, replacement: str) -> str:
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError("fragment reduction span contract changed")
    first, last = source.index(start), source.index(end)
    if last <= first:
        raise RuntimeError("fragment reduction span order changed")
    return source[:first] + replacement + source[last:]


def fragment_source(variant: str, source: str | None = None) -> str:
    """Fail closed if a reduction anchor drifts; preserve the facade suffix."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown fragment reduction: {variant}")
    source = shaders.source("scene.frag") if source is None else source
    if variant == "recompiled":
        return source
    if variant == "combined_no_normal_relief_pattern":
        for part in ("no_scan_normal", "no_pattern_relief", "constant_pattern"):
            source = fragment_source(part, source)
        return source
    if variant == "concrete_scan":
        source = replace_span(
            source, "        if (material == 3) {",
            "        vec3 radiance = reflected_radiance(n, albedo, reflectance);",
            """        if (material == 5 || material == 12) {
            float pores = .5;
            float stains = .5;
            if (micro_detail > .001) {
                pores = fbm2(world_position.xz * 3.1);
                stains = value_noise(world_position.xz * .11 + 8.0);
            }
            albedo *= mix(.82, 1.14, pores * micro_detail + .5 * (1.0-micro_detail));
            albedo *= mix(.78, 1.03, smoothstep(
                .28, .72, mix(.5, stains, micro_detail)
            ));
        }
""",
        )
        return replace_span(source, "        if (material == 2) {",
                            "        radiance += emissive.rgb", "")
    if variant == "no_scan_call":
        return replace_once(source, SCAN_CALL, "        // Probe: scanned-material call removed.")
    if variant == "no_scan_normal":
        return replace_span(source, "    vec3 tangent_normal = texture(",
                            "    // Poly Haven ARM maps", "")
    if variant == "no_pattern_relief":
        return replace_once(source, "if (relief.y > 0.0)", "if (false)")
    if variant == "constant_pattern":
        return replace_once(source, "float pattern_value = surface_pattern(kind, pattern.yz, n);",
                            "float pattern_value = .5;")
    if variant.startswith("scan_brdf"):
        body = """        vec3 albedo = material_base_primary[material];
        float detail = 1.0 - smoothstep(
            90.0, 420.0, length(camera_position - world_position)
        );
        apply_scanned_material(
            scanned_layer_for_material(material), detail, .74,
            n, albedo, reflectance
        );
        frag_color = vec4(reflected_radiance(n, albedo, reflectance), 1.0);
        return;"""
        source = replace_span(source, GENERIC_START, FACADE_START,
                              GENERIC_START + "\n" + body + "\n    }\n\n")
        if variant in {"scan_brdf_no_normal", "scan_brdf_albedo_only"}:
            source = replace_span(source, "    vec3 tangent_normal = texture(",
                                  "    // Poly Haven ARM maps", "")
        if variant == "scan_brdf_albedo_only":
            source = replace_span(source, "    // Poly Haven ARM maps",
                                  "// Trowbridge-Reitz GGX", "}\n")
        if variant == "scan_brdf_arm_only":
            source = replace_span(source, "    vec3 scanned_albedo = pow(texture(",
                                  "    // Poly Haven ARM maps", "")
        if variant == "scan_brdf_fixed_layer":
            source = replace_once(source, "scanned_layer_for_material(material), detail, .74,",
                                  "3, detail, .74,")
        return source
    body = {
        "non_facade_brdf": """        frag_color = vec4(reflected_radiance(
            n, material_base_primary[material], reflectance
        ), 1.0);
        return;""",
        "non_facade_constant": "        frag_color = vec4(.001, .002, .003, 1.0);\n        return;",
        "non_facade_discard": "        discard;",
    }[variant]
    return replace_span(source, GENERIC_START, FACADE_START,
                        GENERIC_START + "\n" + body + "\n    }\n\n")


def summarize_program_controls(report: dict[str, Any]) -> dict[str, Any]:
    """Do not classify a trigger effect when its own target controls drift."""
    stages = report["stages"]
    summaries = {}
    for variant in report["programs"]:
        prefix = f"{variant}/full_frame/"
        if not all(prefix + case in stages for case in DRAW_CASES):
            continue
        before, after = (stages[prefix + case] for case in ("target_only", "target_only_after"))
        hashes_before = {state["sha256"] for state in before["states"]}
        hashes_after = {state["sha256"] for state in after["states"]}
        stable = len(hashes_before) == 1 and hashes_before == hashes_after
        affected = [case for case in ("trigger_then_target", "target_then_trigger")
                    if stages[prefix + case]["samples_outside_target_control_states"] > 0]
        summaries[variant] = {
            "target_controls_stable_and_matching": stable,
            "trigger_effect_observed": bool(affected) if stable else None,
            "orders_outside_target_control": affected,
        }
    return summaries


def probe(iterations: int, *, output: Path | None = None,
          variants: tuple[str, ...] = VARIANTS) -> dict[str, Any]:
    if iterations < 2:
        raise ValueError("iterations must be at least 2")
    if not variants or len(set(variants)) != len(variants) or any(name not in VARIANTS for name in variants):
        raise ValueError("select distinct supported fragment variants")
    if locate_residual_quad().vertex_range != (TARGET_FIRST, TARGET_END):
        raise RuntimeError("residual quad vertex offset changed")
    base = SimulationConfig()
    config = replace(base, render=replace(base.render, vsync=False, target_fps=0))
    app = SimulatorApp(config, scenario_path=DEFAULT_SCENARIO_PATH)
    buffer = None
    try:
        app.world.shells.clear()
        app.world.stars.count = 0
        load_visual_regression_suite().view("water_reflection").apply(app.camera)
        app.renderer.render(app.world, app.camera, app.celestial, 0.0, None)
        app.ctx.finish()
        _zero_light_counts(app)
        original = app.renderer.scene.program
        original_buffer = app.renderer.scene.buffers[0]
        raw = original_buffer.read(3 * VERTEX_BYTES, TRIGGER_FIRST * VERTEX_BYTES)
        raw += original_buffer.read(6 * VERTEX_BYTES, TARGET_FIRST * VERTEX_BYTES)
        buffer = app.ctx.buffer(raw)
        vertex_source = shaders.source("scene.vert")
        report: dict[str, Any] = {
            "schema_version": 1, "probe": "facade_fragment_reduction_v0_15b",
            "complete": False, "iterations_per_stage": iterations,
            "gpu": _context_metadata(app.ctx),
            "region_gl_half_open": list(REGION_HALF_OPEN),
            "input_sha256": sha256(raw).hexdigest(),
            "vertex_shader_sha256": sha256(vertex_source.encode()).hexdigest(),
            "source_vertex_ranges": [[TRIGGER_FIRST, TRIGGER_FIRST + 3], [TARGET_FIRST, TARGET_END]],
            "limitations": [
                "Same nine-vertex fixture as V0-15A; production initialisation and textures retained.",
                "Full-frame rasterisation; only the eight-pixel target is read back.",
                "Reductions affect all non-facade materials, including the surface-12 trigger.",
                "Facade source unchanged, but compilation may change its executable/output.",
                "Sequential finite samples do not prove necessity or identify the driver as cause.",
            ],
            "programs": {}, "stages": {},
        }
        for variant in ("original_before", *variants, "original_after"):
            reused = variant.startswith("original_")
            source = shaders.source("scene.frag") if reused else fragment_source(variant)
            program = original if reused else app.ctx.program(vertex_shader=vertex_source, fragment_shader=source)
            vao = None
            try:
                if not reused:
                    copy_active_uniforms(original, program)
                vao = app.ctx.vertex_array(program, [(buffer, *VERTEX_LAYOUT)], skip_errors=True)
                report["programs"][variant] = {
                    "fragment_sha256": sha256(source.encode()).hexdigest(),
                    "fragment_source_bytes": len(source.encode()),
                    "reuses_original_program": reused,
                }
                for case, ranges in DRAW_CASES.items():
                    name = f"{variant}/full_frame/{case}"
                    print(f"measuring {name}", flush=True)

                    def draw() -> None:
                        _bind_opaque_textures(app)
                        for first, count in ranges:
                            vao.render(moderngl.TRIANGLES, first=first, vertices=count)

                    result = _measure_draws(app.ctx, iterations, draw, depth=True, scissor_region=False)
                    result["gl_error"] = app.ctx.error
                    if result["gl_error"] != "GL_NO_ERROR":
                        raise RuntimeError(f"{name}: {result['gl_error']}")
                    result["draw_ranges_first_count"] = [list(pair) for pair in ranges]
                    report["stages"][name] = result
                    add_control_comparisons(report)
                    report["control_summary"] = summarize_program_controls(report)
                    _save(report, output)
                    print(f"  varying={result['differing_iterations']} outside_control={result['samples_outside_target_control_states']}", flush=True)
            finally:
                if vao is not None:
                    vao.release()
                if not reused:
                    program.release()
        report["complete"] = True
        _save(report, output)
        return report
    finally:
        if buffer is not None:
            buffer.release()
        app.audio_executor.shutdown(wait=True, cancel_futures=True)
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("iterations must be at least 2")
    probe(args.iterations, output=args.output, variants=tuple(args.variants))


if __name__ == "__main__":
    main()
