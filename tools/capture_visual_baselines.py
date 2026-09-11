"""Capture canonical static Yeouido views as linear HDR and display SDR.

The cameras are project-defined regression instruments, not historical
observer positions. Each view receives a fresh application/context so capture
is isolated from view order, preceding retinal adaptation, reflection cache
state, and the physical camera's noise frame index. A bounded comparison gate
handles rare GPU triangle-edge raster differences without masking broad error.

Example::

    python -m tools.capture_visual_baselines --output-dir visual-baseline
    python -m tools.capture_visual_baselines --output-dir one-view \
        --view terrain_shoreline --frames 8
    python -m tools.capture_visual_baselines --compare baseline candidate
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pygame

from simulator.app import SimulatorApp
from simulator.config import SimulationConfig
from simulator.passes.post import DisplayMode
from simulator.scenario import DEFAULT_SCENARIO_PATH
from simulator.validation.capture import (
    compare_coverage,
    compare_display_sdr,
    compare_linear_hdr,
    coverage_statistics,
    display_sdr_statistics,
    linear_hdr_statistics,
    load_coverage_mask,
    load_known_residuals,
    read_display_sdr,
    read_linear_hdr,
    read_scene_coverage,
    save_coverage_mask,
    save_display_sdr,
    save_linear_hdr,
    select_residuals,
)
from simulator.validation.views import (
    DEFAULT_VISUAL_VIEWS_PATH,
    VisualRegressionSuite,
    VisualRegressionView,
    VisualViewError,
    load_visual_regression_suite,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def select_views(
    suite: VisualRegressionSuite, requested: list[str]
) -> tuple[VisualRegressionView, ...]:
    """Resolve a CLI selection without changing canonical suite order."""

    if not requested:
        return suite.views
    unknown = sorted(set(requested) - {view.view_id for view in suite.views})
    if unknown:
        raise VisualViewError(f"unknown visual-regression view(s): {unknown}")
    selected = set(requested)
    return tuple(view for view in suite.views if view.view_id in selected)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _validate_camera_surface(app: SimulatorApp, view: VisualRegressionView) -> dict:
    x_m, y_m, z_m = view.position_eus_m
    terrain = app.terrain_surface
    water = terrain.is_water(x_m, z_m)
    actual_surface = "water" if water else "land"
    if view.expected_surface != "air" and view.expected_surface != actual_surface:
        raise VisualViewError(
            f"view {view.view_id!r} expected {view.expected_surface}, "
            f"found {actual_surface}"
        )
    surface_height_m = terrain.collision_height_at(x_m, z_m)
    clearance_m = y_m - surface_height_m
    if clearance_m < view.minimum_ground_clearance_m:
        raise VisualViewError(
            f"view {view.view_id!r} clearance {clearance_m:.3f} m is below "
            f"its {view.minimum_ground_clearance_m:.3f} m gate"
        )
    return {
        "surface": actual_surface,
        "surface_height_m": surface_height_m,
        "camera_clearance_m": clearance_m,
    }


def capture_view(
    view: VisualRegressionView,
    scenario_path: Path,
    frames: int,
    display_mode: str,
) -> tuple[object, object, object, dict]:
    """Render one static view and return HDR, SDR, coverage and context."""

    base = SimulationConfig()
    config = replace(
        base,
        render=replace(base.render, vsync=False, target_fps=0),
    )
    app = SimulatorApp(config, scenario_path=scenario_path)
    try:
        # The historical scenario has no firing timeline, and SimulatorApp adds
        # one clearly synthetic development shell to keep interactive use
        # inspectable. Static appearance baselines must contain neither.
        app.world.shells.clear()
        app.world.stars.count = 0
        app.renderer.post.set_mode(DisplayMode(display_mode))
        view.apply(app.camera)
        surface = _validate_camera_surface(app, view)
        frame_dt_s = 1.0 / 60.0
        for _ in range(frames):
            app.renderer.render(
                app.world, app.camera, app.celestial, frame_dt_s, None
            )
        app.ctx.finish()
        hdr = read_linear_hdr(app.renderer)
        sdr = read_display_sdr(app.ctx)
        coverage = read_scene_coverage(app.renderer)
        info = app.ctx.info
        context = {
            "view_id": view.view_id,
            "gl": {
                key: info.get(key)
                for key in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")
            },
            "subject": view.subject,
            "position_eus_m": list(view.position_eus_m),
            "target_eus_m": list(view.target_eus_m),
            "yaw_deg": view.yaw_deg,
            "pitch_deg": view.pitch_deg,
            "display_mode": display_mode,
            "frames": frames,
            "synthetic_firework_present": False,
            **surface,
        }
        return hdr, sdr, coverage, context
    finally:
        app.audio_executor.shutdown(wait=True, cancel_futures=True)
        pygame.quit()


def write_capture(
    output_dir: Path,
    view: VisualRegressionView,
    hdr,
    sdr,
    coverage,
    context: dict,
) -> dict:
    """Persist one lossless set plus a compact machine-readable report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    hdr_path = save_linear_hdr(hdr, output_dir / f"{view.view_id}.hdr.npy")
    sdr_path = save_display_sdr(sdr, output_dir / f"{view.view_id}.sdr.png")
    coverage_path = save_coverage_mask(
        coverage, output_dir / f"{view.view_id}.coverage.png"
    )
    report = {
        **context,
        "notes": view.notes,
        "hdr": {
            "path": hdr_path.name,
            "sha256": _file_sha256(hdr_path),
            "statistics": linear_hdr_statistics(hdr),
        },
        "sdr": {
            "path": sdr_path.name,
            "sha256": _file_sha256(sdr_path),
            "statistics": display_sdr_statistics(sdr),
        },
        "coverage": {
            "path": coverage_path.name,
            "sha256": _file_sha256(coverage_path),
            "statistics": coverage_statistics(coverage),
        },
    }
    report_path = output_dir / f"{view.view_id}.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return {**report, "report_path": report_path.name}


def compare_capture_directories(reference_dir: Path, candidate_dir: Path) -> dict:
    """Compare two capture manifests and every common HDR/SDR frame pair."""

    manifests = []
    for directory in (reference_dir, candidate_dir):
        manifest_path = directory / "manifest.json"
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    reference, candidate = manifests
    identity_fields = (
        "schema_version",
        "suite_id",
        "scenario_id",
        "scene_asset_sha256",
        "views_asset_sha256",
        "display_mode",
        "frames",
    )
    mismatches = {
        field: [reference.get(field), candidate.get(field)]
        for field in identity_fields
        if reference.get(field) != candidate.get(field)
    }
    reference_views = {item["view_id"]: item for item in reference["captures"]}
    candidate_views = {item["view_id"]: item for item in candidate["captures"]}
    if reference_views.keys() != candidate_views.keys():
        mismatches["view_ids"] = [sorted(reference_views), sorted(candidate_views)]
    # Allowances describe one driver build, so they apply only when both
    # captures came from a device the record names. Two captures from different
    # devices are not comparable at this tolerance and are refused outright.
    known_residuals = load_known_residuals()
    reference_gl, candidate_gl = reference.get("gl"), candidate.get("gl")
    gl_info = None
    if reference_gl and candidate_gl:
        if reference_gl != candidate_gl:
            mismatches["gl"] = [reference_gl, candidate_gl]
        else:
            gl_info = reference_gl
    comparisons = []
    for view_id in sorted(reference_views.keys() & candidate_views.keys()):
        first = reference_views[view_id]
        second = candidate_views[view_id]
        hdr_reference = np.load(reference_dir / first["hdr"]["path"])
        hdr_candidate = np.load(candidate_dir / second["hdr"]["path"])
        with Image.open(reference_dir / first["sdr"]["path"]) as image:
            sdr_reference = np.asarray(image.convert("RGB")).copy()
        with Image.open(candidate_dir / second["sdr"]["path"]) as image:
            sdr_candidate = np.asarray(image.convert("RGB")).copy()
        allowances = select_residuals(known_residuals, view_id, gl_info) if gl_info else []
        hdr = compare_linear_hdr(hdr_reference, hdr_candidate, allowances=allowances)
        sdr = compare_display_sdr(sdr_reference, sdr_candidate, allowances=allowances)
        # Coverage predates neither capture unconditionally: a directory
        # written before masks existed still compares on colour alone rather
        # than failing for a file it could not have produced.
        coverage = None
        if "coverage" in first and "coverage" in second:
            coverage = compare_coverage(
                load_coverage_mask(reference_dir / first["coverage"]["path"]),
                load_coverage_mask(candidate_dir / second["coverage"]["path"]),
            )
        comparisons.append(
            {
                "view_id": view_id,
                "passed": bool(
                    hdr["passed"]
                    and sdr["passed"]
                    and (coverage is None or coverage["passed"])
                ),
                "hdr": hdr,
                "sdr": sdr,
                "coverage": coverage,
                "allowances_applied": [entry["id"] for entry in allowances],
            }
        )
    return {
        "schema_version": 1,
        "reference": str(reference_dir),
        "candidate": str(candidate_dir),
        "passed": not mismatches
        and all(comparison["passed"] for comparison in comparisons),
        "metadata_mismatches": mismatches,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        metavar=("REFERENCE_DIR", "CANDIDATE_DIR"),
    )
    parser.add_argument("--views", type=Path, default=DEFAULT_VISUAL_VIEWS_PATH)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--view", action="append", default=[])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument(
        "--display-mode",
        choices=tuple(mode.value for mode in DisplayMode),
        default=None,
    )
    arguments = parser.parse_args()
    if arguments.compare:
        if arguments.output_dir is not None:
            parser.error("--compare cannot be combined with --output-dir")
        try:
            comparison = compare_capture_directories(*arguments.compare)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        print(json.dumps(comparison, indent=2))
        return 0 if comparison["passed"] else 1
    if arguments.output_dir is None:
        parser.error("--output-dir is required unless --compare is used")
    if arguments.frames < 1:
        parser.error("--frames must be at least one")
    try:
        suite = load_visual_regression_suite(arguments.views)
        suite.verify_scene_asset()
        views = select_views(suite, arguments.view)
        display_mode = arguments.display_mode or suite.display_mode
        results = []
        for view in views:
            hdr, sdr, coverage, context = capture_view(
                view, arguments.scenario, arguments.frames, display_mode
            )
            results.append(
                write_capture(
                    arguments.output_dir, view, hdr, sdr, coverage, context
                )
            )
    except (OSError, VisualViewError) as error:
        parser.error(str(error))
    manifest = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "scenario_id": suite.scenario_id,
        "scene_asset": suite.scene_asset,
        "scene_asset_sha256": suite.scene_asset_sha256,
        "views_asset": _portable_path(arguments.views),
        "views_asset_sha256": _file_sha256(arguments.views),
        "display_mode": display_mode,
        "frames": arguments.frames,
        "source": suite.source,
        # Recorded so a comparison can tell whether two captures even came from
        # the same driver, and so a device-scoped allowance cannot be applied
        # to a device it was never measured on.
        "gl": results[0].get("gl") if results else None,
        "captures": results,
    }
    manifest_path = arguments.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
