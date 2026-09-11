"""Measure how far each view is from a supersampled reference.

Shimmer under a moving camera is aliasing seen over time: detail finer than the
sample spacing is decided by where the pixel centre happens to fall, so it
changes discontinuously as the camera moves. The cause is visible without any
motion at all — render the same frame at a higher sample rate, box-filter it
back down, and the difference is the error the 1x render is making.

The comparison reads the linear HDR target rather than the display frame. Bloom
and the display transform work in pixel units, so a supersampled render would
apply them at a different world scale and the difference would measure that
instead of aliasing.

Known driver residuals are excluded by region, reusing the record the capture
gate uses, so a reproduced driver difference is not counted as aliasing error.

Example::

    python -m tools.measure_aliasing --view water_reflection --supersample 4
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pygame

from simulator.app import SimulatorApp
from simulator.config import SimulationConfig
from simulator.passes.post import DisplayMode
from simulator.scenario import DEFAULT_SCENARIO_PATH
from simulator.validation.capture import (
    load_known_residuals,
    read_linear_hdr,
    select_residuals,
)
from simulator.validation.views import (
    DEFAULT_VISUAL_VIEWS_PATH,
    load_visual_regression_suite,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("docs/validation/aliasing_v2/aliasing_report.json")
#: Blocks are reported so the worst offenders can be located on screen rather
#: than summarised into one number that names nothing.
BLOCK = 16
#: Relative error above which a pixel counts as visibly wrong. Well above the
#: float16 storage step at these magnitudes, so quantisation is not counted.
RELATIVE_THRESHOLD = 0.05


def render_view(view, frames: int, display_mode: str, width: int, height: int):
    """One view's linear HDR at a given resolution, plus the device it ran on."""

    base = SimulationConfig()
    config = replace(
        base,
        render=replace(
            base.render, vsync=False, target_fps=0, width=width, height=height
        ),
    )
    app = SimulatorApp(config, scenario_path=DEFAULT_SCENARIO_PATH)
    try:
        app.world.shells.clear()
        app.world.stars.count = 0
        app.renderer.post.set_mode(DisplayMode(display_mode))
        view.apply(app.camera)
        for _ in range(frames):
            app.renderer.render(app.world, app.camera, app.celestial, 1.0 / 60.0, None)
        app.ctx.finish()
        frame = read_linear_hdr(app.renderer)[:, :, :3].astype(np.float64)
        info = dict(app.ctx.info)
        gl = {key: info.get(key) for key in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")}
        return frame, gl
    finally:
        app.audio_executor.shutdown(wait=True, cancel_futures=True)
        pygame.quit()


def box_downsample(frame: np.ndarray, factor: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height % factor or width % factor:
        raise ValueError("supersampled frame must divide evenly by the factor")
    return frame.reshape(
        height // factor, factor, width // factor, factor, frame.shape[2]
    ).mean(axis=(1, 3))


def excluded_mask(shape: tuple[int, int], residuals: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for residual in residuals:
        row0, row1 = residual["region_rows"]
        col0, col1 = residual["region_cols"]
        mask[row0 : row1 + 1, col0 : col1 + 1] = True
    return mask


def worst_blocks(error: np.ndarray, count: int) -> list[dict]:
    """The worst square blocks, so the offenders can be found on screen."""

    height, width = error.shape
    rows, cols = height // BLOCK, width // BLOCK
    trimmed = error[: rows * BLOCK, : cols * BLOCK]
    blocks = trimmed.reshape(rows, BLOCK, cols, BLOCK).mean(axis=(1, 3))
    order = np.argsort(blocks, axis=None)[::-1][:count]
    located = []
    for flat in order:
        row, col = divmod(int(flat), cols)
        located.append(
            {
                "row": row * BLOCK,
                "col": col * BLOCK,
                "size": BLOCK,
                "mean_abs_error": float(blocks[row, col]),
            }
        )
    return located


def summarise(
    rendered: np.ndarray,
    reference: np.ndarray,
    residuals: list[dict],
) -> dict:
    """Statistics over the pixels that are not a known driver residual."""

    error = np.abs(rendered - reference).max(axis=2)
    keep = ~excluded_mask(error.shape, residuals)
    kept = error[keep]
    magnitude = np.maximum(reference.max(axis=2), 1e-6)[keep]
    relative = kept / magnitude
    return {
        "excluded_residuals": [residual["id"] for residual in residuals],
        "excluded_pixels": int((~keep).sum()),
        "mean_abs_error": float(kept.mean()),
        "p95_abs_error": float(np.percentile(kept, 95)),
        "max_abs_error": float(kept.max()),
        "relative_threshold": RELATIVE_THRESHOLD,
        "visibly_wrong_fraction": float((relative > RELATIVE_THRESHOLD).mean()),
        "worst_blocks": worst_blocks(np.where(keep, error, 0.0), 5),
    }


def measure_view(view, frames, display_mode, width, height, factor, record):
    """Render once at each rate; exclusion is a mask applied afterwards."""

    supersampled, gl = render_view(
        view, frames, display_mode, width * factor, height * factor
    )
    reference = box_downsample(supersampled, factor)
    rendered, _ = render_view(view, frames, display_mode, width, height)
    if rendered.shape != reference.shape:
        raise RuntimeError(f"shape mismatch {rendered.shape} vs {reference.shape}")
    residuals = select_residuals(record, view.view_id, gl)
    return {
        "view_id": view.view_id,
        "gl": gl,
        "supersample": factor,
        **summarise(rendered, reference, residuals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="append", default=[])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--supersample", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.supersample < 2:
        parser.error("supersample must be at least 2")

    suite = load_visual_regression_suite(DEFAULT_VISUAL_VIEWS_PATH)
    views = [
        view
        for view in suite.views
        if not arguments.view or view.view_id in arguments.view
    ]
    record = load_known_residuals()
    base = SimulationConfig().render

    results = []
    for view in views:
        print(f"measuring {view.view_id}", flush=True)
        results.append(
            measure_view(
                view,
                arguments.frames,
                suite.display_mode,
                base.width,
                base.height,
                arguments.supersample,
                record,
            )
        )

    report = {
        "schema_version": 1,
        "probe": "aliasing_vs_supersampled_reference",
        "frames": arguments.frames,
        "supersample": arguments.supersample,
        "block": BLOCK,
        "views": results,
        "limitations": [
            "A box-filtered supersample is a better reference than the 1x render, not ground truth.",
            "Ambient occlusion and the reflection pre-pass use resolution-scaled buffers, so their sampling improves in the reference too.",
        ],
    }
    output = arguments.output
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}")
    for result in results:
        print(
            f"  {result['view_id']:20} mean={result['mean_abs_error']:.4e} "
            f"p95={result['p95_abs_error']:.4e} visibly_wrong="
            f"{result['visibly_wrong_fraction'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
