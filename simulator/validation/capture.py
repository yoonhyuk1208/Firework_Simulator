"""Linear HDR frame capture, for colour and brightness validation.

Colour and brightness comparisons must read the linear RGBA16F buffer before
exposure and tone mapping. A gamma-encoded screenshot has already lost the
quantity being validated, and an SDR display cannot represent firework
luminance in the first place, so a metric computed on the displayed image would
measure the tone mapper.

This module is the only part of the validation package that needs OpenGL. It is
imported lazily by the runner so a headless agent can still produce the rest of
the report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_HDR_REGRESSION_LIMITS = {
    "changed_pixel_fraction": 1.0e-5,
    "maximum_absolute_error": 1.0e-3,
    "mean_absolute_error": 1.0e-7,
}
DEFAULT_SDR_REGRESSION_LIMITS = {
    "changed_pixel_fraction": 1.0e-5,
    "mean_absolute_error_rgb8": 5.0e-4,
}
#: Coverage is a boolean mask, so a differing pixel is a triangle edge landing
#: on the other side of a sample point rather than a shading difference. The
#: same bound as the colour targets applies for the same reason.
DEFAULT_COVERAGE_REGRESSION_LIMITS = {
    "changed_pixel_fraction": 1.0e-5,
}


def read_linear_hdr(renderer: Any) -> np.ndarray:
    """Read a renderer's HDR colour target as linear float32 RGBA.

    Returns an ``(height, width, 4)`` array in image order — row 0 is the top of
    the frame — with the OpenGL bottom-left origin already corrected. Values are
    scene-referred linear radiance, not display-referred.
    """

    texture = getattr(renderer, "hdr_texture", None)
    if texture is None:
        raise AttributeError(
            "renderer has no hdr_texture; linear capture requires the HDR "
            "colour target created by Renderer.__init__"
        )
    width, height = texture.size
    components = texture.components
    if texture.dtype != "f2":
        raise ValueError(
            f"expected a half-float HDR target, found dtype {texture.dtype!r}"
        )
    raw = np.frombuffer(texture.read(), dtype=np.float16)
    frame = raw.reshape(height, width, components).astype(np.float32)
    # OpenGL texture row 0 is the bottom of the image.
    return np.flipud(frame).copy()


def read_scene_coverage(renderer: Any) -> np.ndarray:
    """Read the scene depth target as a boolean geometry-versus-sky mask.

    The depth attachment clears to the far plane before the scene pass, so any
    fragment nearer than that is geometry. Taking the silhouette from depth
    rather than from the display frame matters: segmenting sky out of a graded
    image would let an exposure change masquerade as a change in shape, which
    is exactly what the silhouette measure exists to rule out.
    """

    texture = getattr(renderer, "scene_depth_texture", None)
    if texture is None:
        raise AttributeError(
            "renderer has no scene_depth_texture; coverage capture requires "
            "the depth attachment created by Renderer.__init__"
        )
    width, height = texture.size
    raw = np.frombuffer(texture.read(), dtype=np.float32)
    expected = width * height
    if raw.size != expected:
        raise ValueError(
            f"expected {expected} depth samples, received {raw.size}"
        )
    depth = np.flipud(raw.reshape(height, width))
    from .frame_comparison import coverage_mask

    return coverage_mask(depth)


def coverage_statistics(mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("coverage mask must have shape (height, width)")
    return {
        "width": int(values.shape[1]),
        "height": int(values.shape[0]),
        "coverage_fraction": float(values.mean()),
    }


def save_coverage_mask(mask: np.ndarray, path: Path) -> Path:
    """Persist a coverage mask losslessly as a 1-bit PNG."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("coverage mask must have shape (height, width)")
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values).save(path, bits=1, optimize=True)
    return path


def load_coverage_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("1"), dtype=bool)


def linear_hdr_statistics(frame: np.ndarray) -> dict[str, float]:
    """Summarise a linear HDR frame without applying any display transform."""

    colour = frame[:, :, :3]
    finite = np.isfinite(colour)
    return {
        "width": float(frame.shape[1]),
        "height": float(frame.shape[0]),
        "minimum": float(colour[finite].min()) if finite.any() else float("nan"),
        "maximum": float(colour[finite].max()) if finite.any() else float("nan"),
        "mean": float(colour[finite].mean()) if finite.any() else float("nan"),
        "percentile_99": (
            float(np.percentile(colour[finite], 99)) if finite.any() else float("nan")
        ),
        "non_finite_fraction": float(1.0 - finite.mean()),
        "negative_fraction": float((colour < 0.0).mean()),
    }


def save_linear_hdr(frame: np.ndarray, path: Path) -> Path:
    """Persist a linear HDR frame as ``.npy`` for later comparison.

    Deliberately not an image format: PNG or JPEG would quantise and gamma-encode
    the very values the comparison depends on.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    np.save(path, frame.astype(np.float32))
    return path


def read_display_sdr(context: Any) -> np.ndarray:
    """Read the final display-referred RGB8 framebuffer in image order."""

    screen = getattr(context, "screen", None)
    if screen is None:
        raise AttributeError("OpenGL context has no screen framebuffer")
    width, height = screen.size
    raw = np.frombuffer(
        screen.read(components=3, alignment=1), dtype=np.uint8
    )
    expected = width * height * 3
    if raw.size != expected:
        raise ValueError(
            f"expected {expected} display bytes, received {raw.size}"
        )
    return np.flipud(raw.reshape(height, width, 3)).copy()


def display_sdr_statistics(frame: np.ndarray) -> dict[str, object]:
    """Summarise display RGB without presenting code values as radiance."""

    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("display frame must have shape (height, width, 3)")
    luminance = rgb.astype(np.float64) @ np.array(
        [0.2126, 0.7152, 0.0722], dtype=np.float64
    )
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "median_rgb8": np.median(rgb.reshape(-1, 3), axis=0).tolist(),
        "luminance_p05_p50_p95": np.percentile(
            luminance, [5, 50, 95]
        ).tolist(),
        "clipped_black_fraction": float(np.all(rgb == 0, axis=2).mean()),
        "clipped_white_fraction": float(np.all(rgb == 255, axis=2).mean()),
    }


def save_display_sdr(frame: np.ndarray, path: Path) -> Path:
    """Save a display-referred RGB8 frame losslessly as PNG."""

    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("display frame must have shape (height, width, 3)")
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


KNOWN_RESIDUALS_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "known_render_residuals.json"
)


def load_known_residuals(path: Path | None = None) -> dict:
    """The committed record of driver differences that are not regressions."""

    import json

    source = Path(path) if path is not None else KNOWN_RESIDUALS_PATH
    if not source.is_file():
        return {"residuals": [], "devices": []}
    return json.loads(source.read_text(encoding="utf-8"))


def device_matches(info: dict, devices: list[dict]) -> bool:
    """Whether a GL context is one the allowances were recorded against.

    An allowance describes what one driver build does. Applying it to an
    unlisted device would hide exactly the regression the gate is for, so the
    match is exact on the vendor and a substring on the strings that carry the
    model and build.
    """

    for device in devices:
        vendor = device.get("GL_VENDOR")
        if vendor is not None and info.get("GL_VENDOR") != vendor:
            continue
        renderer = device.get("GL_RENDERER_contains")
        if renderer is not None and renderer not in str(info.get("GL_RENDERER", "")):
            continue
        version = device.get("GL_VERSION_contains")
        if version is not None and version not in str(info.get("GL_VERSION", "")):
            continue
        return True
    return False


def select_residuals(record: dict, view_id: str, info: dict) -> list[dict]:
    """Allowances that apply to this view on this device, or nothing."""

    if not device_matches(info, record.get("devices", [])):
        return []
    return [r for r in record.get("residuals", []) if r.get("view_id") == view_id]


def _residual_region(shape: tuple[int, int], residual: dict) -> np.ndarray:
    region = np.zeros(shape, dtype=bool)
    row0, row1 = residual["region_rows"]
    col0, col1 = residual["region_cols"]
    region[row0 : row1 + 1, col0 : col1 + 1] = True
    return region


def _apply_allowances(
    changed: np.ndarray,
    error: np.ndarray,
    residuals: list[dict],
    error_key: str,
) -> tuple[np.ndarray, list[dict]]:
    """Split changed pixels into gated and allowed, and audit each allowance.

    The allowed pixels leave the metrics, but each allowance is then checked
    against its own recorded bounds, so an exemption can only ever be narrower
    than relaxing a limit: a residual that grows past what was measured fails
    here even though its pixels were removed from the global count.
    """

    allowed = np.zeros_like(changed)
    audit: list[dict] = []
    for residual in residuals:
        region = _residual_region(changed.shape, residual)
        hit = changed & region
        count = int(hit.sum())
        magnitude = float(error[hit].max()) if count else 0.0
        bounds = residual["bounds"]
        limit = bounds.get(error_key)
        within = count <= bounds["max_pixels"] and (
            limit is None or magnitude <= limit
        )
        audit.append(
            {
                "id": residual["id"],
                "pixels": count,
                "max_error": magnitude,
                "max_pixels_allowed": bounds["max_pixels"],
                "max_error_allowed": limit,
                "within_bounds": within,
            }
        )
        allowed |= hit
    return allowed, audit


def compare_coverage(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    limits: dict[str, float] | None = None,
) -> dict[str, object]:
    """Compare two geometry coverage masks.

    Recorded alongside the colour targets because it fails differently. A mask
    that stopped being written, or came back empty, leaves the colour
    comparison untouched and would otherwise pass unnoticed; conversely a
    silhouette that moved is visible here at a pixel count rather than buried
    in a shading difference.
    """

    first = np.asarray(reference, dtype=bool)
    second = np.asarray(candidate, dtype=bool)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("coverage masks must have the same (height, width) shape")
    thresholds = dict(DEFAULT_COVERAGE_REGRESSION_LIMITS)
    if limits:
        thresholds.update(limits)
    changed = first != second
    metrics = {
        "changed_pixel_count": int(changed.sum()),
        "changed_pixel_fraction": float(changed.mean()),
        "reference_coverage_fraction": float(first.mean()),
        "candidate_coverage_fraction": float(second.mean()),
    }
    passed = all(
        metrics[key] <= value for key, value in thresholds.items()
    )
    return {"passed": bool(passed), "limits": thresholds, "metrics": metrics}


def compare_linear_hdr(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    limits: dict[str, float] | None = None,
    allowances: list[dict] | None = None,
) -> dict[str, object]:
    """Compare scene-linear frames with a bounded raster-edge tolerance."""

    first = np.asarray(reference, dtype=np.float32)
    second = np.asarray(candidate, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 3 or first.shape[2] < 3:
        raise ValueError(
            "HDR frames must have the same (height, width, channels) shape"
        )
    thresholds = dict(DEFAULT_HDR_REGRESSION_LIMITS)
    if limits:
        thresholds.update(limits)
    difference = np.abs(first[:, :, :3] - second[:, :, :3])
    finite = np.isfinite(difference)
    changed_pixels = np.any(difference > 0.0, axis=2)
    allowed, audit = (
        _apply_allowances(
            changed_pixels, difference.max(axis=2), allowances, "hdr_max_abs_error"
        )
        if allowances
        else (np.zeros_like(changed_pixels), [])
    )
    changed_pixels = changed_pixels & ~allowed
    difference = np.where(allowed[:, :, None], 0.0, difference)
    finite = np.isfinite(difference)
    metrics = {
        "changed_pixel_count": int(changed_pixels.sum()),
        "changed_pixel_fraction": float(changed_pixels.mean()),
        "maximum_absolute_error": (
            float(difference[finite].max()) if finite.any() else float("inf")
        ),
        "mean_absolute_error": (
            float(difference[finite].mean()) if finite.any() else float("inf")
        ),
        "non_finite_error_count": int((~finite).sum()),
    }
    passed = (
        metrics["non_finite_error_count"] == 0
        and all(metrics[key] <= value for key, value in thresholds.items())
        and all(entry["within_bounds"] for entry in audit)
    )
    return {
        "passed": bool(passed),
        "limits": thresholds,
        "metrics": metrics,
        "allowed_residuals": audit,
    }


def compare_display_sdr(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    limits: dict[str, float] | None = None,
    allowances: list[dict] | None = None,
) -> dict[str, object]:
    """Compare final RGB8 frames without interpreting code values as light."""

    first = np.asarray(reference)
    second = np.asarray(candidate)
    if first.shape != second.shape or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("SDR frames must have the same (height, width, 3) shape")
    thresholds = dict(DEFAULT_SDR_REGRESSION_LIMITS)
    if limits:
        thresholds.update(limits)
    difference = np.abs(first.astype(np.int16) - second.astype(np.int16))
    changed_pixels = np.any(difference > 0, axis=2)
    allowed, audit = (
        _apply_allowances(
            changed_pixels,
            difference.max(axis=2),
            allowances,
            "sdr_max_abs_error_rgb8",
        )
        if allowances
        else (np.zeros_like(changed_pixels), [])
    )
    changed_pixels = changed_pixels & ~allowed
    difference = np.where(allowed[:, :, None], 0, difference)
    metrics = {
        "changed_pixel_count": int(changed_pixels.sum()),
        "changed_pixel_fraction": float(changed_pixels.mean()),
        "maximum_absolute_error_rgb8": int(difference.max()),
        "mean_absolute_error_rgb8": float(difference.mean()),
    }
    passed = all(metrics[key] <= value for key, value in thresholds.items()) and all(
        entry["within_bounds"] for entry in audit
    )
    return {
        "passed": bool(passed),
        "limits": thresholds,
        "metrics": metrics,
        "allowed_residuals": audit,
    }
