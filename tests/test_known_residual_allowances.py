"""An allowance must be narrower than relaxing a limit, in every direction.

The capture gate was failing five comparisons in six on this machine because of
two reproduced driver residuals, which makes it useless for catching real
regressions. Exempting them is only defensible if the exemption cannot grow:
these tests fail if a difference escapes the declared region, exceeds the
recorded pixel count or magnitude, or is applied to a device the record never
measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simulator.validation.capture import (
    DEFAULT_HDR_REGRESSION_LIMITS,
    DEFAULT_SDR_REGRESSION_LIMITS,
    compare_display_sdr,
    compare_linear_hdr,
    device_matches,
    load_known_residuals,
    select_residuals,
)

RECORD = load_known_residuals()
DEVICE = {
    "GL_VENDOR": "Intel",
    "GL_RENDERER": "Intel(R) Arc(TM) 140V GPU (16GB)",
    "GL_VERSION": "4.3.0 - Build 32.0.101.8626",
}
FACADE = next(r for r in RECORD["residuals"] if r["id"] == "facade_window_grid_quad")


def _hdr(shape=(720, 1280)) -> np.ndarray:
    return np.zeros((*shape, 3), dtype=np.float32)


def _sdr(shape=(720, 1280)) -> np.ndarray:
    return np.zeros((*shape, 3), dtype=np.uint8)


def test_the_global_limits_were_not_relaxed() -> None:
    # The whole justification for an allowance is that it does not widen these.
    assert DEFAULT_HDR_REGRESSION_LIMITS == {
        "changed_pixel_fraction": 1.0e-5,
        "maximum_absolute_error": 1.0e-3,
        "mean_absolute_error": 1.0e-7,
    }
    assert DEFAULT_SDR_REGRESSION_LIMITS == {
        "changed_pixel_fraction": 1.0e-5,
        "mean_absolute_error_rgb8": 5.0e-4,
    }
    assert RECORD["policy"]["global_limits_unchanged"] is True


def test_an_allowance_only_applies_to_the_device_it_was_measured_on() -> None:
    assert device_matches(DEVICE, RECORD["devices"])
    assert select_residuals(RECORD, "water_reflection", DEVICE)

    for altered in (
        {**DEVICE, "GL_VENDOR": "NVIDIA Corporation"},
        {**DEVICE, "GL_RENDERER": "Intel(R) Arc(TM) 130V GPU"},
        {**DEVICE, "GL_VERSION": "4.3.0 - Build 32.0.101.9999"},
    ):
        assert not device_matches(altered, RECORD["devices"]), altered
        assert select_residuals(RECORD, "water_reflection", altered) == []


def test_an_allowance_only_applies_to_the_view_it_was_measured_on() -> None:
    assert select_residuals(RECORD, "terrain_shoreline", DEVICE) == []


def test_the_recorded_residual_passes_and_is_reported() -> None:
    reference, candidate = _hdr(), _hdr()
    row, col = FACADE["region_rows"][0] + 1, FACADE["region_cols"][0] + 1
    candidate[row : row + 4, col : col + 2] = FACADE["observed"]["hdr_max_abs_error"]

    result = compare_linear_hdr(reference, candidate, allowances=[FACADE])

    assert result["passed"]
    entry = next(e for e in result["allowed_residuals"] if e["id"] == FACADE["id"])
    assert entry["pixels"] == 8 and entry["within_bounds"]
    # Removing the allowed pixels must leave nothing behind.
    assert result["metrics"]["changed_pixel_count"] == 0


def test_a_difference_outside_the_region_still_fails() -> None:
    reference, candidate = _hdr(), _hdr()
    candidate[10:14, 10:12] = FACADE["observed"]["hdr_max_abs_error"]

    result = compare_linear_hdr(reference, candidate, allowances=[FACADE])

    assert not result["passed"]
    assert result["metrics"]["changed_pixel_count"] == 8
    assert all(e["pixels"] == 0 for e in result["allowed_residuals"])


def test_too_many_pixels_inside_the_region_fails() -> None:
    reference, candidate = _hdr(), _hdr()
    row0, row1 = FACADE["region_rows"]
    col0, col1 = FACADE["region_cols"]
    candidate[row0 : row1 + 1, col0 : col1 + 1] = 1.0e-4

    result = compare_linear_hdr(reference, candidate, allowances=[FACADE])

    entry = result["allowed_residuals"][0]
    assert entry["pixels"] > FACADE["bounds"]["max_pixels"]
    assert not entry["within_bounds"]
    assert not result["passed"]


def test_too_large_a_magnitude_inside_the_region_fails() -> None:
    reference, candidate = _hdr(), _hdr()
    row, col = FACADE["region_rows"][0] + 1, FACADE["region_cols"][0] + 1
    candidate[row : row + 4, col : col + 2] = (
        FACADE["bounds"]["hdr_max_abs_error"] * 2.0
    )

    result = compare_linear_hdr(reference, candidate, allowances=[FACADE])

    entry = result["allowed_residuals"][0]
    assert entry["pixels"] == 8
    assert not entry["within_bounds"]
    assert not result["passed"]


def test_the_sdr_path_is_bounded_the_same_way() -> None:
    row, col = FACADE["region_rows"][0] + 1, FACADE["region_cols"][0] + 1

    within, beyond = _sdr(), _sdr()
    within[row : row + 4, col : col + 2] = FACADE["observed"]["sdr_max_abs_error_rgb8"]
    beyond[row : row + 4, col : col + 2] = (
        FACADE["bounds"]["sdr_max_abs_error_rgb8"] + 20
    )

    assert compare_display_sdr(_sdr(), within, allowances=[FACADE])["passed"]
    assert not compare_display_sdr(_sdr(), beyond, allowances=[FACADE])["passed"]


def test_without_allowances_the_comparison_is_unchanged() -> None:
    reference, candidate = _hdr(), _hdr()
    row, col = FACADE["region_rows"][0] + 1, FACADE["region_cols"][0] + 1
    candidate[row : row + 4, col : col + 2] = FACADE["observed"]["hdr_max_abs_error"]

    result = compare_linear_hdr(reference, candidate)

    assert result["allowed_residuals"] == []
    assert result["metrics"]["changed_pixel_count"] == 8
    assert not result["passed"]


def test_every_recorded_residual_cites_its_evidence() -> None:
    for residual in RECORD["residuals"]:
        assert residual["evidence"], residual["id"]
        for reference in residual["evidence"]:
            assert Path(reference).is_file(), reference
        assert residual["attribution"].strip()
        # Bounds must not be tighter than what was actually observed, or the
        # gate would fail on the very thing it documents.
        observed = residual["observed"]
        bounds = residual["bounds"]
        assert observed["hdr_pixels"] <= bounds["max_pixels"]
        assert observed["hdr_max_abs_error"] <= bounds["hdr_max_abs_error"]
        assert observed["sdr_max_abs_error_rgb8"] <= bounds["sdr_max_abs_error_rgb8"]
