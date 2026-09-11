"""V0-15D keeps the reproduction while discarding the survey height field.

The bundle only stays unshareable because of the terrain texture, and almost
none of it is read. These tests pin the two properties that make the
substitution safe to rely on: the retained set is derived from the bundle
rather than hard-coded, and every texel a vertex can reach survives untouched.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tools.substitute_facade_terrain import HALO, synthesise, touched_texels


def _bundle(positions, bounds=(-2500.0, -2000.0, 2500.0, 2000.0), size=(64, 64)):
    vertices = np.zeros((len(positions), 10), dtype=np.float32)
    for index, (x, z) in enumerate(positions):
        vertices[index, 0], vertices[index, 2] = x, z
    manifest = {
        "textures": [{"name": "terrain", "size": list(size), "levels": ["textures/terrain/0.bin"]}],
        "programs": {"original": {"uniforms": {"terrain_bounds": struct.pack("<4f", *bounds).hex()}}},
    }
    return manifest, {"vertices.bin": vertices.tobytes()}


def test_the_retained_set_comes_from_the_bundle_not_a_constant() -> None:
    # Moving a vertex must move the retained texels; a hard-coded set would
    # silently keep the wrong ones and drop the sampled height.
    near, _ = touched_texels(*_bundle([(0.0, 0.0)]))
    far, _ = touched_texels(*_bundle([(1000.0, -800.0)]))

    assert near and far
    assert not (near & far)
    # One vertex reaches a (2*HALO+1)^2 block unless it clamps at an edge.
    assert len(near) == (2 * HALO + 1) ** 2


def test_bilinear_and_offset_neighbours_are_inside_the_halo() -> None:
    # The shader reads a 2x2 bilinear block and a one-texel cross around it, so
    # the halo has to cover at least one texel beyond the sampled corner.
    assert HALO >= 2


def test_every_retained_texel_survives_each_fill() -> None:
    manifest, blobs = _bundle([(120.0, -340.0), (-900.0, 450.0)])
    touched, (width, height) = touched_texels(manifest, blobs)
    rng = np.random.default_rng(11)
    field = rng.uniform(0.0, 70.0, size=(height, width)).astype(np.float32)

    for fill in ("zeros", "constant", "ramp"):
        out = synthesise(field, touched, fill)
        assert out.shape == field.shape and out.dtype == field.dtype
        for x, y in touched:
            assert out[y, x] == field[y, x], (fill, x, y)
        changed = np.count_nonzero(out != field)
        # Everything outside the halo should move, or the substitution is not
        # actually discarding the survey content.
        assert changed >= field.size - len(touched) - 8


def test_the_fills_are_deterministic_and_distinct() -> None:
    manifest, blobs = _bundle([(0.0, 0.0)])
    touched, (width, height) = touched_texels(manifest, blobs)
    field = np.full((height, width), 5.0, dtype=np.float32)

    ramp_once = synthesise(field, touched, "ramp")
    ramp_twice = synthesise(field, touched, "ramp")
    assert np.array_equal(ramp_once, ramp_twice)
    assert not np.array_equal(ramp_once, synthesise(field, touched, "zeros"))
    assert np.count_nonzero(synthesise(field, touched, "zeros")) == len(touched)


def test_an_unknown_fill_is_refused() -> None:
    manifest, blobs = _bundle([(0.0, 0.0)])
    touched, (width, height) = touched_texels(manifest, blobs)
    with pytest.raises(ValueError, match="unknown fill"):
        synthesise(np.zeros((height, width), dtype=np.float32), touched, "whatever")
