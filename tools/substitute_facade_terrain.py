"""Rebuild a facade snapshot with the terrain height field mostly synthesised.

The bundle from V0-15C carries a full 1024x1024 terrain height field derived
from official Korean survey data, which is what stops it being redistributed.
Almost none of it is used: the nine vertices sample a handful of texels, and
the vertex shader reads each one bilinearly plus a one-texel cross for its
normal. Everything else can be replaced without moving a single vertex.

This computes the touched set from the bundle itself — the vertex positions and
the `terrain_bounds` uniform — rather than hard-coding coordinates, keeps those
texels with a halo, and fills the rest with a chosen synthetic pattern. The
retained texels are reported by count and by byte size so the amount of
original data left is an explicit number rather than an impression.

Substituting outside the halo is position-preserving by construction; a fill
that also replaced the halo would move the geometry away from the measurement
region and could read as a false negative.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np

from tools.facade_snapshot import read_bundle, write_bundle

#: Bilinear sampling reads a 2x2 block and `textureOffset` a one-texel cross,
#: so two texels of margin covers both with room to spare.
HALO = 2
FILLS = ("zeros", "constant", "ramp")


def touched_texels(manifest: dict, blobs: dict[str, bytes]) -> tuple[set, tuple[int, int]]:
    """Texel indices the nine vertices can reach, and the field's size."""

    terrain = next(t for t in manifest["textures"] if t["name"] == "terrain")
    width, height = terrain["size"]
    uniforms = manifest["programs"]["original"]["uniforms"]
    bounds = struct.unpack("<4f", bytes.fromhex(uniforms["terrain_bounds"]))
    vertices = np.frombuffer(blobs["vertices.bin"], dtype=np.float32).reshape(-1, 10)

    dimension = np.array([float(width), float(height)])
    grid = np.maximum(dimension - 1.0, 1.0)
    lower, upper = np.array(bounds[:2]), np.array(bounds[2:])
    touched: set[tuple[int, int]] = set()
    for row in vertices:
        xz = row[[0, 2]].astype(np.float64)
        geographic = (xz - lower) / (upper - lower)
        uv = (geographic * grid + 0.5) / dimension
        pixel = uv * dimension - 0.5
        x0, y0 = int(np.floor(pixel[0])), int(np.floor(pixel[1]))
        for dx in range(-HALO, HALO + 1):
            for dy in range(-HALO, HALO + 1):
                touched.add(
                    (int(np.clip(x0 + dx, 0, width - 1)), int(np.clip(y0 + dy, 0, height - 1)))
                )
    return touched, (width, height)


def synthesise(field: np.ndarray, touched: set, fill: str) -> np.ndarray:
    height, width = field.shape
    if fill == "zeros":
        replacement = np.zeros_like(field)
    elif fill == "constant":
        replacement = np.full_like(field, float(field.mean()))
    elif fill == "ramp":
        # A smooth deterministic field, so a viewer of the bundle sees plausible
        # terrain rather than a flat plate, while carrying no surveyed content.
        ys, xs = np.mgrid[0:height, 0:width]
        replacement = (
            12.0
            + 6.0 * np.sin(xs / width * 4.0 * np.pi)
            + 4.0 * np.cos(ys / height * 3.0 * np.pi)
        ).astype(field.dtype)
    else:
        raise ValueError(f"unknown fill: {fill}")
    out = replacement.copy()
    for x, y in touched:
        out[y, x] = field[y, x]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fill", choices=FILLS, default="ramp")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"output already exists: {arguments.output}")

    manifest, blobs = read_bundle(arguments.snapshot)
    touched, (width, height) = touched_texels(manifest, blobs)
    terrain = next(t for t in manifest["textures"] if t["name"] == "terrain")
    level = terrain["levels"][0]
    field = np.frombuffer(blobs[level], dtype=np.float32).reshape(height, width)
    substituted = synthesise(field, touched, arguments.fill)

    # The whole point is that no vertex moves, so assert it rather than trust it.
    for x, y in touched:
        assert substituted[y, x] == field[y, x], (x, y)
    changed = int(np.count_nonzero(substituted != field))

    blobs[level] = substituted.astype(np.float32).tobytes()
    manifest = json.loads(json.dumps(manifest))
    manifest["files"][level]["sha256"] = sha256(blobs[level]).hexdigest()
    manifest["files"][level]["bytes"] = len(blobs[level])
    manifest["terrain_substitution"] = {
        "fill": arguments.fill,
        "halo": HALO,
        "retained_texels": len(touched),
        "retained_bytes": len(touched) * 4,
        "total_texels": width * height,
        "retained_fraction": len(touched) / (width * height),
        "texels_changed": changed,
        "source_terrain_sha256": sha256(field.tobytes()).hexdigest(),
    }
    write_bundle(arguments.output, manifest, blobs)

    report = {
        "schema_version": 1,
        "probe": "facade_terrain_substitution_v0_15d",
        "source_snapshot_sha256": sha256(arguments.snapshot.read_bytes()).hexdigest(),
        "output_snapshot_sha256": sha256(arguments.output.read_bytes()).hexdigest(),
        "output_bytes": arguments.output.stat().st_size,
        **manifest["terrain_substitution"],
        "limitations": [
            "Position preservation is exact only outside the retained halo; it is asserted, not assumed.",
            "A positive replay shows the substitution keeps the phenomenon, not that terrain content is irrelevant in general.",
        ],
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {arguments.output}: retained {len(touched)} texels "
        f"({len(touched) * 4} bytes, {len(touched) / (width * height) * 100:.4f}%), "
        f"changed {changed:,}"
    )


if __name__ == "__main__":
    main()
