"""Portable snapshot data/measurement helpers; no simulator imports.

Raw array mip transfer currently supports Windows desktop OpenGL. ModernGL's
TextureArray.read/write only expose level zero. All transferred levels are
read back and checked byte-for-byte, including the RGB 1x1 tail levels.
"""
from __future__ import annotations

from collections import Counter
import ctypes as ct
from hashlib import sha256
import json
import os
from pathlib import Path
import zipfile

import moderngl
import numpy as np

FRAME_SIZE = (1280, 720)
REGION = (598, 380, 2, 4)
LAYOUT = ("3f 3f 1f 2f 1f", "in_position", "in_normal", "in_surface", "in_surface_uv", "in_facade_style")
CASES = {"target_only": ((3, 6),), "trigger_then_target": ((0, 3), (3, 6)),
         "target_then_trigger": ((3, 6), (0, 3)), "target_only_after": ((3, 6),)}


class ArrayMips:
    """Transfer RGB8 array mips with explicit pack/unpack alignment."""
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("exact array mip transfer currently requires Windows OpenGL")
        lib = ct.WinDLL("opengl32")
        self._lib = lib
        self.get_int = lib.glGetIntegerv
        self.get_int.argtypes = [ct.c_uint, ct.POINTER(ct.c_int)]
        self.pixel_store = lib.glPixelStorei
        self.pixel_store.argtypes = [ct.c_uint, ct.c_int]
        self.get_image = lib.glGetTexImage
        self.get_image.argtypes = [ct.c_uint, ct.c_int, ct.c_uint, ct.c_uint, ct.c_void_p]
        self.get_level = lib.glGetTexLevelParameteriv
        self.get_level.argtypes = [ct.c_uint, ct.c_int, ct.c_uint, ct.POINTER(ct.c_int)]
        lib.wglGetProcAddress.argtypes = [ct.c_char_p]
        lib.wglGetProcAddress.restype = ct.c_void_p
        address = lib.wglGetProcAddress(b"glTexSubImage3D")
        if address in (None, 1, 2, 3, ct.c_void_p(-1).value):
            raise RuntimeError("glTexSubImage3D unavailable in the current context")
        self.put_image = ct.WINFUNCTYPE(None, ct.c_uint, ct.c_int,
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
            ct.c_uint, ct.c_uint, ct.c_void_p)(address)

    def transfer(self, texture, level: int, data: bytes | None = None) -> bytes:
        texture.use(0)
        dimensions = []
        for pname in (0x1000, 0x1001, 0x8071):  # width, height, array layers
            value = ct.c_int()
            self.get_level(0x8C1A, level, pname, ct.byref(value))
            dimensions.append(value.value)
        width, height, layers = dimensions
        if min(dimensions) < 1 or texture.components != 3 or texture.dtype != "f1":
            raise ValueError("expected allocated RGB8 array level")
        size = width * height * layers * 3
        if data is not None and len(data) != size:
            raise ValueError("array mip byte length mismatch")
        pname = 0x0CF5 if data is not None else 0x0D05  # UNPACK/PACK_ALIGNMENT
        previous = ct.c_int()
        self.get_int(pname, ct.byref(previous))
        self.pixel_store(pname, 1)
        try:
            memory = ct.create_string_buffer(data, size) if data is not None else ct.create_string_buffer(size)
            if data is None:
                self.get_image(0x8C1A, level, 0x1907, 0x1401, memory)
            else:
                self.put_image(0x8C1A, level, 0, 0, 0, width, height, layers, 0x1907, 0x1401, memory)
            return memory.raw
        finally:
            self.pixel_store(pname, previous.value)


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def write_bundle(path: Path, manifest: dict, blobs: dict[str, bytes]) -> None:
    if path.exists():
        raise FileExistsError(f"snapshot already exists: {path}")
    manifest["files"] = {name: {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
                         for name, raw in blobs.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, raw in blobs.items():
            bundle.writestr(name, raw)
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2))


def read_bundle(path: Path) -> tuple[dict, dict[str, bytes]]:
    with zipfile.ZipFile(path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported facade snapshot schema")
        blobs = {}
        # Never extract archive paths to the filesystem.
        for name, expected in manifest["files"].items():
            raw = bundle.read(name)
            if len(raw) != expected["bytes"] or sha256(raw).hexdigest() != expected["sha256"]:
                raise ValueError(f"snapshot integrity mismatch: {name}")
            blobs[name] = raw
    if len(blobs["vertices.bin"]) != 9 * 40:
        raise ValueError("expected exactly nine packed scene vertices")
    return manifest, blobs


def gpu_info(ctx) -> dict:
    return {name: ctx.info.get(name, "unknown") for name in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")}


def capture_texture(name, texture, unit, blobs, *, array=False) -> dict:
    spec = {"name": name, "size": list(texture.size), "components": texture.components,
            "dtype": texture.dtype, "unit": unit, "array": array,
            "filter": list(texture.filter), "repeat_x": texture.repeat_x,
            "repeat_y": texture.repeat_y, "levels": []}
    if array:
        # ModernGL caches 0 until explicitly set; GL's effective default is 1.
        spec["anisotropy"] = max(1.0, texture.anisotropy)
        transfer = ArrayMips()
        count = max(texture.size[:2]).bit_length()
        levels = [transfer.transfer(texture, level) for level in range(count)]
    else:
        levels = [texture.read(alignment=1)]
    for level, raw in enumerate(levels):
        key = f"textures/{name}/{level}.bin"
        blobs[key] = raw
        spec["levels"].append(key)
    return spec


def restore_textures(ctx, manifest, blobs, *, allow_sampler_mismatch: bool = False):
    """Restore the captured textures; returns (textures, sampler_state).

    Sampler state is reported rather than assumed. A device that cannot apply
    the captured anisotropy is refused by default, because measuring one
    sampler configuration while claiming another is the failure this guard
    exists to prevent. `allow_sampler_mismatch` is for the one case where
    parity is impossible by construction — comparing against an implementation
    that does not expose the extension at all — and every caller that sets it
    records the achieved value beside the result.
    """

    textures = []
    sampler_state = []
    try:
        for spec in manifest["textures"]:
            factory = ctx.texture_array if spec["array"] else ctx.texture
            texture = factory(tuple(spec["size"]), spec["components"],
                              blobs[spec["levels"][0]], dtype=spec["dtype"], alignment=1)
            textures.append(texture)
            if spec["array"]:
                texture.build_mipmaps(max_level=len(spec["levels"]) - 1)
                transfer = ArrayMips()
                for level, key in enumerate(spec["levels"]):
                    transfer.transfer(texture, level, blobs[key])
                    if transfer.transfer(texture, level) != blobs[key]:
                        raise RuntimeError(f"array mip restore differs: {key}")
                texture.anisotropy = spec["anisotropy"]
                achieved = float(texture.anisotropy)
                matched = achieved == spec["anisotropy"]
                sampler_state.append({
                    "texture": spec["name"],
                    "anisotropy_requested": float(spec["anisotropy"]),
                    "anisotropy_achieved": achieved,
                    "matched": matched,
                })
                if not matched and not allow_sampler_mismatch:
                    raise RuntimeError(
                        "device cannot restore captured anisotropy "
                        f"(requested {spec['anisotropy']}, achieved {achieved})"
                    )
            elif texture.read(alignment=1) != blobs[spec["levels"][0]]:
                raise RuntimeError("texture restore differs")
            texture.filter = tuple(spec["filter"])
            texture.repeat_x, texture.repeat_y = spec["repeat_x"], spec["repeat_y"]
        if ctx.error != "GL_NO_ERROR":
            raise RuntimeError("OpenGL error restoring textures")
        return textures, sampler_state
    except BaseException:
        for texture in textures:
            texture.release()
        raise


def restore_program(ctx, spec, blobs):
    program = ctx.program(vertex_shader=blobs["scene.vert"].decode(),
                          fragment_shader=blobs[spec["fragment"]].decode())
    try:
        uniforms = {name for name in program if isinstance(program[name], moderngl.Uniform)}
        if uniforms != set(spec["uniforms"]):
            raise ValueError("active uniform set differs from capture")
        for name, encoded in spec["uniforms"].items():
            raw = bytes.fromhex(encoded)
            if len(program[name].read()) != len(raw):
                raise ValueError(f"uniform size mismatch: {name}")
            program[name].write(raw)
            if program[name].read() != raw:
                raise RuntimeError(f"uniform restore differs: {name}")
        return program
    except BaseException:
        program.release()
        raise


def measure(ctx, program, vertices: bytes, bind, iterations: int) -> dict:
    if iterations < 2:
        raise ValueError("iterations must be at least 2")
    previous = ctx.fbo
    colour = ctx.texture(FRAME_SIZE, 4, dtype="f2")
    depth = ctx.depth_texture(FRAME_SIZE)
    framebuffer = ctx.framebuffer([colour], depth)
    buffer = ctx.buffer(vertices)
    formats, attributes = [], []
    for fmt, name, size in zip(LAYOUT[0].split(), LAYOUT[1:], (12, 12, 4, 8, 4), strict=True):
        if name in program:
            formats.append(fmt)
            attributes.append(name)
        else:
            formats.append(f"{size}x")
    vao = ctx.vertex_array(program, [(buffer, " ".join(formats), *attributes)] if attributes else [])
    result = {}
    reference = None
    try:
        framebuffer.use()
        framebuffer.viewport = (0, 0, *FRAME_SIZE)
        framebuffer.scissor = None
        ctx.enable_only(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"
        framebuffer.depth_mask = True
        for case, ranges in CASES.items():
            print(f"  {case}", flush=True)
            counts = Counter()
            for _ in range(iterations):
                framebuffer.use()
                framebuffer.clear(0, 0, 0, 1, depth=1)
                bind()
                for first, count in ranges:
                    vao.render(moderngl.TRIANGLES, first=first, vertices=count)
                ctx.finish()
                raw = framebuffer.read(viewport=REGION, components=4, dtype="f2", alignment=1)
                counts[raw] += 1
            ordered = counts.most_common()
            if reference is None:
                reference = ordered[0][0]
            if not all(np.isfinite(np.frombuffer(raw, dtype=np.float16)).all() for raw in counts):
                raise RuntimeError("non-finite framebuffer output")
            error = ctx.error
            if error != "GL_NO_ERROR":
                raise RuntimeError(error)
            target = np.frombuffer(reference, dtype=np.float16).reshape(4, 2, 4)[..., :3].astype(np.float32)
            result[case] = {
                "unique_states": len(counts), "differing_iterations": iterations - ordered[0][1],
                "states": [{"sha256": sha256(raw).hexdigest(), "count": count} for raw, count in ordered],
                "samples_outside_target_control": sum(count for raw, count in counts.items() if raw != reference),
                "max_abs_rgb_from_target": max(float(np.max(np.abs(
                    np.frombuffer(raw, dtype=np.float16).reshape(4, 2, 4)[..., :3].astype(np.float32) - target))) for raw in counts),
                "gl_error": error,
            }
        stable = all(result[case]["unique_states"] == 1 and result[case]["samples_outside_target_control"] == 0
                     for case in ("target_only", "target_only_after"))
        return {"cases": result, "target_controls_stable": stable,
                "trigger_effect_observed": any(result[case]["samples_outside_target_control"] > 0
                    for case in ("trigger_then_target", "target_then_trigger")) if stable else None}
    finally:
        if previous is not None:
            previous.use()
        vao.release()
        buffer.release()
        framebuffer.release()
        depth.release()
        colour.release()


def replay(ctx, manifest, blobs, iterations: int, *, allow_sampler_mismatch: bool = False):
    previous = ctx.fbo
    anchor = ctx.simple_framebuffer((1, 1))
    anchor.use()
    textures = []
    try:
        textures, sampler_state = restore_textures(
            ctx, manifest, blobs, allow_sampler_mismatch=allow_sampler_mismatch
        )
        def bind():
            for texture, spec in zip(textures, manifest["textures"], strict=True):
                texture.use(spec["unit"])
        results = {}
        for name, spec in manifest["programs"].items():
            print(f"replay {name}", flush=True)
            program = restore_program(ctx, spec, blobs)
            try:
                results[name] = measure(ctx, program, blobs["vertices.bin"], bind, iterations)
            finally:
                program.release()
        return results, sampler_state
    finally:
        for texture in textures:
            texture.release()
        if previous is not None:
            previous.use()
        anchor.release()
