from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import moderngl
import pytest

from tools.facade_snapshot import ArrayMips, capture_texture, read_bundle, restore_textures, write_bundle


@pytest.fixture
def ctx():
    try:
        context = moderngl.create_standalone_context(require=430)
    except Exception as error:
        pytest.skip(f"OpenGL unavailable: {error}")
    yield context
    context.release()


def test_bundle_integrity_and_no_overwrite(tmp_path):
    path = tmp_path / "snapshot.zip"
    manifest = {"schema_version": 1}
    write_bundle(path, manifest, {"vertices.bin": bytes(360)})
    assert read_bundle(path)[1]["vertices.bin"] == bytes(360)
    with pytest.raises(FileExistsError):
        write_bundle(path, manifest, {})
    corrupt = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(corrupt, "w") as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        bundle.writestr("vertices.bin", bytes(359) + b"x")
    with pytest.raises(ValueError, match="integrity"):
        read_bundle(corrupt)


@pytest.mark.opengl
def test_exact_array_mips_include_odd_rgb_rows_and_tail(ctx):
    original = ctx.texture_array((5, 3, 2), 3, bytes(90), alignment=1)
    original.build_mipmaps()
    transfer = ArrayMips()
    blobs = {}
    restored = []
    try:
        expected = []
        for level in range(3):
            count = max(5 >> level, 1) * max(3 >> level, 1) * 2 * 3
            raw = bytes((index + level * 31) % 256 for index in range(count))
            transfer.transfer(original, level, raw)
            assert transfer.transfer(original, level) == raw
            expected.append(raw)
        with pytest.raises(ValueError, match="length"):
            transfer.transfer(original, 2, b"short")
        spec = capture_texture("array", original, 14, blobs, array=True)
        restored = restore_textures(ctx, {"textures": [spec]}, blobs)
        assert [transfer.transfer(restored[0], level) for level in range(3)] == expected
        assert ctx.error == "GL_NO_ERROR"
    finally:
        for texture in restored:
            texture.release()
        original.release()


@pytest.mark.opengl
def test_direct_replay_from_unrelated_directory_imports_no_simulator(ctx, tmp_path):
    vertex = b"""#version 330
    uniform float gain;
    const vec2 p[3] = vec2[3](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
    void main() { gl_Position = vec4(p[gl_VertexID % 3] * gain, 0, 1); }
    """
    fragment = b"#version 330\nout vec4 frag_color; void main(){frag_color=vec4(1); }"
    program = ctx.program(vertex_shader=vertex.decode(), fragment_shader=fragment.decode())
    program["gain"].value = 1.0
    blobs = {"vertices.bin": bytes(360), "scene.vert": vertex, "simple.frag": fragment}
    manifest = {"schema_version": 1, "context_require": 430, "textures": [],
                "programs": {"first": {"fragment": "simple.frag", "uniforms": {"gain": program["gain"].read().hex()}},
                             "second": {"fragment": "simple.frag", "uniforms": {"gain": program["gain"].read().hex()}}}}
    program.release()
    path, output = tmp_path / "snapshot.zip", tmp_path / "report.json"
    write_bundle(path, manifest, blobs)
    script = Path(__file__).resolve().parents[1] / "tools" / "replay_facade_snapshot.py"
    process = subprocess.run([sys.executable, str(script), "--snapshot", str(path),
                              "--output", str(output), "--iterations", "2"],
                             cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(output.read_text())
    assert report["simulator_modules_loaded"] == []
    assert report["snapshot_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert all(result["target_controls_stable"] for result in report["results"].values())
    assert all(result["trigger_effect_observed"] is False for result in report["results"].values())
