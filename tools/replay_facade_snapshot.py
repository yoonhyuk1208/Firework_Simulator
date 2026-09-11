"""Replay a captured facade in a fresh context without simulator imports/assets.

This file and facade_snapshot.py can be copied beside the local ZIP and run
directly. Dependencies: Python, numpy, moderngl; Windows OpenGL 4.3 required.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys

import moderngl

if __package__:
    from .facade_snapshot import gpu_info, read_bundle, replay, save_report
else:
    from facade_snapshot import gpu_info, read_bundle, replay, save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--allow-sampler-mismatch", action="store_true",
                        help="permit a device that cannot apply the captured anisotropy; "
                             "the achieved value is recorded in the report")
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("iterations must be at least 2")
    manifest, blobs = read_bundle(args.snapshot)
    ctx = moderngl.create_standalone_context(require=manifest["context_require"])
    try:
        results, sampler_state = replay(ctx, manifest, blobs, args.iterations,
                                        allow_sampler_mismatch=args.allow_sampler_mismatch)
        imports = [name for name in sys.modules if name == "simulator" or name.startswith("simulator.")]
        if imports:
            raise RuntimeError(f"independent replay imported simulator modules: {imports}")
        report = {"schema_version": 1, "probe": "facade_snapshot_replay_v0_15c", "complete": True,
                  "gpu": gpu_info(ctx), "iterations_per_case": args.iterations,
                  "snapshot_sha256": sha256(args.snapshot.read_bytes()).hexdigest(),
                  "simulator_modules_loaded": imports, "results": results,
                  "sampler_state": sampler_state,
                  "sampler_mismatch_allowed": args.allow_sampler_mismatch,
                  "limitations": ["Exact resource replay is not a complete capture of prior driver/context history.",
                                  "Finite samples do not prove stability, necessity or a driver fault."]}
        save_report(args.output, report)
    finally:
        ctx.release()


if __name__ == "__main__":
    main()
