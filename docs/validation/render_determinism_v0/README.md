# V0 scene residual shader instrumentation

This validation step localises the remaining scene-owned determinism residual
without changing the shipped fragment shader. `tools/probe_render_determinism.py`
temporarily substitutes one diagnostic expression while the OpenGL program is
created, restores the normal shader loader immediately afterwards, freezes
simulation time, and compares the known half-open GL rectangle
`(598, 380, 600, 384)`.

The production baseline remains an eight-pixel block with a maximum linear-HDR
difference near `1.204e-3`. Raw attributes and independently exposed shading
inputs are stable, but the apparent first varying intermediate moves when the
diagnostic return point or helper-function boundary changes. Five plausible
runtime changes were tested and rejected because none removed the final
residual over long repeated captures. The shipped `scene.frag` therefore stays
unchanged.

The measurements were made on Intel Arc 140V, OpenGL 4.3, driver
32.0.101.8626. The supported conclusion is compiler/driver optimisation
sensitivity, not a confirmed defect in a particular BRDF term. Closing the
mechanism requires either the same probe on another driver/GPU or a minimal
standalone shader reproducer.

V0-11 adds that standalone reproducer in
`tools/probe_facade_shader_reproducer.py`. It uses the captured input ranges
from the known pixels and cumulatively restores the view vector, Fresnel,
environment reflection, helper boundary, zero-count light loops, and final
emission in a 2x4 RGBA16F target. On the Intel Arc 140V driver above, all eight
stages produced exactly one state over 4,096 draws each. A same-session scene
probe still varied in 27 of 95 comparisons, so the negative standalone result
is meaningful: the remaining trigger needs actual pass, interpolation,
raster/depth, viewport-position, or whole-program state.

The machine-readable result is `standalone_intel_arc_140v.json`. The next
isolation ladder should add the 1280x720 viewport and original pixel location,
actual facade triangle interpolation, depth attachment/test, and the complete
scene program in that order.

**2026-09-07 measurement correction:** V0-12 through V0-14 set scissor before
binding their measurement framebuffer. ModernGL restored that framebuffer's
full-frame scissor at bind time. Those historical JSON files therefore describe
full-frame rasterisation with an eight-pixel readback, not eight-pixel raster
isolation. They are preserved unchanged. A driver fault remains a hypothesis,
not an established cause. The corrected helper records its effective scissor.

V0-12 implements that ladder in `tools/probe_facade_pass_ladder.py`. The tool
projects the source building mesh and locks the frontmost covering pair to
triangles 34588-34589 (vertices 103764-103769), then cumulatively adds the
original viewport coordinates, triangle interpolation, depth, and complete
production scene program. All five stages remained single-state over 1,024
draws each. A full-scene contrast run still varied in 50 of 63 comparisons.

The machine-readable result is `pass_ladder_intel_arc_140v.json`. The remaining
trigger is therefore after the single-quad/full-program boundary. The next
ladder should add the complete building VAO, all scene VAOs, the preceding
land/sky draws, and per-frame uniform rewrites in that order.

V0-13 implements that batch ladder in `tools/probe_facade_batch_ladder.py`.
The six-vertex quad remained stable in V0-12, while drawing the complete
115,446-vertex building VAO with the same program and target produced the
minority state in two independent 2,048-draw runs (6 and 1 occurrences).
Additional scene VAOs, preceding land/sky draws, and uniform rewrites are not
required. Their zero-hit stages do not demonstrate stabilisation because the
rare event rate changes between runs.

The minority state changes all eight pixels. Its isolated scene-output maximum
RGB delta is `1.572609e-3`; later full-frame composition leaves the established
`1.204e-3` B residual. `batch_ladder_intel_arc_140v.json` records state hashes,
counts, pixel extents, and deltas. The next probe should distinguish full-buffer
offset from processed vertex count and binary-search the building draw length.

V0-14 implements that search in `tools/probe_facade_draw_threshold.py`. A full
VBO with only the target six vertices drawn, every prefix through the original
103,866 building vertices, and the first 11,328 rooftop-detail vertices were
stable. Adding the next three vertices (115,197 total) changed the target block
in 59 of 8,192 draws. That rooftop side triangle projects near `(1245, 375)`,
far from the target `(598-599, 380-383)`.

The boundary is data-sensitive, not merely a vertex-count threshold: replacing
the triangle with its predecessor changed 9/4,096 occurrences to 0/4,096.
Omitting it from the full batch left 1/4,096, so it is a strong amplifier rather
than an absolute requirement. Splitting the building into two draws did not
stabilise it. Finally, a compact nine-vertex VBO containing only the rooftop
triangle and target facade quad reproduced both states, while the target quad
alone stayed stable over 8,192 draws. Independent runs reversed which draw
order had the higher rate, so order frequency is not treated as causal.

The `draw_threshold_*_intel_arc_140v.json` reports preserve the complete search.
The next step is to neutralise each trigger-triangle attribute and shrink the
production fragment shader around this nine-vertex reproducer.

V0-15A implements that comparison in `tools/probe_facade_input_reduction.py`.
`input_reduction_intel_arc_140v.json` contains 32 stages of 1,024 draws each on
the same driver. Production/full-frame normal replacement, zero UV and the final
baseline varied in 61, 53 and 57 samples respectively, with the original two
hashes and `1.572609e-3` maximum RGB delta. All eight genuinely scissored
production cases matched the target-only `c85f6aac…` state. Full-frame surface
replacement (12 to 0) and position degeneration also matched that state.

Temporal stability alone is not equivalence: the first full-frame baseline and
target-then-trigger control each remained in `ec5e576a…` for all 1,024 samples,
while target-only stayed in `c85f6aac…`. The report separately counts samples
outside the corresponding target-only control's observed state set.

The environment-BRDF subset stayed in one identical state across all sixteen
cases. It retains production vertex maths and copied uniforms, but uses synthetic
material inputs; it is not numerically equivalent to production. No positive
reduced-fragment reproducer or standalone driver submission is ready yet. Next,
reduce the surface-12 concrete/scanned-material fragment path while retaining a
positive full-frame baseline and target-only control. Do not treat finite
sequential zero-hit results as a fix or proof of necessity.

The new probe saves after each stage (`complete: false` until finished), records
exact geometry and shader hashes, and checks GL errors. GPU regression tests use
occlusion queries to confirm 8 versus 921,600 covered samples and verify restoration
of the previous framebuffer's viewport/scissor. Production rendering is unchanged.

Representative commands:

```powershell
python -m tools.probe_render_determinism --view water_reflection --iterations 96 --region 598,380,600,384
python -m tools.probe_render_determinism --view water_reflection --iterations 192 --scene-term raw_normal
python -m tools.probe_render_determinism --view water_reflection --iterations 192 --scene-term radiance_environment
python -m tools.probe_render_determinism --view water_reflection --iterations 192 --scene-term radiance_after_dynamic --static-light-count 0
python -m tools.probe_facade_shader_reproducer --iterations 4096 --output docs/validation/render_determinism_v0/standalone_intel_arc_140v.json
python -m tools.probe_facade_pass_ladder --iterations 1024 --output docs/validation/render_determinism_v0/pass_ladder_intel_arc_140v.json
python -m tools.probe_facade_batch_ladder --iterations 2048 --output docs/validation/render_determinism_v0/batch_ladder_intel_arc_140v.json
python -m tools.probe_facade_draw_threshold --iterations 4096 --output docs/validation/render_determinism_v0/draw_threshold_intel_arc_140v.json
python -m tools.probe_facade_draw_threshold --compare-compact-order --iterations 8192 --output docs/validation/render_determinism_v0/draw_threshold_compact_order_intel_arc_140v.json
python -m tools.probe_facade_input_reduction --iterations 1024 --output docs/validation/render_determinism_v0/input_reduction_intel_arc_140v.json
```

Rows are OpenGL rows counted from the bottom. Image-order captures use
`height - 1 - row` instead.
