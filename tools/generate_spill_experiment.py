#!/usr/bin/env python3
"""
tools/generate_spill_experiment.py
Generates an experiment kernel with 40 cross-coupled simultaneously live variables,
forcing the register allocator to exceed the 24-register threshold.
"""

lines = [
    ".version 7.5",
    ".target sm_75",
    ".address_size 64",
    ".visible .entry _Z4testPf(.param .u64 _Z4testPf_param_0) {",
    "  .reg .b32 %r<5>;",
    "  .reg .f32 %f<150>;",
    "  .reg .b64 %rd<5>;",
    "  ld.param.u64 %rd1, [_Z4testPf_param_0];",
    "  mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;",
    "  mad.lo.s32 %r4, %r1, %r2, %r3;",
    "  mul.wide.s32 %rd2, %r4, 4;",
    "  add.s64 %rd3, %rd1, %rd2;",
    "  ld.global.f32 %f0, [%rd3];"
]

N = 36  # 36 live variables
# Compute variables f1 .. f36
for i in range(1, N + 1):
    lines.append(f"  add.f32 %f{i}, %f0, 0f3F800000;")  # each is f0 + 1.0f = 2.0f

# Cross-couple: pair i with (N + 1 - i)
# Each product is 2.0 * 2.0 = 4.0
# Sum over 18 pairs = 18 * 4.0 = 72.0f
lines.append("  mov.f32 %f100, 0f00000000;")
for i in range(1, (N // 2) + 1):
    j = N + 1 - i
    prod_reg = 100 + i
    lines.append(f"  mul.f32 %f{prod_reg}, %f{i}, %f{j};")
    lines.append(f"  add.f32 %f100, %f100, %f{prod_reg};")

lines.append("  st.global.f32 [%rd3], %f100;")
lines.append("  ret;")
lines.append("}")

with open("experiments/005_register_spill.ptx", "w") as f:
    f.write("\n".join(lines))

print("[DONE] Generated experiments/005_register_spill.ptx")
