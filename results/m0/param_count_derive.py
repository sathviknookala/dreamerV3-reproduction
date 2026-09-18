import math

A = 6
DETER, HIDDEN, STOCH, CLASSES, BLOCKS = 512, 64, 32, 4, 8
DEPTHS = (8, 12, 16, 16)
KERNEL, MINRES, BSPACE, DECUNITS = 5, 4, 8, 64
BINS = 255
TOKENS = MINRES * MINRES * DEPTHS[-1]
FEAT = DETER + STOCH * CLASSES

P = []


def add(name, *shapes):
    for s in shapes:
        P.append((name, s, math.prod(s)))


def linear(name, i, o, norm=False):
    add(f"{name}/kernel", (i, o))
    add(f"{name}/bias", (o,))
    if norm:
        add(f"{name}norm/scale", (o,))


def conv(name, ci, co, norm=False):
    add(f"{name}/kernel", (KERNEL, KERNEL, ci, co))
    add(f"{name}/bias", (co,))
    if norm:
        add(f"{name}norm/scale", (co,))


linear("dyn/dynin0", DETER, HIDDEN, norm=True)
linear("dyn/dynin1", STOCH * CLASSES, HIDDEN, norm=True)
linear("dyn/dynin2", A, HIDDEN, norm=True)
# dynhid0: BlockLinear(2048 -> 512, g=8); input is deter-block ++ broadcast 3*hidden
add("dyn/dynhid0/kernel", (BLOCKS, (DETER + 3 * HIDDEN * BLOCKS) // BLOCKS, DETER // BLOCKS))
add("dyn/dynhid0/bias", (DETER,))
add("dyn/dynhid0norm/scale", (DETER,))
add("dyn/dyngru/kernel", (BLOCKS, DETER // BLOCKS, 3 * DETER // BLOCKS))
add("dyn/dyngru/bias", (3 * DETER,))
linear("dyn/obs0", DETER + TOKENS, HIDDEN, norm=True)
linear("dyn/obslogit", HIDDEN, STOCH * CLASSES)
linear("dyn/prior0", DETER, HIDDEN, norm=True)
linear("dyn/prior1", HIDDEN, HIDDEN, norm=True)
linear("dyn/priorlogit", HIDDEN, STOCH * CLASSES)

chans = (3,) + DEPTHS
for i in range(len(DEPTHS)):
    conv(f"enc/cnn{i}", chans[i], chans[i + 1], norm=True)

add("dec/sp0/kernel", (BLOCKS, DETER // BLOCKS, TOKENS // BLOCKS))
add("dec/sp0/bias", (TOKENS,))
linear("dec/sp1", STOCH * CLASSES, 2 * DECUNITS, norm=True)
linear("dec/sp2", 2 * DECUNITS, TOKENS)
add("dec/spnorm/scale", (DEPTHS[-1],))
for i in reversed(range(len(DEPTHS) - 1)):
    ci = DEPTHS[i + 1]
    conv(f"dec/conv{i}", ci, DEPTHS[i], norm=True)
conv("dec/imgout", DEPTHS[0], 3)

for name, layers, out in [("rew", 1, BINS), ("con", 1, 1), ("val", 3, BINS)]:
    for i in range(layers):
        linear(f"{name}/mlp/linear{i}", FEAT if i == 0 else HIDDEN, HIDDEN, norm=True)
    linear(f"{name}/head/out", HIDDEN, out)
for i in range(3):
    linear(f"pol/mlp/linear{i}", FEAT if i == 0 else HIDDEN, HIDDEN, norm=True)
linear("pol/head/action/mean", HIDDEN, A)
linear("pol/head/action/stddev", HIDDEN, A)

groups = {}
for name, shape, n in P:
    groups.setdefault(name.split("/")[0], 0)
    groups[name.split("/")[0]] += n
total = sum(n for _, _, n in P)

print("# Parameter count DERIVED FROM THE SPECIFICATION (not measured)")
print(f"# size1m + dmc_vision, walker_walk, action dim A={A}")
print("#")
print("# This is arithmetic over the shapes recorded in docs/spec.md section 4.")
print("# It is NOT a measurement of instantiated modules. Deferred item 7 in")
print("# section 10 requires sum(p.numel()) over real modules at M3-M4; a")
print("# mismatch means the spec and the code disagree and one of them is wrong.")
print()
print(f"{'module':<10}{'params':>12}")
for k in ("dyn", "enc", "dec", "rew", "con", "pol", "val"):
    print(f"{k:<10}{groups[k]:>12,}")
print(f"{'-' * 22}")
print(f"{'TOTAL':<10}{total:>12,}")
print()
print("Excludes: the slowval EMA mirror (a byte-identical copy of val,")
print(f"          {groups['val']:,} params, untrained), optimizer state, and retnorm scalars.")
print(f"Including the slow-critic mirror: {total + groups['val']:,}")
print()
print(f"{'largest tensors':<28}{'shape':>18}{'params':>12}")
for name, shape, n in sorted(P, key=lambda x: -x[2])[:8]:
    print(f"{name:<28}{str(shape):>18}{n:>12,}")
