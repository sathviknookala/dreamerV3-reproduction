import math

import numpy as np

BINS = 255
LIM = 20.0


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def symexp(x):
    return np.sign(x) * np.expm1(np.abs(x))


def bins_v1():
    return np.linspace(-LIM, LIM, BINS)


def bins_v2():
    half = symexp(np.linspace(-LIM, 0.0, (BINS - 1) // 2 + 1))
    return np.concatenate([half, -half[:-1][::-1]])


def twohot(y, bins):
    p = np.zeros(len(bins))
    if y <= bins[0]:
        p[0] = 1.0
        return p
    if y >= bins[-1]:
        p[-1] = 1.0
        return p
    hi = int(np.searchsorted(bins, y))
    lo = hi - 1
    span = bins[hi] - bins[lo]
    w = (y - bins[lo]) / span
    p[lo], p[hi] = 1.0 - w, w
    return p


def read_v1(p, b):
    return symexp((p * b).sum())


def read_v2(p, b):
    return (p * b).sum()


def seq_sum(terms, dtype=np.float32):
    a = dtype(0.0)
    for t in terms:
        a = dtype(a + dtype(t))
    return a


def read_v2_split(p, b, dtype=np.float32):
    t = (p * b).astype(dtype)
    neg = t[b < 0][np.argsort(np.abs(b[b < 0]))]
    pos = t[b > 0][np.argsort(np.abs(b[b > 0]))]
    return dtype(seq_sum(neg, dtype) + seq_sum(t[b == 0], dtype) + seq_sum(pos, dtype))


def read_v2_mirror(p, b, dtype=np.float32):
    # reference algorithm: reverse the negative half onto the positive half so
    # each mirror pair cancels exactly, then sum
    p, b = p.astype(dtype), b.astype(dtype)
    n = len(b)
    m = (n - 1) // 2
    t1 = (p[:m] * b[:m])[::-1]
    t2 = (p[m:m + 1] * b[m:m + 1])
    t3 = (p[m + 1:] * b[m + 1:])
    return dtype(t2.sum(dtype=dtype) + (t1 + t3).sum(dtype=dtype))


b1, b2 = bins_v1(), bins_v2()
out = []
w = out.append

w("# symexp_twohot readout: transform-of-expectation vs expectation-of-transform")
w("")
w(f"bins={BINS}  support=+/-symexp({LIM:.0f})=+/-{symexp(LIM):.6e}")
w(f"v1 bin spacing: uniform in symlog space, b in [{b1[0]:.1f},{b1[-1]:.1f}]")
w(f"v2 bin spacing: symexp-spaced in value space, b in [{b2[0]:.6e},{b2[-1]:.6e}]")
w(f"v2 neighbours around zero: {b2[126]:.6f}, {b2[127]:.6f}, {b2[128]:.6f}")
w("")

w("## 1. Each encoding is exact for its OWN two-hot target")
w("")
w(f"{'y':>12} {'v1 symexp(E[b])':>18} {'v2 E[b]':>18}")
for y in [0.0, 1.0, -3.5, 100.0, 1e4]:
    r1 = read_v1(twohot(symlog(y), b1), b1)
    r2 = read_v2(twohot(y, b2), b2)
    w(f"{y:>12.4g} {r1:>18.6g} {r2:>18.6g}")
w("")
w("Two-hot is linear interpolation, so each recovers its target exactly. No bias here.")
w("")

w("## 2. They diverge on SPREAD distributions - what a trained critic actually emits")
w("")
w("Bimodal return: mass q on value `hi`, mass (1-q) on 0. True mean = q*hi.")
w("")
w(f"{'hi':>10} {'q':>6} {'true mean':>12} {'v1 readout':>14} {'v2 readout':>14} {'v1 err':>12}")
for hi, q in [(100.0, 0.5), (100.0, 0.1), (1000.0, 0.5), (10.0, 0.5), (1.0, 0.5)]:
    p1 = q * twohot(symlog(hi), b1) + (1 - q) * twohot(symlog(0.0), b1)
    p2 = q * twohot(hi, b2) + (1 - q) * twohot(0.0, b2)
    true = q * hi
    r1, r2 = read_v1(p1, b1), read_v2(p2, b2)
    w(f"{hi:>10.4g} {q:>6.2f} {true:>12.6g} {r1:>14.6g} {r2:>14.6g} {r1-true:>12.6g}")
w("")
w("symexp is convex on the positive half, so symexp(E[b]) <= E[symexp(b)] (Jensen).")
w("v1 reads out a symlog-space average, biased toward zero for spread mass.")
w("v2 reads out the mean of its own predicted distribution over values.")
w("")

w("## 3. Accumulation order matters in float32 - by cancellation, not magnitude")
w("")
w("The bins are symmetric, so a symmetric p must read out exactly 0. Zero-init of the")
w("twohot output weights is what makes predictions start at 0; naive accumulation can")
w("break that guarantee. Cases below all have true readout 0.")
w("")
w(f"{'case':<20}{'fsum f64':>11}{'seq f32':>14}{'np.sum f32':>15}"
  f"{'halves f32':>13}{'mirror f32':>13}")
cases = [
    ("uniform 1/255", np.full(BINS, 1.0 / BINS)),
]
for eps in [1e-4, 1e-6]:
    q = np.full(BINS, eps)
    q[(BINS - 1) // 2] = 1 - eps * (BINS - 1)
    cases.append((f"spike + tail {eps:.0e}", q))
for name, p in cases:
    t = p * b2
    ref = math.fsum(t.tolist())
    a = seq_sum(t.tolist())
    v = np.float32(t.astype(np.float32).sum(dtype=np.float32))
    sp = read_v2_split(p, b2)
    mi = read_v2_mirror(p, b2)
    w(f"{name:<20}{ref:>11.6g}{a:>14.8g}{v:>15.8g}{sp:>13.6g}{mi:>13.6g}")
w("")
w(f"Extreme bin magnitude is {abs(b2[0]):.3e}. Terms of that size cancel between the")
w("symmetric tails; in float32 an O(1e8) partial sum has an ulp near 32, so anything")
w("O(1) added between the two tail terms is absorbed and the cancellation leaves a")
w("residue instead of 0. The residue is an ABSOLUTE error floor, not a relative one,")
w("so it matters most where the true value is near zero - i.e. at initialisation.")
w("")
w("Note which reductions fail: sequential and pairwise (np.sum) break on DIFFERENT")
w("cases above. A framework's default reduction is not a safeguard.")
w("")
w("`halves` sums each signed half from small |b| to large. `mirror` is the")
w("reference algorithm (outs.py TwoHot.pred at the pinned commit): reverse the")
w("negative-half products onto the positive-half products and add ELEMENTWISE, so")
w("each mirror pair cancels in one operation, then sum. Implement `mirror` - it is")
w("exact by construction for symmetric p rather than exact by luck of ordering,")
w("and it is what the reference does.")
w("")
w("## 4. What this does NOT establish")
w("")
w("v2 removes the transform-order bias in reading the mean of the PREDICTED")
w("distribution. It does not make the value estimate an unbiased estimator of true")
w("environment return. Independently of readout order:")
w("  - the predicted distribution is a learned approximation of unknown accuracy;")
w("  - lambda-returns bootstrap off the critic, so targets carry the critic's own error;")
w("  - the two-hot target is a projection onto a finite support, and |y| > symexp(20)")
w("    saturates at the end bins;")
w("  - the slow-critic regulariser biases toward an older parameter set by construction.")
w("Unbiasedness w.r.t. environment return is not claimed and is not measurable here.")

txt = "\n".join(out)
print(txt)
