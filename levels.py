"""'GRIEF (fly edition)': an original, Grief-inspired gauntlet - cube, ship, wave, mini-wave.
Deterministic (seeded), ~410 blocks (~40 s)."""
import numpy as np
from game import Level, CUBE, SHIP, WAVE


def zigzag(rng, x0, x1, y_start, slope, seg_min, seg_max, ylo, yhi, lead=0.0):
    """Piecewise-linear wave path with slope +-slope; returns dense xs, ys."""
    xs = [x0]; ys = [y_start]; d = rng.choice([-1, 1])
    if lead > 0: xs.append(x0 + lead); ys.append(y_start)
    x = xs[-1]; y = y_start
    while x < x1:
        L = rng.uniform(seg_min, seg_max)
        if y + d * slope * L > yhi or y + d * slope * L < ylo:
            d = -d
        L = min(L, (yhi - y) / slope if d > 0 else (y - ylo) / slope, x1 - x)
        L = max(L, 0.6)
        x += L; y += d * slope * L; xs.append(x); ys.append(y); d = -d
    xd = np.arange(x0, x1, 1 / 8)
    return xd, np.interp(xd, xs, ys)


def grief(seed=7):
    rng = np.random.default_rng(seed)
    L = Level(410, 'GRIEF (fly edition)')
    # ---------- 1. cube intro 0-70 ----------
    L.solid(0, 70, 0, 1)
    L.spike(12); L.spike(18, n=2); L.spike(25, n=3)
    L.solid(32, 35, 1, 2); L.spike(37)
    L.solid(41, 44, 1, 2); L.solid(44, 47, 1, 3); L.spike(44, n=3)
    L.spike(51, n=2); L.spike(56); L.spike(60, n=1); L.spike(64, n=3)
    # ---------- 2. ship 70-150 ----------
    L.set_mode(70, SHIP)
    xs = np.arange(70, 150, 1 / 8)
    lo = np.full_like(xs, 1.0); hi = np.full_like(xs, 11.0)
    for xp, top, bottom in [(80, 5.5, None), (88, None, 5.0), (96, 6.0, None), (103, None, 4.5),
                            (110, 7.0, None)]:
        m = (xs >= xp) & (xs < xp + 1.5)
        if top is not None: lo[m] = top
        if bottom is not None: hi[m] = bottom + 0.0
    m = xs >= 116
    c = 5.5 + 2.2 * np.sin((xs - 116) / 5.0); gap = np.interp(xs, [116, 150], [4.0, 3.0])
    lo[m] = (c - gap / 2)[m]; hi[m] = (c + gap / 2)[m]
    L.corridor(xs, lo, hi)
    # ---------- 3. cube 150-200 ----------
    L.set_mode(150, CUBE)
    L.solid(150, 200, 0, 1)
    L.spike(158); L.spike(163, n=3)
    L.solid(169, 176, 1, 2); L.spike(172, y=2)
    L.spike(177, n=2); L.spike(182); L.spike(186, n=3); L.spike(193)
    # ---------- 4. wave 200-245, mini wave 245-290 ----------
    L.set_mode(200, WAVE, 1.0)
    xd, yd = zigzag(rng, 200, 245, 1.5, 1.0, 1.8, 4.0, 1.5, 12.0, lead=2.5)
    hw = np.where(xd < 202, 1.25, 1.0)
    L.corridor(xd, yd - hw, yd + hw)
    L.set_mode(245, WAVE, 2.0)
    xd2, yd2 = zigzag(rng, 245, 290, yd[-1], 2.0, 0.9, 2.2, 2.0, 12.5)
    L.corridor(xd2, yd2 - 0.8, yd2 + 0.8)
    # ---------- 5. ship 290-340 ----------
    L.set_mode(290, SHIP)
    xs = np.arange(290, 340, 1 / 8)
    c = 6.0 + 2.8 * np.sin((xs - 290) / 3.6) * np.interp(xs, [290, 296], [0.3, 1.0])
    c = c + (yd2[-1] - 6.0) * np.interp(xs, [290, 297], [1.0, 0.0])
    gap = np.interp(xs, [290, 300, 340], [5.0, 3.0, 2.6])
    L.corridor(xs, c - gap / 2, c + gap / 2)
    # ---------- 6. wave finale 340-400 ----------
    L.set_mode(340, WAVE, 1.0)
    xd3, yd3 = zigzag(rng, 340, 400, float(c[-1]), 1.0, 1.0, 3.0, 2.0, 12.0, lead=2.0)
    hw = np.where(xd3 < 342, 1.3, 0.8)
    L.corridor(xd3, yd3 - hw, yd3 + hw)
    # ---------- 7. end ----------
    L.set_mode(400, CUBE)
    L.solid(400, 422, 0, 1)
    L.sections = [(0, 'cube'), (70, 'ship'), (150, 'cube'), (200, 'wave'), (245, 'mini wave'),
                  (290, 'ship'), (340, 'wave'), (400, 'end')]
    return L.finalize()
