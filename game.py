"""FlyDash: a fast, headless, vectorised Geometry-Dash-style simulator.

Everything is in block units. Physics runs at 240 Hz; the controller (fly brain) decides
hold / release at 60 Hz, like a player on a 60 Hz screen. Thousands of runs can be
simulated in parallel because every agent is just a row in a few numpy arrays.
"""
import numpy as np

RES = 1 / 8            # grid resolution (blocks)
H_BLOCKS = 16          # world height
VX = 10.386            # GD 1x speed, blocks / s
TPS = 240              # physics ticks per second
DT = 1 / TPS
TICKS_PER_DECISION = 4 # -> 60 Hz decisions

CUBE, SHIP, WAVE = 0, 1, 2
G_CUBE, JUMP_V, VFALL = 100.0, 21.0, 26.0
SHIP_A, SHIP_V = 48.0, 9.0
SIZE = {CUBE: (1.0, 1.0), SHIP: (1.0, 0.6), WAVE: (0.4, 0.4)}   # (w, h) solid box
HZ_INSET = {CUBE: 0.12, SHIP: 0.08, WAVE: 0.04}                   # hazard box inset


class Level:
    def __init__(self, length, name='level'):
        self.name = name
        self.length = length
        self.nc = int(np.ceil((length + 12) / RES)); self.nr = int(H_BLOCKS / RES)
        self.hz = np.zeros((self.nc, self.nr), bool)
        self.so = np.zeros((self.nc, self.nr), bool)
        self.mode_at = np.zeros(self.nc, np.int8)
        self.slope_at = np.ones(self.nc, np.float32)
        self.objects = []   # for rendering: (kind, x0, y0, x1, y1)
        self.portals = []

    def _cols(self, x0, x1):
        return max(0, int(round(x0 / RES))), min(self.nc, int(round(x1 / RES)))

    def _rows(self, y0, y1):
        return max(0, int(round(y0 / RES))), min(self.nr, int(round(y1 / RES)))

    def solid(self, x0, x1, y0, y1):
        c0, c1 = self._cols(x0, x1); r0, r1 = self._rows(y0, y1)
        self.so[c0:c1, r0:r1] = True; self.objects.append(('block', x0, y0, x1, y1))

    def hazard(self, x0, x1, y0, y1, kind='hz'):
        c0, c1 = self._cols(x0, x1); r0, r1 = self._rows(y0, y1)
        self.hz[c0:c1, r0:r1] = True; self.objects.append((kind, x0, y0, x1, y1))

    def spike(self, x, y=1.0, n=1, down=False):
        for i in range(n):
            xx = x + i
            if down:
                self.hazard(xx + 0.3, xx + 0.7, y - 0.5, y, 'hz_hidden')
                self.objects.append(('spike_down', xx, y - 1, xx + 1, y))
            else:
                self.hazard(xx + 0.3, xx + 0.7, y, y + 0.5, 'hz_hidden')
                self.objects.append(('spike', xx, y, xx + 1, y + 1))

    def set_mode(self, x, mode, slope=1.0):
        c = int(round(x / RES)); self.mode_at[c:] = mode; self.slope_at[c:] = slope
        self.portals.append((x, int(mode), float(slope)))

    def corridor(self, xs, lo, hi):
        """Hazard everywhere below lo(x) and above hi(x) for x in xs (arrays of same length)."""
        for x, l, h in zip(xs, lo, hi):
            c = int(round(x / RES))
            if c >= self.nc: continue
            rl = int(np.floor(l / RES)); rh = int(np.ceil(h / RES))
            self.hz[c, :max(rl, 0)] = True; self.hz[c, min(rh, self.nr):] = True
        self.objects.append(('corridor', np.asarray(xs).tolist(), np.asarray(lo).tolist(), np.asarray(hi).tolist()))

    def finalize(self):
        nr = self.nr; blk = self.hz | self.so
        r = np.arange(nr)
        # last solid row <= r, first solid row >= r  (per column)
        self.so_last = np.where(self.so, r, -1); np.maximum.accumulate(self.so_last, 1, out=self.so_last)
        f = np.where(self.so, r, nr)[:, ::-1]; self.so_first = np.minimum.accumulate(f, 1)[:, ::-1].copy()
        # clearance to anything blocking, above and below (in rows)
        up = np.where(blk, r, nr)[:, ::-1]; self.blk_up = np.minimum.accumulate(up, 1)[:, ::-1].copy()
        dn = np.where(blk, r, -1); self.blk_dn = np.maximum.accumulate(dn, 1)
        # next hazard / solid column at or after c, per row (for looming rangefinder rays)
        cidx = np.arange(self.nc)[:, None]
        self.nxt_hz = np.minimum.accumulate(np.where(self.hz, cidx, self.nc)[::-1], 0)[::-1].copy()
        self.nxt_so = np.minimum.accumulate(np.where(self.so, cidx, self.nc)[::-1], 0)[::-1].copy()
        # 2-D prefix sums for O(1) box queries
        self.P_hz = np.zeros((self.nc + 1, nr + 1), np.int32); self.P_hz[1:, 1:] = self.hz.cumsum(0).cumsum(1)
        self.P_so = np.zeros((self.nc + 1, nr + 1), np.int32); self.P_so[1:, 1:] = self.so.cumsum(0).cumsum(1)
        # coarse 0.5-block "retina" grids (max-pooled)
        k = 4; ncc = self.nc // k; nrc = nr // k
        self.hz_c = self.hz[:ncc * k].reshape(ncc, k, nrc, k).any((1, 3))
        self.so_c = self.so[:ncc * k].reshape(ncc, k, nrc, k).any((1, 3))
        return self


def box_any(P, c0, c1, r0, r1):
    c0 = np.clip(c0, 0, P.shape[0] - 1); c1 = np.clip(c1, 0, P.shape[0] - 1)
    r0 = np.clip(r0, 0, P.shape[1] - 1); r1 = np.clip(r1, 0, P.shape[1] - 1)
    s = P[c1, r1] - P[c0, r1] - P[c1, r0] + P[c0, r0]
    return (s > 0) & (c1 > c0) & (r1 > r0)


class Sim:
    """Batch of independent runs. x0 can differ per agent (practice-mode checkpoints)."""
    def __init__(self, lvl, x0, y0=None, mode0=None):
        self.L = lvl; n = len(x0); self.n = n
        self.x = np.asarray(x0, np.float64).copy()
        c = (self.x / RES).astype(int)
        self.mode = lvl.mode_at[c].astype(np.int8) if mode0 is None else np.asarray(mode0, np.int8)
        if y0 is None:
            y0 = np.where(self.mode == CUBE, 1.0, 5.0)
            # start on top of whatever is below in cube mode
            for i in range(n):
                if self.mode[i] == CUBE:
                    cc = c[i]; col = lvl.so[cc:cc + 8].any(0); r = np.where(col)[0]
                    y0[i] = (r.max() + 1) * RES if len(r) else 1.0
                else:  # middle of the corridor
                    cc = c[i] + 2; free = ~(lvl.hz[cc] | lvl.so[cc]); r = np.where(free)[0]
                    y0[i] = r.mean() * RES - SIZE[int(self.mode[i])][1] / 2 if len(r) else 5.0
        self.y = np.asarray(y0, np.float64).copy()
        self.vy = np.zeros(n); self.alive = np.ones(n, bool); self.won = np.zeros(n, bool)
        self.grounded = np.zeros(n, bool); self.ticks = 0
        self.start_x = self.x.copy()

    def dims(self):
        w = np.where(self.mode == WAVE, 0.4, 1.0)
        h = np.select([self.mode == CUBE, self.mode == SHIP], [1.0, 0.6], 0.4)
        mini = (self.mode == WAVE) & (self.L.slope_at[np.clip((self.x / RES).astype(int), 0, self.L.nc - 1)] > 1.5)
        w = np.where(mini, 0.3, w); h = np.where(mini, 0.3, h)
        return w, h

    def _q(self, P, x, y, w, h, inset=0.0):
        c0 = np.floor((x + inset) / RES + 1e-9).astype(int); c1 = np.ceil((x + w - inset) / RES - 1e-9).astype(int)
        r0 = np.floor((y + inset) / RES + 1e-9).astype(int); r1 = np.ceil((y + h - inset) / RES - 1e-9).astype(int)
        return box_any(P, c0, c1, r0, r1)

    def step(self, hold):
        """One physics tick. hold: bool array."""
        L = self.L; a = self.alive & ~self.won
        if not a.any(): return
        w_old, h_old = self.dims()
        self.x = np.where(a, self.x + VX * DT, self.x)
        c = np.clip((self.x / RES).astype(int), 0, L.nc - 1)
        new_mode = L.mode_at[c]
        changed = a & (new_mode != self.mode)
        if changed.any():
            cy = self.y + h_old / 2
            self.mode = np.where(changed, new_mode, self.mode)
            w, h = self.dims(); self.y = np.where(changed, cy - h / 2, self.y)
            self.vy = np.where(changed, np.clip(self.vy * 0.4, -SHIP_V, SHIP_V), self.vy)
            self.grounded &= ~changed
        w, h = self.dims()
        # horizontal crash into a solid
        crash = a & self._q(L.P_so, self.x, self.y, w, h)
        # vertical motion
        m = self.mode
        slope = L.slope_at[c]
        vy_cube = np.where(self.grounded & hold, JUMP_V, np.maximum(self.vy - G_CUBE * DT, -VFALL))
        vy_ship = np.clip(self.vy + np.where(hold, SHIP_A, -SHIP_A) * DT, -SHIP_V, SHIP_V)
        vy_wave = np.where(hold, 1.0, -1.0) * VX * slope
        vy = np.select([m == CUBE, m == SHIP], [vy_cube, vy_ship], vy_wave)
        dy = vy * DT
        y = self.y + np.where(a, dy, 0)
        grounded = np.zeros(self.n, bool)
        ov = a & self._q(L.P_so, self.x, y, w, h)
        if ov.any():
            i = np.where(ov)[0]
            c0 = np.floor(self.x[i] / RES + 1e-9).astype(int); ncol = 9
            cols = np.clip(c0[:, None] + np.arange(ncol), 0, L.nc - 1)
            colmask = (np.arange(ncol)[None] < np.ceil(w[i] / RES - 1e-9)[:, None])
            r_top = np.clip(np.ceil((y[i] + h[i]) / RES - 1e-9).astype(int) - 1, 0, L.nr - 1)
            r_bot = np.clip(np.floor(y[i] / RES + 1e-9).astype(int), 0, L.nr - 1)
            down = dy[i] <= 0
            last = np.where(colmask, L.so_last[cols, r_top[:, None]], -1).max(1)
            first = np.where(colmask, L.so_first[cols, r_bot[:, None]], L.nr).min(1)
            y_land = (last + 1) * RES
            y_ceil = first * RES - h[i]
            tol = np.abs(dy[i]) + 1e-6
            ok_land = down & (y_land - y[i] <= tol + 1e-9)
            ok_ceil = ~down & (y[i] - y_ceil <= tol + 1e-9)
            ny = np.where(ok_land, y_land, np.where(ok_ceil, y_ceil, y[i]))
            dead = ~(ok_land | ok_ceil)
            # the cube dies when it hits a ceiling; ship and wave slide
            dead |= ok_ceil & (m[i] == CUBE)
            y[i] = ny; vy[i] = np.where(ok_land | ok_ceil, 0.0, vy[i])
            grounded[i] = ok_land
            crash[i] |= dead
        self.y = np.where(a, y, self.y); self.vy = np.where(a, vy, self.vy)
        self.grounded = np.where(a, grounded, self.grounded)
        ins = np.select([m == CUBE, m == SHIP], [HZ_INSET[CUBE], HZ_INSET[SHIP]], HZ_INSET[WAVE])
        hit = a & self._q(L.P_hz, self.x, self.y, w, h, ins)
        oob = a & ((self.y < 0) | (self.y + h > H_BLOCKS))
        self.alive &= ~(crash | hit | oob)
        self.won |= self.alive & (self.x >= L.length)
        self.ticks += 1

    def decide_and_run(self, hold):
        for _ in range(TICKS_PER_DECISION):
            self.step(hold)

    # ---------------- what the fly sees ----------------
    XS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.5])
    YS = np.array([-2.5, -1.5, -0.75, 0.0, 0.75, 1.5, 2.5])
    RAY_DY = np.array([-0.75, -0.25, 0.25, 0.75, 1.25, 2.0, 3.0])
    CLR_DX = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    NFEAT = 2 * 8 * 7 + 2 * 9 + 6 + 4 * 7

    def features(self):
        L = self.L; w, h = self.dims()
        cx = self.x + w / 2; cy = self.y + h / 2
        k = 0.5
        gx = np.clip(((cx[:, None] + self.XS[None]) / k).astype(int), 0, L.hz_c.shape[0] - 1)
        gy = ((cy[:, None] + self.YS[None]) / k).astype(int)
        oob = (gy < 0) | (gy >= L.hz_c.shape[1]); gy = np.clip(gy, 0, L.hz_c.shape[1] - 1)
        hz = L.hz_c[gx[:, :, None], gy[:, None, :]] | oob[:, None, :]
        so = L.so_c[gx[:, :, None], gy[:, None, :]]
        feats = [hz.reshape(self.n, -1), so.reshape(self.n, -1)]
        # fine clearance above / below at a few look-ahead points (blocks, clipped)
        clr = []
        for dx in self.CLR_DX:
            c = np.clip(((cx + dx) / RES).astype(int), 0, L.nc - 1)
            rt = np.clip(np.ceil((self.y + h) / RES).astype(int), 0, L.nr - 1)
            rb = np.clip(np.floor(self.y / RES).astype(int) - 1, 0, L.nr - 1)
            up = (L.blk_up[c, rt] * RES - (self.y + h))
            dn = (self.y - (L.blk_dn[c, rb] + 1) * RES)
            clr += [np.clip(up, -1, 4) / 4, np.clip(dn, -1, 4) / 4]
        # looming rangefinder: distance from the box front to the next hazard / solid at several heights
        cf = np.clip(np.ceil((self.x + w) / RES).astype(int), 0, L.nc - 1)
        rr = np.clip(np.floor((self.y[:, None] + self.RAY_DY[None]) / RES).astype(int), 0, L.nr - 1)
        rays = []
        for nxt in (L.nxt_hz, L.nxt_so):
            d = np.clip(nxt[cf[:, None], rr] * RES - (self.x + w)[:, None], 0.0, 8.0)
            rays += [d / 8.0, 1.0 / (1.0 + 2.0 * d)]
        extra = [self.vy / 20, self.mode == CUBE, self.mode == SHIP, self.mode == WAVE,
                 self.grounded, np.ones(self.n)]
        return np.concatenate([f.astype(np.float32) for f in feats] +
                              [np.stack(clr, 1).astype(np.float32)] + [r.astype(np.float32) for r in rays] +
                              [
                               np.stack(extra, 1).astype(np.float32)], 1)
