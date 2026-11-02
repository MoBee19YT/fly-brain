"""Fly brain server: a FlyWire fruit-fly connectome plays your Geometry Dash.

    python fly_server.py            # wait for the Geode mod (open a level in GD)
    python fly_server.py --demo     # no GD needed: the fly plays the built-in practice level

Dashboard (how the fly thinks, live):  http://127.0.0.1:5578
Requires: Python 3.10+, numpy, scipy   (pip install numpy scipy)
"""
import argparse, base64, hashlib, http.server, json, os, socket, socketserver, threading, time
import numpy as np
import scipy.sparse as sp

from game import Level, Sim, RES, CUBE, SHIP, WAVE

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = 30.0                     # GD units per block
SPEED_1X = 0.9                   # GD m_playerSpeed value for normal (1x) speed
MODE_NAMES = ['cube', 'ship', 'wave', 'ball', 'ufo', 'robot', 'spider', 'swing']
HAZARD_TYPES = {2, 47}           # Hazard, AnimatedHazard
SOLID_TYPES = {0, 25}            # Solid, Slope (slopes approximated as boxes)
WIN_BEHIND, WIN_LEN = 3.0, 24.0  # vision window around the player (blocks)


# ============================================================== the brain
class FlyBrain:
    """Frozen FlyWire circuit + the two trained layers (eyes -> 32 visual channels, DNs -> hold)."""
    def __init__(self, brain_path, weights_path):
        B = np.load(brain_path, allow_pickle=True)
        self.W = sp.csr_matrix(B['W']); self.G = B['G']
        self.ncore = self.W.shape[0]; self.ndn = int(B['n_dn'])
        self.DN = np.arange(self.ncore - self.ndn, self.ncore)
        self.gf = [int(i) for i in B['gf_idx']]
        self.groups = [str(g) for g in B['group_names']]
        self.alpha, self.substeps = 0.6, 2
        Wt = np.load(weights_path)
        self.win, self.wout, self.b = Wt['win'], Wt['wout'], float(Wt['b'].ravel()[0])
        self.layout = self._layout(B)
        self.reset()

    def _layout(self, B):
        xy = B['core_xy'].astype(float); gxy = B['group_xy'].astype(float)
        allxy = np.vstack([xy, gxy]); lo = np.nanmin(allxy, 0); hi = np.nanmax(allxy, 0)
        nrm = lambda a: np.round((a - lo) / (hi - lo), 4).tolist()
        return dict(xy=nrm(xy), gxy=nrm(gxy), groups=self.groups, ndn=self.ndn, gf=self.gf,
                    ncore=self.ncore, types=[str(t) for t in B['core_types']])

    def reset(self):
        self.r = np.zeros((1, self.ncore), np.float32)

    def step(self, f):
        u = np.maximum(f @ self.win, 0); D = u @ self.G.T
        for _ in range(self.substeps):
            pre = (self.W @ self.r.T).T + D
            self.r = self.r + self.alpha * (np.tanh(np.maximum(pre, 0)) - self.r)
        logit = float(self.r[0, self.DN] @ self.wout + self.b)
        return logit > 0, logit, np.atleast_1d(u)


# ============================================================== what the fly sees
class World:
    """Level geometry (in blocks, 'sim frame': GD ground top -> y = 1, like the training levels)."""
    def __init__(self):
        self.rects = np.zeros((0, 5))   # x0, y0, x1, y1, kind(0 solid, 1 hazard)
        self.ground_top_units = 90.0    # GD: player bottom on the ground
        self.auto_ground = True

    @property
    def y_off(self):                    # GD blocks -> sim blocks
        return self.ground_top_units / UNITS - 1.0

    def load(self, objs):
        rows = []
        for oid, typ, x, y, w, h in objs:
            if typ in HAZARD_TYPES: k = 1
            elif typ in SOLID_TYPES: k = 0
            else: continue
            if w <= 0 or h <= 0: continue
            rows.append([x / UNITS, y / UNITS - self.y_off, (x + w) / UNITS, (y + h) / UNITS - self.y_off, k])
        self.rects = np.array(rows, float).reshape(-1, 5)
        self.rects = self.rects[np.argsort(self.rects[:, 0])]
        self.maxw = float((self.rects[:, 2] - self.rects[:, 0]).max()) if len(self.rects) else 0

    def view(self, cx, cy, mode, mini, grounded, vy, upside):
        """Build the fly's local world and return (features, sim, level, origin)."""
        wx0 = np.floor((cx - WIN_BEHIND) / 0.5) * 0.5
        wy0 = 0.0 if cy < 12 else np.floor((cy - 8) / 0.5) * 0.5
        L = Level(WIN_LEN)
        R = self.rects
        if len(R):
            lo = np.searchsorted(R[:, 0], wx0 - self.maxw - 1); hi = np.searchsorted(R[:, 0], wx0 + WIN_LEN + 1)
            R = R[lo:hi]; R = R[R[:, 2] > wx0]
        for x0, y0, x1, y1, k in R:
            if upside:  # mirror the world vertically around the player
                y0, y1 = 2 * cy - y1, 2 * cy - y0
            c0 = int(round((x0 - wx0) / RES)); c1 = int(round((x1 - wx0) / RES))
            r0 = int(round((y0 - wy0) / RES)); r1 = int(round((y1 - wy0) / RES))
            if c1 <= c0: c1 = c0 + 1
            if r1 <= r0: r1 = r0 + 1
            c0, c1 = max(c0, 0), min(c1, L.nc); r0, r1 = max(r0, 0), min(r1, L.nr)
            if c1 <= c0 or r1 <= r0: continue
            (L.hz if k == 1 else L.so)[c0:c1, r0:r1] = True
        if self.auto_ground and not upside:
            rg = int(round((1.0 - wy0) / RES))
            if rg > 0: L.so[:, :rg] = True
        m = mode if mode in (CUBE, SHIP, WAVE) else CUBE
        L.mode_at[:] = m; L.slope_at[:] = 2.0 if (mini and m == WAVE) else 1.0
        L.finalize()
        w = 0.4 if m == WAVE else 1.0; h = {CUBE: 1.0, SHIP: 0.6, WAVE: 0.4}[m]
        if m == WAVE and mini: w = h = 0.3
        sim = Sim(L, [cx - wx0 - w / 2], y0=[cy - wy0 - h / 2], mode0=[m])
        sim.vy[:] = -vy if upside else vy
        sim.grounded[:] = bool(grounded)
        return sim.features(), sim, L, (wx0, wy0)


# ============================================================== shared live state
class State:
    def __init__(self, brain):
        self.lock = threading.Lock(); self.brain = brain
        self.d = dict(connected=False, source='', attempts=0, decisions=0, hold=False, logit=0.0,
                      mode='-', x=0.0, y=0.0, best_x=0.0, level_objects=0, level_len=0.0,
                      warning='', dps=0.0)
        self.r = np.zeros(brain.ncore); self.u = np.zeros(len(brain.groups)); self.grid = None
        self.hist = []    # recent logits

    def update(self, **kw):
        with self.lock: self.d.update(kw)


# ============================================================== mod connection
class Session:
    """One connected game (the Geode mod, or the built-in demo)."""
    def __init__(self, state, world, record_dir):
        self.S = state; self.world = world; self.record_dir = record_dir
        self.last_t = None; self.last_cy = None; self.hold = False; self.t_rate = time.time(); self.n_rate = 0

    def on_level(self, objs, raw_lines):
        self.world.load(objs)
        xs = [o[2] + o[4] for o in objs if o[1] in HAZARD_TYPES | SOLID_TYPES]
        self.S.update(level_objects=len(objs), level_len=(max(xs) / UNITS if xs else 0), best_x=0.0, attempts=1)
        if self.record_dir and raw_lines:
            os.makedirs(self.record_dir, exist_ok=True)
            h = hashlib.sha1('\n'.join(raw_lines).encode()).hexdigest()[:10]
            path = os.path.join(self.record_dir, f'level_{h}.txt')
            if not os.path.exists(path):
                open(path, 'w').write('\n'.join(raw_lines) + '\n')
            print(f'[fly] level with {len(objs)} objects received (saved to {path})')
        self.reset(count=False)

    def reset(self, count=True):
        self.S.brain.reset(); self.last_t = None; self.last_cy = None; self.hold = False
        if count:
            with self.S.lock: self.S.d['attempts'] += 1

    def on_state(self, t, cx_u, cy_u, mode, mini, ground, upside, speed):
        """Returns hold (bool). Decides at 60 Hz of game time, like in training."""
        cx = cx_u / UNITS; cy = cy_u / UNITS - self.world.y_off
        if self.last_t is not None and t - self.last_t < 1 / 60 - 1e-4:
            return self.hold
        vy = 0.0 if self.last_t is None else (cy - self.last_cy) / max(t - self.last_t, 1e-4)
        self.last_t, self.last_cy = t, cy
        f, sim, L, (wx0, wy0) = self.world.view(cx, cy, mode, mini, ground, vy, upside)
        hold, logit, u = self.S.brain.step(f[0])
        self.hold = hold
        warn = ''
        if mode not in (CUBE, SHIP, WAVE): warn = f'{MODE_NAMES[mode]} mode was never trained - fly treats it as cube'
        elif abs(speed - SPEED_1X) > 0.05 and speed > 0: warn = 'speed portal: the fly was trained at 1x speed'
        elif upside: warn = 'upside-down gravity: world is mirrored for the fly'
        # coarse view for the dashboard: 0.5-block cells, ~20 x 16
        k = 4; ncc = min(L.nc // k, int((WIN_LEN - 2) / 0.5)); nrc = L.nr // k
        hz = L.hz[:ncc * k].reshape(ncc, k, nrc, k).any((1, 3)); so = L.so[:ncc * k].reshape(ncc, k, nrc, k).any((1, 3))
        grid = (hz.astype(np.uint8) * 2 + so.astype(np.uint8))
        w, h = sim.dims()
        self.n_rate += 1
        now = time.time()
        with self.S.lock:
            d = self.S.d
            d.update(hold=bool(hold), logit=round(logit, 3), mode=MODE_NAMES[mode], x=round(cx, 2), y=round(cy, 2),
                     decisions=d['decisions'] + 1, best_x=max(d['best_x'], cx), warning=warn,
                     px=round(float(sim.x[0] + w[0] / 2) / 0.5, 2), py=round(float(sim.y[0] + h[0] / 2) / 0.5, 2),
                     pw=round(float(w[0]) / 0.5, 2), ph=round(float(h[0]) / 0.5, 2))
            if now - self.t_rate > 1:
                d['dps'] = round(self.n_rate / (now - self.t_rate), 1); self.t_rate = now; self.n_rate = 0
            self.S.r = self.S.brain.r[0].copy(); self.S.u = u.copy(); self.S.grid = grid
            self.S.hist.append(logit); self.S.hist = self.S.hist[-180:]
        return hold


class ModHandler(socketserver.StreamRequestHandler):
    def handle(self):
        S, world = self.server.state, self.server.world
        sess = Session(S, world, self.server.record_dir)
        S.update(connected=True, source='Geometry Dash'); print(f'[fly] game connected: {self.client_address}')
        objs = []; raw = []; reading = False
        try:
            for line in self.rfile:
                line = line.decode(errors='replace').strip()
                if not line: continue
                c = line[0]
                if reading:
                    if c == 'O':
                        raw.append(line); p = line.split()
                        objs.append((int(p[1]), int(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])))
                        continue
                    if c == 'E':
                        reading = False; sess.on_level(objs, raw); continue
                if c == 'S':
                    p = line.split()
                    hold = sess.on_state(float(p[1]), float(p[2]), float(p[3]), int(p[5]), p[6] == '1',
                                         p[7] == '1', p[8] == '1', float(p[9]))
                    self.wfile.write(b'1\n' if hold else b'0\n')
                elif c == 'L':
                    reading = True; objs = []; raw = []
                elif c == 'R':
                    sess.reset()
                elif c == 'G':     # ground override (demo): "G -1" = no automatic ground
                    v = float(line.split()[1]); world.auto_ground = v >= 0
                    S.update(source='demo (built-in level)')
                    if v >= 0: world.ground_top_units = v
                elif c == 'Q':
                    break
        except (ConnectionError, OSError):
            pass
        S.update(connected=False); print('[fly] game disconnected')


class ModServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


# ============================================================== dashboard
class Dash(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        S = self.server.state
        if self.path in ('/', '/index.html'):
            body = open(os.path.join(HERE, 'dashboard.html'), 'rb').read()
            self._send(200, 'text/html; charset=utf-8', body)
        elif self.path == '/layout.json':
            self._send(200, 'application/json', json.dumps(S.brain.layout).encode())
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream'); self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            try:
                while True:
                    with S.lock:
                        msg = dict(S.d)
                        msg['r'] = base64.b64encode((np.clip(S.r, 0, 1) * 255).astype(np.uint8).tobytes()).decode()
                        msg['u'] = np.round(np.clip(S.u / 3.0, 0, 1), 3).tolist()
                        msg['hist'] = [round(v, 2) for v in S.hist[-120:]]
                        if S.grid is not None:
                            msg['grid'] = base64.b64encode(S.grid.T.copy().tobytes()).decode()
                            msg['gw'], msg['gh'] = S.grid.shape
                    self.wfile.write(b'data: ' + json.dumps(msg).encode() + b'\n\n'); self.wfile.flush()
                    time.sleep(1 / 30)
            except (ConnectionError, OSError):
                return
        else:
            self._send(404, 'text/plain', b'not found')

    def _send(self, code, ctype, body):
        self.send_response(code); self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)


class DashServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ============================================================== demo: a fake Geometry Dash
def level_to_objects(L, y_shift_units):
    """Turn a simulator level into GD-style object rects (one per vertical run per column)."""
    lines = []
    for grid, typ in ((L.hz, 2), (L.so, 0)):
        for c in range(grid.shape[0]):
            col = grid[c]
            if not col.any(): continue
            d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
            for r0, r1 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                lines.append(f'O 0 {typ} {c * RES * UNITS:.2f} {r0 * RES * UNITS + y_shift_units:.2f} '
                             f'{RES * UNITS:.2f} {(r1 - r0) * RES * UNITS:.2f}')
    return lines


def fake_gd(port, realtime=True, attempts=None, on_result=None):
    """Plays the built-in Grief-inspired level through the exact same socket protocol as the mod."""
    from levels import grief
    L = grief(); y_shift = 2 * UNITS
    s = socket.create_connection(('127.0.0.1', port)); f = s.makefile('rwb')
    objs = level_to_objects(L, y_shift)
    f.write(b'H 1\nG -1\n'); f.write(f'L {len(objs)}\n'.encode()); f.write(('\n'.join(objs) + '\nE\n').encode()); f.flush()
    n = 0
    while attempts is None or n < attempts:
        sim = Sim(L, np.zeros(1)); t = 0.0; frame = 0; t0 = time.time()
        f.write(b'R\n'); f.flush()
        while sim.alive[0] and not sim.won[0]:
            w, h = sim.dims()
            cx = (sim.x[0] + w[0] / 2) * UNITS; cy = (sim.y[0] + h[0] / 2) * UNITS + y_shift
            mini = int(sim.mode[0] == WAVE and L.slope_at[min(int(sim.x[0] / RES), L.nc - 1)] > 1.5)
            f.write(f'S {t:.5f} {cx:.3f} {cy:.3f} 0 {int(sim.mode[0])} {mini} {int(sim.grounded[0])} 0 {SPEED_1X}\n'.encode())
            f.flush()
            hold = f.readline().strip() == b'1'
            sim.decide_and_run(np.array([hold])); t += 1 / 60; frame += 1
            if realtime:
                lag = t0 + frame / 60 - time.time()
                if lag > 0: time.sleep(lag)
        pct = min(sim.x[0] / L.length, 1) * 100
        print(f'[demo] attempt {n + 1}: {"LEVEL COMPLETE" if sim.won[0] else f"crashed at {pct:.1f}%"}')
        if on_result: on_result(pct, bool(sim.won[0]))
        n += 1
        if realtime: time.sleep(1.2)
    f.write(b'Q\n'); f.flush(); s.close()


# ============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', action='store_true', help='no GD needed: play the built-in level')
    ap.add_argument('--port', type=int, default=5577, help='port the Geode mod connects to')
    ap.add_argument('--dash-port', type=int, default=5578, help='dashboard port')
    ap.add_argument('--weights', default=os.path.join(HERE, 'weights.npz'))
    ap.add_argument('--record', default=os.path.join(HERE, 'levels_seen'), help='where to save level dumps')
    a = ap.parse_args()

    brain = FlyBrain(os.path.join(HERE, 'brain.npz'), a.weights)
    state = State(brain); world = World()
    ms = ModServer(('127.0.0.1', a.port), ModHandler); ms.state, ms.world, ms.record_dir = state, world, a.record
    threading.Thread(target=ms.serve_forever, daemon=True).start()
    ds = DashServer(('127.0.0.1', a.dash_port), Dash); ds.state = state
    threading.Thread(target=ds.serve_forever, daemon=True).start()
    print(f'[fly] brain loaded: {brain.ncore} neurons, {brain.W.nnz} connections, {len(brain.groups)} eye channels')
    print(f'[fly] waiting for Geometry Dash on port {a.port}')
    print(f'[fly] dashboard: http://127.0.0.1:{a.dash_port}   (Ctrl+C to quit)')
    try:
        if a.demo:
            print('[demo] playing the built-in GRIEF (fly edition) level in real time')
            while True:
                fake_gd(a.port, realtime=True)
        else:
            while True: time.sleep(1)
    except KeyboardInterrupt:
        print('\nbye')


if __name__ == '__main__':
    main()
