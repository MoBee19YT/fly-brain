# Fly Brain plays Geometry Dash

A real fruit-fly brain wiring diagram (FlyWire connectome: 1,060 central + descending neurons,
32 channels of visual neurons, ~709k synapses) controls your Geometry Dash, and a live dashboard
shows how it thinks.

```
Geometry Dash + Fly Brain mod  --(player + level, every tick)-->  fly_server.py  --> dashboard
                               <--------- hold / release ---------                http://127.0.0.1:5578
```

## What's in here

| file | what it is |
|---|---|
| `fly_server.py` | the brain server + dashboard (run this on your PC) |
| `dashboard.html` | the live "how it thinks" page |
| `brain.npz` | the FlyWire circuit (frozen wiring) |
| `weights.npz` | the trained eye-input and motor-readout weights |
| `game.py`, `levels.py` | the fly's vision code + the built-in practice level (for `--demo`) |
| `mod/` | source of the Geode mod (C++) + GitHub Actions build |
| `run_fly.bat` / `run_demo.bat` | double-click launchers for Windows |

## 1. Try it without the game (2 minutes)

1. Install Python 3.10+ from python.org (tick **"Add python.exe to PATH"**).
2. In this folder: `pip install numpy scipy`
3. Double-click `run_demo.bat` (or `python fly_server.py --demo`)
4. Open http://127.0.0.1:5578 and watch the fly play the built-in level.

## 2. Build the mod

You need Geometry Dash **2.2081** with **Geode v5** installed (geode-sdk.org).

**Easiest: build on GitHub (nothing to install)**
1. Create a new GitHub repo and upload the *contents* of the `mod/` folder (including `.github/`).
2. Go to **Actions** → the "Build Fly Brain mod" run → wait ~5 min → download **Build Output**.
3. Unzip it and copy `tomas.fly-brain.geode` to
   `%LOCALAPPDATA%\GeometryDash\geode\mods\` (on Steam you can also find it via Geode → mods folder button).

**Or build locally:** install the Geode CLI + SDK and Visual Studio 2022 (C++), then in `mod/`:
`geode build`

If the build fails, copy the error log and send it to Claude: the code was written against the
official 2.2081 bindings but has not been compiled yet.

## 3. Let the fly play

1. Double-click `run_fly.bat` (or `python fly_server.py`). Leave it running.
2. Open http://127.0.0.1:5578.
3. Start Geometry Dash, open a level. The mod connects automatically; the dashboard turns green.
   Practice mode works too (every respawn = new attempt).
4. Turn it off any time: Geode → Fly Brain → settings → "Let the fly play".
   If the server isn't running the game plays normally.

Every level you open is saved to `levels_seen/level_xxxx.txt`. Send that file to Claude to have the
fly trained on that exact level in the fast simulator, then drop the new `weights.npz` in here.

## Honest limitations (v0.1)

- The included weights were trained on the built-in practice level, and the fly currently clears
  about 15% of it. In a real level, expect it to die early until it's trained on that level's dump.
- Trained modes: cube, ship, wave (+ mini wave). Ball, UFO, robot, spider and swing are treated as cube.
- Trained at 1x speed. Speed portals, moving objects (move triggers), orbs and pads aren't
  understood yet. Upside-down gravity is handled by mirroring the world.
- The fly decides 60 times per second, like a 60 Hz player. Grief's tightest parts need more.
