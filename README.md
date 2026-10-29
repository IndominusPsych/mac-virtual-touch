# Virtual Touch

**Use your webcam as a multi-touch trackpad.** Virtual Touch tracks your hands with
[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
and turns hand gestures into the same operations a physical trackpad gives you:
pointer movement, click, double click, drag, right click, two-finger scroll with
inertia, pinch zoom, rotate, and three/four-finger swipes. It works with one hand
or two, on **macOS** and **Windows** (Linux best-effort).

> Status: early (v0.1). The core works; thresholds still need tuning across
> different cameras and lighting. Feedback and PRs are welcome.

---

## Gestures

The thumb acts as the mouse button. The number of extended fingers picks the
trackpad "finger count".

### One hand (your dominant hand)

| Trackpad action | Gesture |
|---|---|
| Move pointer | ✋ Just move your **relaxed hand**, in any natural pose. The cursor follows your knuckles (the ring in the preview), so curling or pinching fingers doesn't nudge it |
| Lift finger (reposition) | ✊ Make a **tight fist**: the cursor stays put while you move |
| Click | 👌 Quick **thumb + index** pinch |
| Double / triple click | Two or three quick pinches |
| Drag / select | Pinch and **hold** (about 0.3 s), move, release |
| Right click | Quick **thumb + middle** pinch |
| Two-finger scroll | ✌️ **Index + middle** up, ring + pinky folded, move in any direction (flick for inertia) |
| Zoom | 🤏 **Grab**: thumb, index and middle tips together. Push toward the camera to zoom in, pull back to zoom out |
| Rotate | Grab pose, then roll your wrist |
| 3-finger swipe | **Index + middle + ring** up, quick **flick** ←/→ (switch desktop), ↑ (Mission Control / Task View), ↓ (App Exposé / show desktop) |
| 4-finger swipe | 🖐 **Open palm**, quick **flick** ←/→ (forward/back), ↑, ↓ (show desktop) |

Moving slowly with three fingers or an open palm just moves the pointer. Only a
fast flick counts as a swipe, and the pointer snaps back to where it was. When
you click, the cursor rewinds about 80 ms so the small hand movement of pinching
doesn't make you miss.

### Two hands

| Trackpad action | Gesture |
|---|---|
| Pinch zoom | Pinch with **both** hands, move them apart or together |
| Rotate | Pinch with both hands, then turn them like a steering wheel |
| Pause / resume | Hold **both palms open** for 1.5 s (or press `Ctrl+Alt+P`) |

Every swipe, zoom and rotate gesture sends a keyboard shortcut. You can remap
them per OS in the config ([below](#configuration)).

---

## Install

Requirements: Python **3.9–3.12** (MediaPipe doesn't support 3.13 yet), a webcam.

```bash
git clone https://github.com/IndominusPsych/virtual-touch.git
cd virtual-touch

# with uv (recommended)
uv venv --python 3.11 && uv pip install -e .

# or with plain pip
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

The first run downloads the hand model (about 7.5 MB) into your user cache.

### macOS permissions (required)

macOS asks for two permissions. Grant both to the app you **launch from**
(Terminal, iTerm, VS Code, and so on), then **restart that app**:

1. **Camera**: *System Settings → Privacy & Security → Camera*
2. **Accessibility** (to move the cursor and send keys): *System Settings → Privacy & Security → Accessibility*
3. *(Optional)* **Input Monitoring**, only needed for the global pause hotkey.

Run `virtual-touch doctor` to check what's granted.

### Windows

No extra permissions are needed. If Windows Defender SmartScreen or your
antivirus warns about synthetic input, allow Python. Virtual Touch makes itself
DPI-aware, so the cursor matches the screen on scaled displays (125%, 150%, and so on).

---

## Usage

```bash
virtual-touch                 # start (same as `virtual-touch run`)
virtual-touch run --dry-run   # show recognised gestures without touching the real cursor
virtual-touch run --hand left # left-handed
virtual-touch run --absolute  # hand position maps directly to screen position
virtual-touch run --camera 1  # pick a camera
virtual-touch calibrate       # measure your pinch and save personal thresholds
virtual-touch doctor          # show detected displays, cameras, permissions
virtual-touch list-cameras
virtual-touch config --init   # create an editable config file
virtual-touch config --path   # where that file lives
```

Keys in the preview window: **q** quit · **p** pause · **d** toggle dry-run · **m** relative/absolute pointer.

**Tips for good tracking**
- Sit 40–80 cm from the camera with your hand roughly at chest height.
- Light your hand from the front. A bright window behind you hurts tracking.
- Start with `--dry-run` to learn the poses, then run it for real. In dry-run the real cursor doesn't move, by design.
- The label under your hand shows the pose and each finger's state (index, middle, ring, pinky): `E` extended, `r` relaxed, `C` curled. If a pose isn't recognised, this shows which finger is off.
- Run `virtual-touch calibrate` once. It adapts the pinch thresholds to your hand and camera.

### Small camera field of view?

Laptop webcams are aimed at your face, so your hands easily leave the frame.
Virtual Touch makes up for this in software:

| Technique | What it does | Setting |
|---|---|---|
| **Full-sensor capture** | Requests the camera's largest format instead of a cropped 16:9 mode. On a MacBook Neo this switches from 1920×1080 to the square 1552×1552 sensor view: **~43% more vertical coverage**, including the area below your face where your hands are | `camera.width/height: auto` |
| **Edge push** | The box in the preview is your "trackpad". Push your knuckles past an edge and the cursor keeps gliding that way, faster the further out you go, so small movements reach the whole screen and other monitors | `pointer.edge_push`, `edge_region`, `edge_speed` |
| **Pointer acceleration** | Slow movements are precise and fast flicks cross the screen, so you need little room | `pointer.min_gain/max_gain` |
| **Clutch** | Make a fist, bring your hand back to the middle, open it, and keep going, like lifting your finger off a trackpad | always on |
| **Distance compensation** | Gestures are measured in palm lengths, so you can **sit further back** to fit more of your hand's movement in view without the pointer slowing down | `pointer.distance_compensation` |
| **Leaving-view warning** | The preview warns you, with a direction, when your hand is about to leave the frame | always on |

Hardware tips: tilt the laptop screen back a little so the camera sees lower.
On macOS, turn off *Center Stage* (Control Center → Video Effects) if your Mac
has it, because it crops the image. An iPhone used as a Continuity Camera
(`--camera 1`) has a much wider lens.

---

## Configuration

Virtual Touch detects everything that differs between laptops each time it
starts:

| Detected at start | How |
|---|---|
| Displays: count, position, size, Retina/DPI scale | Quartz (macOS), Win32 per-monitor DPI APIs (Windows), `screeninfo` (Linux) |
| Camera | Uses the first camera that delivers frames |
| Pointer & scroll gain | Scaled from your main display's height |
| Hand distance | Thresholds are measured in palm lengths, so they work at any distance |

You can override any of these. Create your config with `virtual-touch config --init`
and edit it. It lives at:

- macOS: `~/Library/Application Support/virtual-touch/config.yaml`
- Windows: `%LOCALAPPDATA%\virtual-touch\config.yaml`
- Linux: `~/.config/virtual-touch/config.yaml`

Or override values for a single run:

```bash
virtual-touch run --set pointer.speed=1.5 --set scroll.natural=false --set camera.index=1
```

Common settings (see [`default_config.yaml`](src/virtual_touch/default_config.yaml) for all of them, with comments):

| Setting | Default | Meaning |
|---|---|---|
| `camera.index` | `auto` | Camera to use |
| `camera.width/height` | `auto` | Capture format (`auto` = largest = widest view) |
| `camera.process_width` | 640 | Frames are downscaled to this width for tracking |
| `pointer.edge_push` | `true` | Glide the cursor while your knuckles are past the box edge |
| `hands.dominant` | `right` | Hand that drives the pointer |
| `pointer.mode` | `relative` | `relative` (trackpad-like) or `absolute` |
| `pointer.speed` | `1.0` | Pointer speed multiplier |
| `pointer.min_gain/max_gain` | 1.2 / 4.0 | Precision vs. acceleration |
| `screen.displays` | `auto` | Override display geometry if detection is wrong |
| `gestures.pinch_close/pinch_open` | 0.28 / 0.40 | Pinch thresholds (`calibrate` sets them) |
| `gestures.drag_hold_time` | 0.28 s | Pinch hold before a drag starts |
| `scroll.speed`, `scroll.natural`, `scroll.inertia` | 2.0, true, true | Scrolling feel |
| `actions.<os>.<gesture>` | per OS | Keyboard shortcut for each swipe / zoom / rotate |

Example of remapping a gesture:

```yaml
actions:
  darwin:
    swipe4_up: cmd+space        # open Spotlight
  win32:
    swipe4_up: win+s
```

---

## How it works

```
camera ─► MediaPipe HandLandmarker ─► hand features ─► gesture state machine ─► OS backend
          (21 landmarks per hand)     (palm-normalised     (click/drag/scroll/     (Quartz CGEvents on macOS,
                                       pinches, poses)      swipe/zoom logic)       pynput/SendInput on Windows)
```

- **Trackpad-style relative pointer** with a One-Euro filter (smooth when still,
  responsive when moving) and pointer acceleration.
- **Natural hover**: the cursor follows your knuckles whenever your hand is relaxed. Only a fist "lifts" it.
- **Click stability**: on pinch, the cursor rewinds to where it was just before, so clicks land where you aimed.
- **Hysteresis and debouncing** on every pose, so gestures don't flicker.
- **Native macOS events**: real double-click state, drag events, pixel-precise
  scrolling, and Spaces / Mission Control shortcuts that work.

See [`docs/PLAN.md`](docs/PLAN.md) for the design and roadmap.

## Development

```bash
uv pip install -e ".[dev]"
pytest                       # runs without a camera: synthetic hands drive the gesture engine
```

Project layout:

```
src/virtual_touch/
  cli.py            command-line entry point
  app.py            main loop
  tracker.py        camera thread + MediaPipe
  hand.py           landmark features, pose classification
  gestures.py       gesture state machine
  filters.py        One-Euro filter
  system_info.py    display / camera / permission detection
  calibrate.py      threshold calibration
  overlay.py        preview window
  backends/         macOS (Quartz), generic (pynput), dry-run
  default_config.yaml
tests/              engine tests with synthetic hands
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Hand keeps leaving the camera view | See [Small camera field of view](#small-camera-field-of-view); sit further back |
| Cursor drifts on its own near an edge | Widen `pointer.edge_region` or raise `pointer.edge_dwell` |
| Low frame rate on an older webcam | Set `camera.width: 1280` and `camera.height: 720` |
| `No working camera found` | Grant Camera permission to your terminal and restart it; try `--camera 1` |
| Overlay works but cursor doesn't move (macOS) | Grant Accessibility permission to your terminal and restart it |
| Clicks fire by accident / not at all | Run `virtual-touch calibrate`, or adjust `gestures.pinch_close` |
| Fist not detected (cursor won't stop) | Raise `gestures.finger_curl` (for example 1.35) |
| Relaxed hand stops the cursor | Lower `gestures.finger_curl` (for example 1.05) |
| Swipes fire when you just move | Raise `gestures.swipe_distance` or lower `gestures.swipe_window` |
| Pointer jittery | Lower `pointer.smoothing_min_cutoff` (for example 0.8) |
| Pointer laggy | Raise `pointer.smoothing_beta` |
| Scroll goes the wrong way | `--set scroll.natural=false` (or `scroll.invert_x=true` for horizontal) |
| Swipe shortcut does nothing | Check the shortcut is enabled in your OS (macOS: *Keyboard → Keyboard Shortcuts → Mission Control*) |

## Roadmap

- Menu-bar / system-tray app and packaged `.app` / `.exe` releases
- Native macOS magnify/rotate gesture events (true smooth pinch-zoom)
- Per-app gesture profiles
- Learned gesture classifier trained on recorded landmarks

## License

[MIT](LICENSE)
