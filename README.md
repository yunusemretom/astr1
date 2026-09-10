# ASTRO V1

A ROS 2 indoor service robot that navigates, recognizes the people it has met,
and holds a spoken Turkish conversation. Every AI engine it depends on has a
working fallback, so a missing API key or a dead network degrades the robot
instead of stopping it.

> **This is a fork, and ASTRO is not my project.** It is developed by
> [BarlineTR](https://github.com/BarlineTR/astr1), who wrote the large majority
> of it. I am a contributor with roughly 100 commits. My work is concentrated in
> specific subsystems, listed under [My contributions](#-my-contributions)
> below. Everything else in this README documents the project as a whole, not
> my authorship of it.

![demo](docs/demo.gif)

## Tech stack

![ROS 2 Humble](https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Arduino](https://img.shields.io/badge/Arduino-Mega_2560-00979D?logo=arduino&logoColor=white)
![Nav2](https://img.shields.io/badge/Nav2-SLAM_Toolbox-22314E)
![Gazebo](https://img.shields.io/badge/Gazebo-Simulation-FF6C00?logo=gazebo&logoColor=white)
![OAK-D](https://img.shields.io/badge/OAK--D_Lite-DepthAI-FF1F5A)
![Whisper](https://img.shields.io/badge/Faster--Whisper-STT-412991?logo=openai&logoColor=white)
![XTTS](https://img.shields.io/badge/XTTS_v2-Voice_cloning-4B32C3)
![License](https://img.shields.io/badge/License-Apache_2.0-blue)

ASTRO V1 is a modular ROS 2 based robotics project containing drivers and launch files for base movement, LiDAR (RPLIDAR), Vision (OAK-D Lite), and Audio (ReSpeaker 4-Mic Array) systems.

## 🚀 ROS 2 Architecture

The system has been completely modularized into ROS 2 Humble packages:

### 📦 Packages
- `astro_base`: Arduino Mega serial bridge for motor control and base sensors.
- `astro_lidar`: RPLIDAR A1 wrapper and NaN/Range filter node (`scan_filter_node`).
- `astro_vision`: OAK-D Lite driver (plus a USB-webcam publisher), face **detection and recognition** (`face_detector_node`), and the on-chip spatial pipeline (`oak_spatial_native_node`).
- `astro_audio`: ReSpeaker capture, Speech Recognition (local Faster-Whisper or cloud Whisper), **speaker recognition**, and TTS (local XTTS voice cloning, OpenAI, edge-tts).
- `astro_ai`: AI Brain — LLM engines, persona, long-term memory, conversation state machine.
- `astro_bringup`: Centralized launch files and parameters for the whole system.
- `astro_description`: URDF models and Robot State Publisher (tf2).

### 🧠 Who does what — engine matrix

Every engine is chosen in `.env`; each has a working fallback, so a missing key or a
dead network degrades the robot instead of stopping it.

| Job | Shipped default | Alternatives | Local? |
|---|---|---|---|
| LLM (chat) | OpenAI `gpt-4o-mini` (`LLM_PROVIDER="openai"`) | Groq LPU, Gemini | cloud |
| Speech → text | OpenAI `whisper-1` (`STT_ENGINE="openai"`) | Groq Whisper, local Faster-Whisper | optional |
| Text → speech | OpenAI Speech API (`TTS_ENGINE="openai"`) | ElevenLabs, XTTS, edge-tts, espeak | optional |
| Who is this face? | SFace embeddings (OpenCV ONNX) | — | **yes**, CPU |
| Who is speaking? | WeSpeaker ResNet34 (ONNX) | — | **yes**, CPU |

The shipped `.env` runs the whole speech stack through OpenAI. Every engine still has a
fallback chain, so a dead network degrades the robot instead of stopping it — but the
*selection* is explicit, not implicit: see [Configuration](#-configuration).

Face and speaker recognition need no extra pip package — they run on `opencv-python`
and `onnxruntime`, which are already installed. Only the model files are downloaded:

```bash
./scripts/install_face_models.sh     # YuNet + SFace + WeSpeaker (~63 MB total)
```

### 👥 Teaching the robot who people are

Faces and voices are enrolled separately but should use **the same name**, so the
brain can fuse "the person I see" with "the person who is talking".

```bash
# Faces — from photos, or live from the camera
./scripts/enroll_face.py --name Yunus --photos faces/Yunus
./scripts/enroll_face.py --name Yunus --capture --count 5
./scripts/enroll_face.py --list
./scripts/enroll_face.py --test some_photo.jpg     # who is this?

# Voices — from WAV files, or live from the microphone
./scripts/enroll_speaker.py --name Yunus --audio voices/Yunus
./scripts/enroll_speaker.py --name Yunus --record --count 3 --seconds 5
./scripts/enroll_speaker.py --list
```

The curated gallery of public officials lives in
`ros2_ws/src/astro_vision/data/known_faces/<person>/*.jpg` and is indexed automatically
at startup, together with their titles (`Sayın Valim`, `Sayın Başkanım`, …).

**Accuracy notes, measured on this repo's data:**

- Faces: same person 0.74–0.95, different people 0.10–0.41 cosine. Default threshold
  `FACE_MATCH_THRESHOLD=0.45`. With only one photo per person, two officials in the
  gallery reach 0.414 — adding 2–3 photos per person is what actually fixes that, and
  then the threshold can go back down to 0.40.
- Voices: same person 0.46–0.81, different people 0.16–0.33. Default
  `SPEAKER_MATCH_THRESHOLD=0.40`. Enroll 3–5 recordings of at least 3 seconds each.
- Large gallery photos (≥1500 px) used to fail detection entirely; detection now runs on
  a downscaled copy, which took undetected gallery faces from 9/25 down to 1/25.

## 🛠️ Installation & Build

**Prerequisites:** Ubuntu 22.04 with ROS 2 Humble and Python 3.10 (Jetson Orin Nano or an x86 dev machine). Everything else the installer handles.

### Quick install (one command)

```bash
git clone <this-repo> astr1 && cd astr1
./scripts/install.sh                 # apt packages + venv + build + verification
./scripts/install.sh --with-xtts     # also install local XTTS voice cloning (~5 GB)
```

The script is re-runnable — it completes what is missing and never overwrites an existing `.env`. Flags:

| Flag | Effect |
|---|---|
| `--with-xtts` | Also run `scripts/install_xtts.sh` (local XTTS voice cloning) |
| `--skip-apt` | Skip the apt step (no sudo, or packages already present) |
| `--skip-build` | Skip `colcon build` |
| `--clean` | Delete `build/ install/ log/` and rebuild from scratch |

It ends with a verification pass — 7 ROS packages present, every node importable, `serial_bridge.py` executable, entry points pointing at the venv — and exits non-zero if any check fails. Missing apt packages are reported as warnings with the exact `sudo apt install` line, since apt needs a terminal for the password.

Then:

```bash
source .venv/bin/activate
source ros2_ws/install/setup.bash
ros2 launch astro_bringup robot.launch.py
```

| Script | Purpose |
|---|---|
| `scripts/install.sh` | Full install: apt packages, uv, venv, `.env`, workspace build, verification |
| `scripts/build.sh` | Rebuild the workspace correctly (right directory, right interpreter) |
| `scripts/install_xtts.sh` | XTTS voice cloning only — clones the TTS fork from GitHub into its own venv |
| `scripts/install_stt_deps.sh` | Legacy: `pip3 install`s the STT/TTS packages system-wide. Superseded by `requirements.txt` + the venv; kept only for machines set up before the venv layout |

The sections below document what the installer does, in case you want to run the steps by hand or debug one of them.

### 1. System packages (apt)

These cannot come from pip — ROS packages, and the native libraries that `sounddevice` / `pyttsx3` bind to:

```bash
sudo apt update
sudo apt install -y python3-rosdep python3-colcon-common-extensions
sudo apt install -y ros-humble-rplidar-ros ros-humble-depthai-ros ros-humble-robot-state-publisher
sudo apt install -y libportaudio2 espeak-ng mpg123 alsa-utils
```

| Package | Needed by |
|---|---|
| `libportaudio2` | `sounddevice` — without it `audio_capture_node` fails with `OSError: PortAudio library not found` |
| `espeak-ng` | `pyttsx3` offline TTS engine, and XTTS phonemisation |
| `mpg123` | MP3 playback in `tts_node` — the `edge-tts`, `gTTS` and ElevenLabs engines all shell out to it, so without it TTS generates audio but plays nothing |
| `alsa-utils` | WAV playback in `tts_node` — XTTS produces WAV, which `mpg123` cannot play |

### 2. Python environment (uv)

Python dependencies live in a project-local virtualenv so they never mix with system or `~/.local` packages. We use [uv](https://docs.astral.sh/uv/); install it once with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
cd <repo-root>
uv venv --python 3.10 --system-site-packages .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

> **`--system-site-packages` is mandatory.** `rclpy`, `sensor_msgs`, `std_msgs`, `diagnostic_msgs`, `launch`, `launch_ros` and `ament_index_python` are not on PyPI — they come from `/opt/ros/humble`. A fully isolated venv cannot import any node in this repo.

**Dependency files.** `requirements.in` is the hand-edited list of direct dependencies; `requirements.txt` is the fully pinned lock file generated from it (73 packages including transitive ones). Never edit the lock by hand:

```bash
# after adding or removing a package in requirements.in
uv pip compile requirements.in -o requirements.txt   # resolve + pin
uv pip install -r requirements.txt                   # apply to the venv
```

Two pins are deliberate and must not be relaxed:

- `numpy==2.2.6` — matches the system/ROS NumPy so the venv copy cannot break `rclpy`'s ABI.
- `opencv-python<5` — OpenCV 5 removed the Haar cascade API, and `face_detector_node` uses `cv2.CascadeClassifier`. If a 5.x wheel is installed system-wide (`~/.local`), the venv shadows it; outside the venv the node raises `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`.

### 3. Build the workspace

```bash
./scripts/build.sh                 # everything
./scripts/build.sh astro_audio     # one package
./scripts/build.sh --clean         # wipe build/ install/ log/ first
```

**Do not run bare `colcon build`.** Two things about this workspace make the plain command produce a broken system, and the wrapper handles both:

1. **Build from `ros2_ws/`, never from the repository root.** A root build creates a *second* `install/` tree next to the source, and `ros2 launch` then runs whichever one your shell happens to have sourced — usually the stale one. The repo root carries a `COLCON_IGNORE` file so an accidental root build finds 0 packages instead of silently shadowing the real tree.
2. **Build with the venv's Python** (`../.venv/bin/python -m colcon build --symlink-install`). setuptools writes the shebang of every generated entry point (`install/astro_audio/lib/astro_audio/tts_node`, …) from the interpreter that runs the build. `/usr/bin/colcon` runs under the system Python, so the entry points get `#!/usr/bin/python3` — and then `ros2 run` cannot see a single venv package. The symptom is a launch full of lines like:

   ```text
   [tts_node] edge-tts paketi kurulu değil, pyttsx3'e düşürülüyor
   [speech_recognition_node] faster-whisper kütüphanesi kurulu değil! Vosk'a dönülüyor
   [audio_capture_node] sounddevice başlatılamadı. arecord fallback moduna geçiliyor...
   ```

   Those messages are lying: the packages are installed, just not visible to `/usr/bin/python3`. Building through the venv points the entry points at `.venv/bin/python`, and `ros2 run` / `ros2 launch` then work whether or not the venv is activated. (`colcon` stays importable inside the venv because it was created with `--system-site-packages`.)

To check which tree a running node came from, look at the path in the launch output: it must start with `<repo-root>/ros2_ws/install/`, not `<repo-root>/install/`. If your shell still has a deleted tree sourced, open a new shell and `source ros2_ws/install/setup.bash`.

Expected result: 7 packages finished. Warnings about `tests_require` and CMake policy `CMP0148` are harmless.

**Verify the build:**

```bash
ros2 pkg list | grep astro                 # 7 packages
ros2 pkg executables | grep astro          # 7 executables
```

Or just re-run `./scripts/install.sh --skip-apt`, which performs the same checks and prints a pass/fail line per node.

## 🚦 Running the robot

Every shell that talks to the robot needs both environments sourced — the venv (Python
packages) and the workspace overlay (ROS packages):

```bash
cd <repo-root>
source .venv/bin/activate
source ros2_ws/install/setup.bash      # zsh users: setup.zsh
```

### 1. Check the environment is clean

**Do this first.** ROS nodes are plain background processes: a leftover `tts_node` from an
earlier debugging session stays subscribed to `/tts/say` and speaks *every* sentence
alongside the new one. Two stray nodes means you hear the same reply three times, overlapping.

```bash
ros2 node list                                        # must be empty
ps -ef | grep -E "astro_(audio|ai|vision|lidar)/lib" | grep -v grep   # must be empty
```

If anything is listed:

```bash
ps -eo pid,cmd | grep -E "astro_(audio|ai|vision|lidar)/lib" | grep -v grep \
  | awk '{print $1}' | xargs -r kill -9
ros2 daemon stop && ros2 daemon start                 # clears stale discovery cache
```

> `ros2 node list` can keep showing nodes whose processes are already dead — that is the
> ROS daemon's cached discovery data, not a running node. Restarting the daemon clears it.

### 2. Launch

```bash
ros2 launch astro_bringup robot.launch.py
```

Starts base, LiDAR, camera, audio and the AI brain together. Missing hardware is handled
gracefully and the corresponding warnings are **expected**, not errors:

| Log line | Meaning |
|---|---|
| `LiDAR port '/dev/astro_lidar' not found — skipping` | No RPLIDAR; launch skips that node |
| `Arduino port not found (expected /dev/astro_arduino)` | No Arduino; retries every 2 s |
| `Cannot find any device with given deviceInfo` | No OAK-D camera attached |
| `ReSpeaker donanımı bulunamadı` | Falls back to the system default microphone |

### 3. Talk to it

Say the wake word (`WAKE_WORD`, default `hey astro`), then your sentence. The startup
banners tell you exactly which engines are live — they reflect real state, so trust them:

```text
✅ [STT] Hazır | zincir: openai/whisper-1 | STT_ENGINE=openai | ...
🚀 [TTS Node] Hazır | TTS_ENGINE=openai | zincir: openai_realtime -> openai_tts(gpt-4o-mini-tts) -> edge_tts -> espeak
```

The chain is a *fallback order*, not parallel execution: the first engine that returns
audio wins and the rest are never called.

### Shutting down

`Ctrl-C` in the launch terminal. Then re-run the check in step 1 — a node that fails to
exit cleanly is exactly what causes the overlapping-voices problem next time.

### Launching Individual Subsystems
If you want to test or launch sensors individually for debugging:

**Vision (camera + face recognition):**
```bash
ros2 launch astro_vision camera.launch.py                      # OAK-D driver
ros2 launch astro_vision camera.launch.py source:=webcam       # USB webcam instead
ros2 launch astro_vision camera.launch.py use_native_spatial:=true   # on-chip OAK-D pipeline
```

All sources publish to the same topic (`/oak/rgb/image_raw`), so the vision nodes do not
care which one is running. `source:=webcam` is what makes the stack testable on a laptop
with no OAK-D attached. `face_detector_node` publishes `/vision/recognized_person` with the
recognized person, plus `/vision/faces` (JSON with names, similarity and boxes).

**LiDAR (RPLIDAR + Filter):**
```bash
ros2 launch astro_lidar lidar.launch.py
```

**Audio (Mic Capture + STT + TTS):**
```bash
ros2 launch astro_audio audio.launch.py
```

## 🗺️ Simulation, mapping and navigation

A Gazebo Harmonic simulation of the robot, used to develop LiDAR mapping and
navigation without hardware. Packages: `astro_sim` (world, spawn, ros_gz bridge)
and `astro_navigation` (slam_toolbox, Nav2).

> **Full reference: [docs/simulasyon-ve-gercek-robot.md](docs/simulasyon-ve-gercek-robot.md)** —
> every parameter with its simulation and real-robot value, the calibration
> procedure, troubleshooting, and what is still missing before the stack can run
> on the physical robot.

### Prerequisites

```bash
sudo apt install -y ros-humble-slam-toolbox ros-humble-nav2-bringup \
  ros-humble-nav2-common ros-humble-robot-localization \
  ros-humble-joint-state-publisher ros-humble-twist-mux
```

Gazebo Harmonic (`gz sim`, package `ros-humble-ros-gzharmonic`) is the simulator.
Gazebo Classic and Ignition Fortress use different plugin filenames and will not
load this robot.

### 1. Run the simulation

```bash
ros2 launch astro_sim simulation.launch.py
ros2 launch astro_sim simulation.launch.py rviz:=false headless:=true   # no GUI
ros2 launch astro_sim simulation.launch.py x:=0 y:=-3                   # spawn elsewhere
```

The world is an indoor plan: a corridor, three rooms behind 0.9 m doorways, and
furniture that gives scan matching something to lock onto. The robot spawns in
the corridor at (-4, 0).

Simulation-only parts of the description live in `astro_gazebo.xacro` and are
included only under `sim_mode:=true`, so the real robot's tf tree is unchanged.
Sensors publish to the same topics as the hardware (`/scan`, `/imu`,
`/oak/rgb/image_raw`), which is what lets the same nodes run in both worlds.

### 2. Build a map

```bash
ros2 launch astro_navigation slam.launch.py                    # 2nd terminal
```

Drive from the **Teleop panel** in the Gazebo window — buttons and speed
sliders, no extra terminal. `ros2 run teleop_twist_keyboard teleop_twist_keyboard`
still works if you prefer keys.

Drive through every room, and return to where you started so slam_toolbox can
close the loop. Then save:

```bash
ros2 run nav2_map_server map_saver_cli \
  -f ros2_ws/src/astro_navigation/maps/astro_indoor --ros-args -p use_sim_time:=true
```

### 3. Navigate on the saved map

```bash
ros2 launch astro_navigation navigation.launch.py
```

AMCL starts at the map origin, which is the simulation spawn point. On the real
robot, or after moving the robot by hand, give it a starting estimate with
RViz's **2D Pose Estimate** before sending a goal — an unlocalised AMCL makes
the planner produce nonsense. Then use **Nav2 Goal**, or send one directly:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 6.0, y: 0.0}}}}"
```

### On the real robot

Both launch files take `use_sim_time:=false`. `astro_lidar`'s `scan_filter_node`
strips NaN and out-of-range returns and republishes on `/scan_filtered`; point
SLAM at it with `scan_topic:=/scan_filtered`. The simulated LiDAR is already
clean, so `/scan` is the default.

> **One odom publisher at a time.** In simulation the DiffDrive plugin publishes
> `/odom` and `odom -> base_footprint`; on the real robot `serial_bridge` does.
> Running both puts two publishers on the same transform and tf becomes
> unusable.

## 🧪 Tests & CI

```bash
source .venv/bin/activate
source ros2_ws/install/setup.bash
pytest                      # 292 tests
```

`pytest.ini` disables the two ROS plugins (`launch_testing`, `launch_ros`) whose old hook
signatures crash modern pytest — without it the run dies with
`PluginValidationError` before collecting anything.

`.github/workflows/ci.yml` runs the same thing on every push inside a `ros:humble-ros-base`
container: apt deps → uv venv → `scripts/build.sh` → `pytest` → `scripts/check_env_drift.py`.

## 🎙️ Realtime mode (OpenAI speech-to-speech)

A second, **alternative** pipeline sends microphone audio straight to the OpenAI Realtime
API over a WebSocket and plays the returned audio — no local STT or TTS in the loop:

```bash
ros2 launch astro_bringup realtime_sensors.launch.py
```

It starts `audio_stream_node` (24 kHz) + `astro_realtime_node` + the camera. It is **not**
an add-on to `robot.launch.py`: both publish `/speech/text` and both capture the
microphone, so running them together gives you two brains fighting over one robot. Pick one.

| | `robot.launch.py` | `realtime_sensors.launch.py` |
|---|---|---|
| Speech path | STT → LLM → TTS (separate calls) | single WebSocket, audio in / audio out |
| Latency | higher | lowest |
| Persona, long-term memory, reminders | `ai_brain_node` | `astro_realtime_node` (separate implementation) |
| Base, LiDAR | yes | no |

## ⚙️ Configuration

Two files, two jobs:

| File | Holds | Tracked in git? |
|---|---|---|
| `.env` | API keys, engine selection, thresholds | **no** (`.gitignore`) |
| `astro_bringup/config/astro_params.yaml` | ROS parameters: serial ports, camera FPS, VAD thresholds, LiDAR ranges | yes |

### The one rule about `.env`

**Define every key exactly once.** `python-dotenv` keeps the *last* definition of a
duplicated key, so a setting added at the top of the file is silently overwritten by a
template line further down. This is not hypothetical — it is why an "OpenAI everything"
block once sat in `.env` while the robot kept using edge-tts and a local Whisper model.
When changing a setting, **edit the existing line**; never append a second one.

`.env.example` is generated from the source with an AST scan, so every key in it is
actually read by the code, with its real default and the file that reads it. A CI check
keeps them in sync:

```bash
python scripts/check_env_drift.py
# ✅ Yapılandırma tutarlı — 101 anahtar, hepsi kodda okunuyor, tekrar yok.
```

It fails the build on three things: a key read by code but missing from `.env.example`,
a key documented but never read (dead setting), and any duplicate key.

### Getting started

```bash
cp .env.example .env
```

Then fill in **`OPENAI_API_KEY`** — that single key drives STT, LLM and TTS in the default
configuration. Everything else has a working default.

### Choosing engines

```ini
# LLM — built-in chain is groq -> gemini -> openai.
# Providers BEFORE the selected one are skipped; the ones after remain as fallback.
LLM_PROVIDER="openai"
LLM_FALLBACK_ENABLED="true"    # "false" -> only LLM_PROVIDER is ever tried

# STT — "openai" | "groq" | "faster-whisper"
# Any value other than "faster-whisper" prevents the local model from loading at all.
STT_ENGINE="openai"

# TTS — "openai" | "elevenlabs" | "xtts"
# Selects the primary engine. edge-tts and espeak stay as emergency fallbacks
# so the robot never goes mute on a network failure.
TTS_ENGINE="openai"
OPENAI_TTS_MODEL="gpt-4o-mini-tts"   # "tts-1" is noticeably faster, slightly lower quality
OPENAI_TTS_VOICE="echo"
```

> **Selecting a provider means emptying the others' keys is no longer required.**
> `LLM_PROVIDER` used to be documented but never read — the order was hard-coded and the
> only way to pick OpenAI was to blank out `GROQ_API_KEY` and `GEMINI_API_KEY`. It is now
> honoured for real.

> **`.env` overrides the process environment.** The nodes call `load_dotenv(override=True)`,
> so `TTS_ENGINE=xtts ros2 run ...` will *not* work — change the value in `.env` instead.

### Persistent memory

The robot's long-term memory (known people, learned facts, reminders, conversation
summaries) lives in `ros2_ws/astro_memory.json`. That file is **not tracked by git** — it
changes on every run and contains personal data. On first start it is seeded from the
tracked template `ros2_ws/astro_memory.seed.json`; point `MEMORY_FILE_PATH` elsewhere if
you want it outside the repo.

To reset the robot's memory, delete the runtime file — it will be re-seeded on next start:

```bash
rm ros2_ws/astro_memory.json
```

### 4. Advanced STT (Ses Tanıma) Options

`STT_ENGINE` in `.env` picks the primary engine; the router falls through
Groq → OpenAI → local Faster-Whisper, so a missing key never leaves the robot deaf.

**Faster-Whisper (local, no internet, no API key)**

```ini
STT_ENGINE="faster-whisper"
STT_FW_MODEL="large-v2"        # "turbo", "medium", "small" on weaker GPUs
STT_FW_DEVICE="cuda"           # falls back to CPU automatically if CUDA fails
STT_FW_COMPUTE_TYPE="float16"  # "int8" on CPU
STT_FW_CPU_COMPUTE_TYPE="int8"
STT_FW_CPU_MODEL="base"        # model used after a CPU fallback
```

> 🟩 **On Jetson (aarch64), `STT_FW_DEVICE="cuda"` silently falls back to CPU.** The
> `ctranslate2` aarch64 wheel on PyPI is built without CUDA, so Faster-Whisper cannot
> reach the GPU no matter how you configure it — you get the small `base` model on CPU,
> which is slow and hallucinates on silence. Fixing it means building CTranslate2 from
> source: see **[docs/jetson-cuda-stt.md](docs/jetson-cuda-stt.md)**.

> ⚠️ **Never use `distil-*` models for Turkish.** Every Distil-Whisper checkpoint is an
> English-only distillation — it ignores `language="tr"` and returns English text.

**Cloud Whisper (Groq / OpenAI)**

```ini
STT_ENGINE="groq"      # or "openai"
GROQ_API_KEY="gsk-..."
```

Groq's `whisper-large-v3` is the fastest of the three; it needs internet and a key.

**Who is speaking.** Every transcription also runs speaker recognition and publishes a
JSON payload (name, confidence, embedding) on `/audio/speaker_id`, alongside the plain
text on `/speech/text` — kept separate so wake-word matching is not disturbed. Thresholds
live in `SPEAKER_MATCH_THRESHOLD`; the model and profile paths in `SPEAKER_MODEL_DIR` /
`SPEAKER_DB_PATH`.

### 5. Advanced TTS (Ses Sentezi) Options

`TTS_ENGINE` in `.env` selects the **primary** engine — `openai` (default), `elevenlabs`
or `xtts`. It is not a full list of engines: whatever you pick, the router keeps
`edge-tts` and offline `espeak` at the end of the chain as emergency fallbacks, so the
robot never goes mute. Only `xtts` and `espeak` work without internet.

The chain in order: `openai_realtime` → *your engine* → `edge_tts` → `espeak` →
pre-generated emergency WAV. The startup banner prints the live chain.

> **`openai_realtime` only produces audio when `astro_realtime_node` is running**, which
> `robot.launch.py` does not start — see [Realtime mode](#-realtime-mode-openai-speech-to-speech).
> In the classic pipeline the realtime step is a no-op and `TTS_ENGINE` decides what you hear.

**XTTS v2 — local voice cloning (offline, GPU recommended)**

`xtts` speaks with a cloned voice taken from a short reference recording, with no internet and no API key. It is **not** installed from PyPI: `pip install TTS` resolves Coqui TTS 0.22.0 against today's package versions and produces an environment that imports but does not work. Instead a dedicated script clones the maintained fork and pins the version set that is known to work:

```bash
./scripts/install_xtts.sh
```

The script clones <https://github.com/yunusemretom/TTS.git> into `~/.astro/tts` (override with `TTS_XTTS_HOME`), creates a Python 3.10 venv there, installs the repo with `uv pip install -e .`, picks the torch CUDA build matching your driver, pins `librosa`/`transformers`/`numpy`/`scipy`, symlinks `espeak` → `espeak-ng`, and pre-downloads the ~1.8 GB XTTS v2 checkpoint (skip with `XTTS_SKIP_DOWNLOAD=1`). It is re-runnable: an existing clone is updated instead of re-cloned.

> **Why a second virtualenv?** XTTS needs `numpy==1.26.4` — its compiled `monotonic_align` Cython extension is built against the NumPy 1.x ABI — while this repo pins `numpy==2.2.6` to match `rclpy`. The two cannot share an interpreter. So `tts_node` never imports XTTS: it launches `xtts_worker.py` with the XTTS venv's Python as a long-lived child process and talks to it over line-based JSON on stdin/stdout (`xtts_client.py`). The model and speaker latents are loaded once at node start, so each sentence pays only inference cost.

Enable it in `.env`:

```ini
TTS_ENGINE="xtts"
TTS_XTTS_HOME="$HOME/.astro/tts"
TTS_XTTS_SPEAKER_WAV=""      # empty → packaged voices/astro.wav
TTS_XTTS_DEVICE="auto"       # "auto" | "cuda" | "cpu"
TTS_XTTS_HALF=1              # fp16, CUDA only
TTS_XTTS_BATCH_SIZE=4        # batch sentence decoding on long text
```

**Changing the robot's voice.** The reference clip ships with the package at `ros2_ws/src/astro_audio/voices/astro.wav` (~9 s). Replace that file, or point `TTS_XTTS_SPEAKER_WAV` at an absolute path. Use 6–30 s of clean, single-speaker audio.

**Using your own fine-tuned XTTS model.** If you trained or downloaded an XTTS checkpoint, point the node at it and the stock `xtts_v2` is never downloaded:

```ini
TTS_XTTS_MODEL_DIR="/home/user/Downloads/optimized_model"
```

The directory is expected to hold `model.pth`, `config.json`, `vocab.json` and — optionally — `speakers_xtts.pth`. A missing `speakers_xtts.pth` is fine: it only carries the built-in speaker table, and reference-clip cloning does not use it. If your files sit elsewhere or have different names, set them one by one; these override anything derived from the directory:

```ini
TTS_XTTS_CHECKPOINT="/path/to/model.pth"
TTS_XTTS_CONFIG="/path/to/config.json"
TTS_XTTS_VOCAB="/path/to/vocab.json"
TTS_XTTS_SPEAKERS="/path/to/speakers_xtts.pth"
```

Paths are checked before the worker starts, so a typo is reported as `Özel XTTS modeli dosyası bulunamadı: <path>` and the node falls back to edge-tts instead of hanging. The startup log tells you which model is live — `kendi modeliniz` versus `hazır xtts_v2`. Voice cloning, fp16 and batching work identically on a custom checkpoint.

**Behaviour and expectations.**
- Startup takes ~10–30 s (model load + warm-up) and much longer on the first run if the checkpoint still has to download. The node does not block: sentences arriving before XTTS is ready are spoken by `edge-tts` instead, and `✅ [TTS] XTTS hazır` is logged when the worker is warm.
- If the install is missing, the worker fails to start, or it dies mid-run, `tts_node` logs the reason and falls back to `edge-tts` (or `pyttsx3` when there is no internet package) — the robot never goes silent.
- XTTS emits WAV, so playback uses `paplay`/`aplay`/`ffplay`, not `mpg123`. Install `alsa-utils` if none is present.
- On CPU XTTS is very slow (RTF > 1, i.e. slower than real time). On an RTX 4050 Laptop with fp16 + batch=4 a paragraph runs at RTF ≈ 0.09 using ~1.5 GB VRAM.

> **Note:** For AI API keys (`AI_API_KEY`), use the `.env` file at the root of the project (copy from `.env.example`). Do not hardcode API keys in the source code!

---

## 🧠 How it works — the design decision that shaped everything

The robot depends on six things that can each fail independently: an LLM, speech
recognition, speech synthesis, a depth camera, a LiDAR and a serial link to the
motor board. Early versions treated each as required. In practice that meant one
expired API key made the robot mute, or a `pip` conflict in a Whisper dependency
made it deaf, and every failure looked the same from outside: a robot that sits
there.

So every engine is selected in `.env` and **every engine has a fallback chain**
rather than a hard dependency:

| Job | Primary | Falls back to |
|---|---|---|
| LLM | Groq, Gemini or OpenAI by config | the remaining providers in the chain |
| Speech to text | OpenAI or Groq Whisper | local Faster-Whisper |
| Speech synthesis | XTTS, OpenAI or ElevenLabs | `edge-tts`, then `espeak` |

Two consequences are worth calling out because they were not obvious up front.

**Selection has to skip, not reorder.** Providers *before* the selected one in
the chain are skipped and the ones after remain as fallback. Choosing a provider
therefore expresses a preference, not an exclusive choice, and there is no
configuration that leaves the robot with nothing to fall back on.

**Loading has to be gated on selection.** Setting speech recognition to anything
other than `faster-whisper` prevents the local model from loading at all. Before
that, the local model was loaded eagerly "just in case" and cost gigabytes of
VRAM on a machine that was never going to use it.

The synthesis path shows the pattern end to end: XTTS takes 10 to 30 seconds to
warm up, so sentences arriving before it is ready are spoken by `edge-tts` and
the switchover is logged. The robot answers immediately and its voice improves
mid-conversation, rather than staying silent until the good model is ready.

A configuration checker validates that all 101 keys are read somewhere in the
code and that none are duplicated, because a silently-unused config key is a
setting you believe you changed.

---

## ⚠️ Known limitations

- Tuned for one physical build: an Arduino Mega base, RPLIDAR A1, OAK-D Lite and
  a ReSpeaker 4-mic array. Pin maps and frame offsets are specific to it.
- Navigation is validated in Gazebo and in one mapped indoor space. It has no
  dynamic-obstacle or crowd handling.
- Face and speaker recognition store personal biometric data. `faces/`,
  `Persons/` and the memory files are gitignored for that reason and must not be
  committed.
- Speech recognition accuracy in Turkish drops noticeably with background noise
  and with more than one speaker.
- Local XTTS on CPU is slower than real time. It needs a GPU to be usable.
- The conversation state machine is rule-based, so it handles interruptions and
  topic changes poorly.
- **Repository history.** The project's history descends from `ros2/rosidl`, so
  the commit log contains several hundred unrelated upstream ROS commits
  alongside the ASTRO work.

## 🗺️ Roadmap

- Dynamic obstacle avoidance and recovery behaviors in Nav2.
- Speaker diarization so multi-person conversation is tracked properly.
- Move the conversation manager off hand-written rules.
- Onboard deployment on Jetson with measured end-to-end latency.
- Continue upstream in [BarlineTR/astr1](https://github.com/BarlineTR/astr1),
  which is where development happens.

---

## 👤 My contributions

For anyone evaluating my work specifically, these are the parts of ASTRO I
wrote. Everything else is the upstream author's.

**Test suite.** Essentially all of it, across four packages: detection quality
and backend tests for vision, detection-hold behaviour, image payload handling,
speech detection, ReSpeaker idle sector logic, the direction-of-arrival yaw
convention, session recording, and the `conftest.py` fixtures the rest depend
on. The project had no automated tests before this.

**Navigation and simulation.** The `astro_navigation` package: `nav2_params.yaml`
(the largest single file I own here), the SLAM Toolbox configuration and both
launch files. Plus the `astro_sim` Gazebo launch and its EKF configuration, which
is what makes it possible to work on navigation without the physical robot.

**Audio.** The OpenAI TTS engine and the speaker-recognition database, plus the
ReSpeaker USB capture layer and its sector mapping. I also introduced the local
XTTS integration (`xtts_client.py`, `xtts_worker.py`), though those two files
have since been substantially rewritten upstream and are now mostly not mine.

**Firmware and tooling.** The Arduino base firmware for the motor and encoder
layer, `MotorTest` almost entirely and `AstroFirmware` in large part, plus face
enrolment and session recording.

The theme is the unglamorous half: tests, Nav2 tuning, simulation setup and
hardware integration. Those are the parts I would point at in an interview,
because tuning a Nav2 stack until it stops oscillating and writing the tests
that catch a regression in face-detection hold are where I learned the most.
