<!-- .github/copilot-instructions.md -->
# One-Gesture — Copilot instructions for code edits

Purpose: give an AI coding agent the minimal, concrete knowledge needed to make safe, correct changes and add features in this repository.

- **Entry points & how to run**
  - Prototype app: run `python main.py` (Qt overlay + vision thread). On Windows the camera is opened with `cv2.VideoCapture(0, cv2.CAP_DSHOW)`.
  - Debug dashboard: `streamlit run ui/streamlit_app.py` (streamlit prototype; may be incomplete).
  - Install dependencies: `pip install -r requirements.txt` (project pins specific versions).

- **Big picture / architecture**
  - `main.py` launches a Qt application in the main thread (required) and starts a background vision thread (`vision_loop`) that captures from the webcam, runs detection, computes features and gestures, and writes minimal shared state used by the Qt overlay.
  - Vision stack: `core/detection.py` (MediaPipe wrapper) → `core/features.py` (relative landmarks, angles, smoothing) → `core/gestures.py` (static + dynamic gesture logic) → optional `core/actions.py` for OS automation.
  - UI: `ui/air_pointer.py` is a transparent PyQt widget used as an always-on-top pointer overlay. Any change to the overlay must keep Qt running in main thread and preserve the window flags used (`WindowStaysOnTopHint`, `WindowTransparentForInput`, `FramelessWindowHint`).

- **Concurrency & shared state rules**
  - Qt (AirPointer) MUST be started in the process main thread (`QApplication` created in `main()`); do not instantiate Qt widgets from worker threads.
  - Vision code runs in a background Python thread and writes a very small set of globals used by the overlay: `pointer_target_x`, `pointer_target_y`, and debug variables like `last_static_gesture`. Use these globals when adding simple integrations — avoid adding heavy state coupling.

- **Project-specific patterns to preserve / emulate**
  - Landmark normalization: `FeatureExtractor.landmark_to_relative` normalizes by wrist→middle-fingertip distance. New detectors must output landmarks in the same format or wrap conversions into `core/features.py`.
  - Finger state logic: `finger_states` uses 3-point angles and a hard threshold `2.2` radians (≈126°) for extended/closed decisions. This threshold is used elsewhere (gesture semantics rely on it) — if you change it, update tests/threshold comments.
  - Smoothing: two levels — `Smoother` (moving average on landmark coordinates) and `TemporalFingerStabilizer` (majority voting with >0.6 cutoff). The `GestureEngine.filter_static` uses a >70% dominance rule over the last 10–12 frames.
  - Dynamic gestures: centroid-history based swipe detection with tuned thresholds `DISP_THRESH=0.020` and `VEL_THRESH=0.012` in `core/gestures.py`. These constants were tuned for webcam/CPU; adjust cautiously and document new values.
  - Pointer mapping & movement: `core/actions.py::move_mouse_rel` maps relative coords `[-0.7, 0.7]` x and `[-0.5, 0.5]` y to screen space and smooths with easing factor `0.25` when `smooth=True`.

- **Where to change detection backend**
  - Replace `core/detection.py` implementation with an adapter that exposes the same API: `HandDetector.detect(frame)` → returns an object with `.multi_hand_landmarks` and `hand.landmark` entries compatible with the existing feature extractor. Keep `draw(frame, results)` semantics.

- **Testing & debugging tips**
  - Quick runtime checks: run `python main.py`; press `ESC` in the OpenCV window to stop the vision loop cleanly.
  - On Windows, camera access commonly needs `cv2.CAP_DSHOW` (already used). If camera fails, inspect `cap.isOpened()` error path.
  - For visual verification of changes to gesture logic, use the HUD drawn in `main.py::draw_hud` (shows FPS, static/dynamic gestures, recent actions).

- **Files of highest importance**
  - `main.py` — entrypoint, thread orchestration, virtual desktop mapping
  - `core/detection.py` — detection adapter (MediaPipe by default)
  - `core/features.py` — landmark normalization, angle math, smoothing
  - `core/gestures.py` — gesture rules & thresholds (static + dynamic)
  - `core/actions.py` — OS-level effects (pynput / mouse+keyboard)
  - `ui/air_pointer.py` — Qt overlay widget (must run in main thread)

- **Common pitfalls for automated edits**
  - Do not move Qt initialization into a background thread — tests and runtime will fail silently or crash.
  - If changing landmark formats, keep backward-compatible adapters so `FeatureExtractor` continues to receive wrist-indexed landmarks.
  - Avoid introducing heavy CPU work into the main Qt event loop — keep vision and processing in the worker thread and pass only small, serializable state to UI.

If any section above is unclear or you'd like the agent to expand examples (unit tests, a small adapter for a different detector, or CI checks), tell me which part to expand and I'll iterate.
