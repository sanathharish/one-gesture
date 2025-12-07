🤝 One-Gesture — Real-Time Hand Tracking & OS Interaction

A modular, extensible Python-based system for real-time hand- and finger-gesture control, featuring live visualization, gesture training, smoothing algorithms, and full OS automation support.
Designed for research, prototyping, accessibility, robotics, AR/VR, and desktop interaction.

🚀 Features

🎥 Real-time webcam-based hand tracking

🏷 Gesture classification (static + dynamic patterns)

🖱 OS control actions (mouse, keyboard, scroll, drag, hotkeys)

🧠 Built-in gesture training studio (record, label, save, re-train)

📊 Debug dashboard with FPS, gesture history, confidence, raw values

🔧 Modular design (replace MediaPipe with YOLO, TF Lite, OpenVINO, etc.)

🛑 Safety controls: kill-gesture, calibration lock, manual override

💻 Streamlit prototype UI → Flutter production UI pipeline

📦 Tech Stack
Component	Technology
Hand Detection	MediaPipe Hands (default), Ultralytics (optional)
Interface	Streamlit (debug), Flutter Desktop/Web (production)
Computer Vision	OpenCV
OS Automation	PyAutoGUI / pynput
ML Training	Scikit-Learn / TensorFlow Lite
Performance	Async loops, temporal smoothing (EMA/Kalman)

📂 Project Structure
one-gesture/
 ├─ main.py                    # Entry point
 ├─ requirements.txt
 ├─ README.md
 │
 ├─ core/                     
 │   ├─ detection.py           # Hand tracking + landmarks
 │   ├─ calibration.py         # Screen mapping + scaling
 │   ├─ smoothing.py           # Filtering + stabilization
 │   └─ gestures.py            # Rule-based gesture definitions
 │
 ├─ os_actions/
 │   ├─ mouse.py               # Cursor, click, drag
 │   └─ keyboard.py            # Typing, hotkeys
 │
 ├─ training/
 │   ├─ recorder.py            # Collect labeled gesture data
 │   ├─ trainer.py             # Train ML classifier
 │   └─ models.pkl             # Saved user-defined gesture profiles
 │
 ├─ ui/
 │   ├─ streamlit_app.py       # Debug + experiment dashboard
 │   └─ flutter/               # Final production UI (later stage)
 │
 └─ utils/
     └─ logger.py              # Action history + performance logs

🛠 Installation
git clone https://github.com/sanathharish/one-gesture
cd one-gesture
pip install -r requirements.txt

▶️ Run the Prototype
python main.py


or to run the Debug Dashboard:

streamlit run ui/streamlit_app.py

🧠 Training New Gestures (Optional)
python training/recorder.py
python training/trainer.py

🧰 Use Cases

Assistive technology (hands-free computing)

Human-computer interaction research

Robotics control

XR/Metaverse gesture interfaces

Gaming or music performance using gestures

📄 License

MIT License — free to use, modify, and contribute.

📚 References

MediaPipe Hands Docs

OpenCV Python

Streamlit Docs

PyAutoGUI

Ultralytics YOLO Keypoint Models
