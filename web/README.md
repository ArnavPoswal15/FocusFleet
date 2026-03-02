---
title: FocusFleet
emoji: 🌖
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.8.0
python_version: 3.10.13
app_file: app.py
pinned: false
license: mit
short_description: 'AI-Powered Driver Drowsiness Detection SAAS'
---

# 🌖 FocusFleet: Driver Drowsiness Detection

FocusFleet is an AI-powered safety solution that monitors driver alertness in real-time. By analyzing facial landmarks, the system identifies signs of fatigue and provides immediate feedback to prevent accidents.

##🚀 Features

- **Real-time Live Monitoring**: Process camera feeds directly in your browser.
- **Precision Detection**: Uses High-fidelity MediaPipe Face Mesh for EAR (Eye Aspect Ratio) monitoring.
- **Yawn Detection**: Monitors mouth patterns to identify early signs of drowsiness.
- **Cloud-Native**: Optimized for Hugging Face Spaces with Python 3.10 stability.

## 🛠️ How it Works

1. **Access Camera**: Grant camera permissions in your browser.
2. **Login/Register**: Create an account to start a monitoring session.
3. **Set Thresholds**: Adjust the EAR threshold based on your environment or eyelid shape.
4. **Monitor**: The system will alert you visually and log the session data.

## 📦 Tech Stack

- **Gradio** for the web interface.
- **TensorFlow/Keras** for drowsiness classification.
- **MediaPipe** for facial landmark extraction.
- **OpenCV** for image processing.

---
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
