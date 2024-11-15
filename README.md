
# Parkinson Hand Recognizing Program V3.1.1

This program is a Python-based hand recognition system designed to track hand gestures, calculate metrics like completion rate and hand stability, and interact with external applications using sockets and mouse simulations. It leverages **MediaPipe Hands**, **OpenCV**, and **PyAutoGUI** to provide real-time feedback and control.

---

## Features

- **Hand Gesture Recognition**: Detects and tracks hand gestures using MediaPipe.
- **Real-Time Metrics**: Calculates completion rate, hand stability, and hand flip direction.
- **External Integration**: Sends data to external programs via socket communication.
- **Mouse Simulation**: Controls mouse cursor based on hand position.
- **High Frame Rate Support**: Configured for 120 FPS for smooth tracking.

---

## Prerequisites

Before running the program, ensure you have the following installed:

- Python 3.8+
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy
- Keyboard

Install the dependencies with:
```bash
pip install opencv-python mediapipe pyautogui numpy
```

---

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/Parkinson-Hand-Recognizing.git
cd Parkinson-Hand-Recognizing
```

### 2. Run the Program
Start the program with:
```bash
python hand_recognition.py
```

### 3. Control and Metrics
- **Place your palm in view of the camera** for the program to start recognizing gestures.
- Metrics such as `Completion Rate`, `Flip Direction`, and `FPS` will be displayed on the screen.
- Press **'Q'** to quit the program.

---

## Customization

### Configure External Socket Communication
- Update the `HOST` and `PORT` variables in the code to match your external application:
```python
HOST = 'localhost'  # Replace with the external program's IP address
PORT = 10888  # Replace with the desired port number
```

### Adjust Camera Settings
You can modify the camera resolution and frame rate:
```python
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FPS, 120)
```

### Debugging
To debug connection issues or errors, monitor the terminal for exceptions or print statements.

---

## Key Features in Detail

### Hand Gesture Detection
- Uses MediaPipe Hands to extract key hand landmarks in real time.
- Tracks critical points like wrist, thumb tip, and index finger MCP for calculating distances and stability.

### Mouse Simulation
- Calculates the center position of the hand and moves the mouse cursor accordingly.
- Smooth and responsive control using threading.

### Completion Rate and Stability
- Dynamically calculates a "completion rate" based on the relative positions of hand landmarks.
- Stabilizes the rate to avoid false spikes or noise.

### Socket Communication
- Sends real-time metrics to external applications using TCP sockets.

---

## Troubleshooting

- **Camera Not Detected**: Ensure your camera is connected and accessible. Check the `cv2.VideoCapture` settings.
- **Low FPS**: Reduce resolution or adjust `cap.set(cv2.CAP_PROP_FPS)` for better performance on lower-end hardware.
- **Dependencies Missing**: Use `pip install -r requirements.txt` to install all dependencies at once.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact

For questions or support, contact [Steven-ZN](mailto:stevenhashiru@gmail.com) or open an issue on GitHub.
```

This README provides a clear overview of your project's functionality, setup, and usage while offering room for customization and troubleshooting. Let me know if you want further tweaks!
