Imports:

The necessary libraries are imported:
cv2: OpenCV for computer vision tasks.
mediapipe as mp: MediaPipe library for hand tracking.
Various other libraries for tasks like math operations, keyboard input, time management, socket communication, GUI automation, threading, and data manipulation.
Configuration:

Several variables and constants are initialized, including flags, counters, and parameters for the hand tracking process.
Initialization:

MediaPipe components are set up (mp_hands and mp_drawing).
The camera stream is accessed using cv2.VideoCapture().
Resolution and frame rate settings are configured for the camera.
Mouse Simulation and Socket Functions:

There are functions mouse_sim() and socket_connect() defined.
mouse_sim() moves the mouse cursor using pyautogui library based on the detected hand position.
socket_connect() establishes a connection to a remote host using sockets.
Main Loop:

The program enters a loop that captures frames from the camera stream.
Frame Processing:

Each captured frame is converted to RGB format.
The hand landmarks are processed using MediaPipe, and information about the detected hands is extracted.
Hand Landmark Processing:

The script extracts various landmark points from the hand.
It calculates distances, angles, and other parameters based on these landmarks.
Hand Gesture Interpretation:

The code interprets the hand landmarks to recognize gestures.
It computes completion rates, identifies left or right hand, and determines clockwise/counterclockwise directions.
User Interface:

Information and statistics about the hand tracking process are displayed on the video feed using cv2.putText().
Thread Execution:

The code spawns separate threads (mouse_thread and thread2) for handling mouse movement and socket communication, respectively.
Input Handling and User Interaction:

The script checks for user input. If 'q' is pressed, the program gracefully exits, releasing resources.
Displaying the Video Feed:

The modified frame with overlays and text is displayed in a window using cv2.imshow().
Program Termination:

The program gracefully closes and releases resources when the user closes the window.
This code demonstrates a moderate to advanced level of proficiency in Python, particularly in areas related to computer vision, gesture recognition, socket communication, and multithreading. The author also shows competence in using third-party libraries like OpenCV, MediaPipe, and pyautogui.



**The algorithmic part of this code**

Hand Landmark Extraction:

The results variable contains information about detected hands, including their landmarks.
The script accesses the landmarks' coordinates to perform various calculations.
Distance Calculation:

The distance between the index finger MCP (Metacarpophalangeal) joint and the ring finger MCP joint is computed. This represents the distance between the index and ring fingers.
Stability Measurement:

The stability of the hand is calculated by finding the distance between the index finger MCP joint and the wrist.
Gesture Labeling:

The script identifies whether the detected hand is left or right based on the wrist and pinky positions.
Completion Rate Computation:

The completion rate is calculated based on the ratio of the distance between the index and ring fingers to the hand stability.
Completion Rate Adjustment:

The completion rate is adjusted based on various conditions (e.g., if it's too high or too low).
Maximum Completion Rate Tracking:

The code keeps track of the maximum completion rate achieved during the session.
Reset Detection:

A reset condition is checked based on the completion rate. If the rate is close to 2.5, it's considered a reset.
Mouse Position Estimation:

The final position for the mouse cursor is calculated based on the positions of hand landmarks.
Mouse Movement:

The mouse is moved using pyautogui to simulate mouse movement. The mouse position is updated if it has changed.
Socket Connection:

The program establishes a socket connection with a remote host (presumably for some form of communication).
Thread Handling:

Threads are used to handle tasks concurrently, such as mouse movement and socket communication.
User Interface Feedback:

The completion rate, stability, frame rate, and other information are displayed on the video feed for user feedback.


这是一个在python环境中，主要基于mediapipe和cv2库实现的判断手掌活动的程序。
This is a python environment, mainly based on the mediapipe and cv2 library implementation of the determination of palm activity program.
导入所需的库和模块：
基于mediapipe库，你也可以在此项目的基础上进行修改，以便进行躯干或者面部的判断。

```import cv2
import mediapipe as mp
import math
import keyboard
import time
import pyautogui
import threading
import sys
import socket
import pickle
import struct
import base64
import numpy as np
```

这段代码导入了一系列用于图像处理、手势识别、键盘控制、时间操作、鼠标模拟、多线程、网络通信等功能的库和模块。

初始化MediaPipe和OpenCV相关的对象和变量：
```
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Initialize mediapipe hands
hands = mp_hands.Hands(max_num_hands=1)
# Get the camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
```
这些代码初始化了MediaPipe手势识别的绘图工具、手势识别模型以及OpenCV的摄像头对象。

建立与Unity程序的Socket连接：

```
HOST = 'localhost'  # unity程序的IP地址
PORT = 10888
def socket_connect():
    global reset, maxrateaft80
    try:
        ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ss.connect((HOST, PORT))
        s = "{} {:.2f}".format(str(reset), float(maxrateaft80))
        ss.sendall(s.encode())
    except Exception as e:
        print(e)
```
这部分代码定义了一个socket_connect()函数，用于与Unity程序建立Socket连接，并将reset和maxrateaft80两个变量的值发送给Unity程序。

打开摄像头：

```
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FPS, 120)
```
这部分代码设置摄像头的分辨率和帧率,你可以根据自己需要进行调整。

进入循环，读取视频流：

```
while cap.isOpened():
    isput = False
    # Read the video stream
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
    # Convert the image to RGB format
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    num_hands = 0 if results.multi_hand_landmarks is None else len(results.multi_hand_landmarks)
```
这部分代码使用cap.read()方法读取摄像头的视频流，并进行一些预处理操作。

对每一帧图像进行处理：
```
if results.multi_hand_landmarks:
    palm_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in results.multi_hand_landmarks[0].landmark])
else:
    palm_landmarks = None
if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
        landmarks = handLms.landmark
```
这部分代码对每一帧图像进行手势识别操作。首先，提取出手部关键点的坐标信息。然后，遍历检测到的每一只手，获取手的关键点和其他相关信息。
```
def cursor_positions():
    ...
```
这部分代码定义了一个cursor_positions()函数，用于计算鼠标的移动位置。

```
cv2.imshow('MediaPipe Hands', image)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```
这部分代码显示处理后的图像，并监听键盘按键。如果按下了键盘上的'q'键，就退出程序。

通过键盘按键控制程序退出。

```
if keyboard.is_pressed('q'):
    cv2.destroyAllWindows()
    cap.release()
    break
```
这部分代码通过检测键盘按键来控制程序的退出。如果按下了键盘上的'q'键，就关闭窗口和摄像头，并跳出循环。

关闭摄像头：

```
cap.release()
```
这部分代码释放摄像头资源。

关闭窗口：

```
cv2.destroyAllWindows()
```
这部分代码关闭显示图像的窗口。

总体来说，我利用MediaPipe和OpenCV实现了手势识别，并将识别结果通过Socket连接发送给Unity程序进行交互。

代码的主要逻辑是读取摄像头的视频流，使用MediaPipe进行手势识别，计算鼠标移动的位置，将识别结果发送给Unity程序，并通过键盘按键来控制程序的退出。

2023年7月3日14点23分by Steven.Z
