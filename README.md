
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
