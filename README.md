# handrecognition
这是一个在python环境中，主要基于mediapipe和cv2库实现的判断手掌活动的程序。
This is a python environment, mainly based on the mediapipe and cv2 library implementation of the determination of palm activity program.
主要功能是使用MediaPipe和OpenCV实现手势识别，并将识别结果通过Socket连接发送给Unity程序进行交互。
代码的执行逻辑如下：
导入所需的库和模块。
初始化MediaPipe和OpenCV相关的对象和变量。
建立与Unity程序的Socket连接。
打开摄像头。
进入循环，读取视频流。
对每一帧图像进行处理：
a. 将图像转换为RGB格式。
b. 使用MediaPipe进行手部关键点检测。
c. 获取检测到的手的关键点和其他相关信息。
d. 根据手的关键点位置判断手的左右方向。
e. 根据手指位置计算手指间距和稳定性。
f. 根据计算结果确定鼠标的移动位置。
g. 发送识别结果和相关数据给Unity程序。
h. 在图像上绘制识别结果和其他信息。
i. 显示图像。
通过键盘按键控制程序退出。
关闭摄像头。
关闭窗口。
总体来说，代码的逻辑是通过MediaPipe检测手的关键点并计算相关指标，根据指标来控制鼠标移动，并将识别结果通过socket发送给Unity程序进行处理（你也可以向其他程序传输）
