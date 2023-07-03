import cv2
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

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Initialize mediapipe hands
hands = mp_hands.Hands(max_num_hands=1)
# Get the camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
temp = 0
lasttemp = 2
pTime = 0
cTime = 0
isclockwise = True
centerpos_y  = 0
centerpos_x = 0
final_x = 0
final_y = 0
dTI = 0
dTM = 0
distance = 1
tempdistance = 0
filptime = 0
thumb_x = 0
thumb_y = 0
index_x = 0
index_y = 0
completion = 0
initialdistance = 0
stable = 1
ful = 1
HOST = 'localhost'  # unity程序的IP地址
PORT = 10888
lastx = 0
lasty = 0
filp_num = 0
lastcompletion = 10
maxrateaft80 = 80
reset = False
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m', 'j', 'p', 'g'))


cap.set(cv2.CAP_PROP_FPS, 120)

def mouse_sim():
    global final_x,final_y,lastx,lasty
    if(final_y != lasty and final_x != lastx):
        pyautogui.moveTo(final_x, final_y)
    lastx = final_x
    lasty = final_y

def socket_connect():
    global reset, maxrateaft80
    try:
        ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ss.connect((HOST, PORT))
        s = "{} {:.2f}".format(str(reset), float(maxrateaft80))
        ss.sendall(s.encode())
    except Exception as e:
        print(e)


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
    if results.multi_hand_landmarks:
        palm_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in results.multi_hand_landmarks[0].landmark])
    else:
        palm_landmarks = None
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmarks = handLms.landmark
            index_y = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
            pinky_x = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].x
            index_x = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
            pinky_y = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y
            distance = int((((index_x - pinky_x) ** 2 + (index_y - pinky_y) ** 2) ** 0.5)*100)
            wrist_x = landmarks[mp_hands.HandLandmark.WRIST].x
            wrist_y = landmarks[mp_hands.HandLandmark.WRIST].y
            thumb_x = landmarks[mp_hands.HandLandmark.THUMB_TIP].x
            thumb_y = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
            pinky_x1 = landmarks[mp_hands.HandLandmark.PINKY_TIP].x
            pinky_y1 = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
            stable = int((((index_x - wrist_x) ** 2 + (index_y - wrist_y) ** 2) ** 0.5) * 100)
            for hand_label in results.multi_handedness:
                # 获取label和score
                label = hand_label.classification[0].label
                cv2.putText(image, f''' {label}''', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

            completion = ((((distance / stable+0.01)+1)*7.52)-8)*5

            completion = completion * 10

            if(label =="Left"):
                if(wrist_x > pinky_x):
                    completion*=-1
            elif(label =="Right"):
                if(wrist_x < pinky_x):
                    completion*=-1
            completion = ((completion/2 +50)+2)/1.25
            #print((((thumb_x - pinky_x1) ** 2 + (thumb_y - pinky_y1) ** 2) ** 0.5)*100)
            if (((thumb_x - pinky_x1) ** 2 + (thumb_y - pinky_y1) ** 2) ** 0.5)*100<=2:
                completion = 50
            if completion > 100:
                completion = 100
            elif completion <0:
                completion = 0
            elif abs(completion-10)<3:
                completion = 10
            elif abs(completion-0)<3:
                completion = 0
            if abs(completion-100)<6:
                completion = 100
            if lastcompletion<80 and completion>80:
                filp_num +=1
                maxrateaft80 = 80
            if completion > maxrateaft80:
                maxrateaft80 = completion
            if maxrateaft80 != 100 and maxrateaft80 !=80:
                maxrateaft80 = "{:.2f}".format(maxrateaft80)
                maxrateaft80 = float(maxrateaft80)

            if abs(2.5-completion) <=2.5:
                reset = True
            else:
                reset = False
                            #print(filp_num)
            cv2.putText(image, f'''MaxCRate: {'' if maxrateaft80 == 80  else maxrateaft80}%''', (260, 330), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            lastcompletion = completion
            completion = "{:.2f}".format(completion)
            completion = float(completion)
            comp = ((landmarks[mp_hands.HandLandmark.THUMB_TIP].x-landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)**2+(landmarks[mp_hands.HandLandmark.THUMB_TIP].y-landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)**2)*1000
            centerpos_y  = (landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y + landmarks[mp_hands.HandLandmark.WRIST].y)/2
            centerpos_x = (landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x + landmarks[mp_hands.HandLandmark.WRIST].x)/2
            final_x = ((centerpos_x + landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].x)/2)*1920
            final_y = ((centerpos_y + landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y)/2)*1080
            mouse_thread = threading.Thread(target=mouse_sim)
            mouse_thread.start()
            isput = True
    if num_hands == 1:
        # 获取手的关键点
        hand_landmarks = results.multi_hand_landmarks[0]
        # 获取手腕和小指指尖的位置
        wrist_x, wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks.landmark[
            mp_hands.HandLandmark.WRIST].y
        pinky_x, pinky_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x, hand_landmarks.landmark[
            mp_hands.HandLandmark.PINKY_TIP].y
        # 判断手的左右位置
        if label == "Right":
            if wrist_x < pinky_x:
                temp = 1
                if lasttemp == 0:
                    keyboard.press_and_release("2")
                    isclockwise = False
            else:
                temp = 0
                if lasttemp == 1:
                    keyboard.press_and_release("1")
                    isclockwise = True
            lasttemp = temp
        else:
            if wrist_x < pinky_x:
                # print("1")
                temp = 1
                if lasttemp == 0:
                    keyboard.press_and_release("2")
                    isclockwise = False
            else:
                temp = 0
                if lasttemp == 1:
                    keyboard.press_and_release("1")
                    isclockwise = True
            lasttemp = temp
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    #if(completion = 100 and  ):
       # d
    thread2 = threading.Thread(target=socket_connect)
    thread2.start()

    cv2.putText(image, f'''Filptime: {filp_num}''', (260, 430), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
    cv2.putText(image, f'''CompletionRate: {completion}%''', (240, 460), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
    cv2.putText(image, f'''FPS: {int(fps)}''', (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv2.putText(image, f'''Handflipdirection: {'clockwise' if isclockwise else 'counterclockwise'}''', (85, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv2.imshow('Parkinson Hand_Recognizing_Program V3.1.1 Designed by Steven.Z', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not isput:
        cv2.putText(image, '     PLEASE PLACE YOUR PALM ', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        break
    if cv2.getWindowProperty('Parkinson Hand_Recognizing_Program V3.1.1 Designed by Steven.Z', cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()

cv2.destroyAllWindows()



