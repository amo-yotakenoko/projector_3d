import cv2
import numpy as np
import math
from pathlib import Path
import shutil, os
import queue
import time
from  settings import *



def phase_shifting_capture_loop(cap_frame_queue,  loop_max=-1):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


        

    
    # Wait for camera to stabilize
    for _ in range(5):
        cap.read()

    phase_frames_dict={}
    for k in range(1, 2**PHASE_DETAIL):
        for direction in ["v","h"]:
            for i,shita in enumerate( [0,1/3,2/3]):
                name = f"{direction}_{k}_{i}"
                phase_frame=phase_shift(freq=k, shift=shita+1/6,direction=direction)
                phase_frame = cv2.cvtColor(phase_frame, cv2.COLOR_GRAY2BGR)
                cv2.putText(phase_frame, str(name), (70,120), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2, cv2.LINE_AA)
                phase_frames_dict[name]=phase_frame

    keys = list(phase_frames_dict.keys())
    values = list(phase_frames_dict.values())

    view_frames=[None]*6

    
    cap_frames={}
    frame_count=0
    offset = 1
    
    while True:
        if loop_max==frame_count:
            break

        # 前のロジックを復元: offsetでカメラとモニタの遅延を吸収
        name = keys[(frame_count ) % len(keys)]
        phase_frame = values[(frame_count+ offset) % len(values)]

        # メインスレッドに投影指示
        # project_queue.put(phase_frame)
        if not IS_PROJECTOR:
            # print("宿所")
            phase_frame=cv2.resize(phase_frame,None ,fx=0.5, fy=0.5)
        cv2.imshow("phase", phase_frame)


        for _ in range(20):
            key =cv2.waitKey(1) & 0xFF
            if key== ord("q"):
                return
            if key==ord("w"):
                offset-=1
                print(f"offset:{offset}")
            if key==ord("e"):
                offset+=1
                print(f"offset:{offset}")

        
        
        # わずかな待ち（以前のwaitKey(1)相当）
        # time.sleep(0.5)

        # 高速にキャプチャ
        ret, cap_frame = cap.read()
        cap_frames[name] = cap_frame
        # print( [f for f in  cap_frames.keys() if f is not None])
        import util
        stack_cap=util.safe_vstack( [f for f in  cap_frames.values() if f is not None])
        stack_cap= cv2.resize(stack_cap, None, fx=0.3, fy=0.3) 
        cv2.imshow("cap",stack_cap)

        

        if frame_count % len(keys) == 0:
            cap_frame_queue.put(cap_frames.copy())
            # cap_frames={}


        frame_count += 1

# 画像サイズ
width = 800*2
height = 400*2

def phase_shift(freq, shift, direction="v"):
    if direction == "v":
        x = np.arange(height)
        wave = np.cos(math.tau * ((freq * x / height) + shift)) * 100 + 128
        gray = np.clip(wave, 0, 255).astype(np.uint8)
        return np.tile(gray.reshape(-1, 1), (1, width))
    else:
        x = np.arange(width)
        wave = np.cos(math.tau * ((freq * x / width) + shift)) * 100 + 128
        gray = np.clip(wave, 0, 255).astype(np.uint8)
        return np.tile(gray, (height, 1))

if __name__ == "__main__":
    q1 = queue.Queue()
    q2 = queue.Queue()
    phase_shifting_capture_loop(q1, q2, loop_max=10)
