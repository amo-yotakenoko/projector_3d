import phase_shifting
import cv2
import pos_ditect
import numpy as np
import os
import threading
import queue
import phase_shifting_capture_loop 
import calucullate_depth


PHASE_DETAIL=1

ENABLE_CAMERA=True


phase_frames={"v":{},"h":{}}

window_name ="phase"

# 2. 表示したいディスプレイの「左上端の座標」を指定して移動させる
    # 例：メイン画面の右側に「1920x1080」のサブ画面があり、そこに表示したい場合
display_x = 1920  # サブ画面の開始X座標
display_y = 0     # サブ画面の開始Y座標
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.moveWindow(window_name, display_x, display_y)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


cap_frame_queue = queue.Queue(maxsize=1)


# カメラらの取得

if ENABLE_CAMERA:
    th = threading.Thread(target=phase_shifting_capture_loop.phase_shifting_capture_loop, args=(cap_frame_queue,))
    th.daemon = True  
    th.start()
    
    # 初回データ取得まで待機
    print("Waiting for initial capture...")
    # 初回はプロジェクタの表示を回しながら待つ
    while True:
      
        
        try:
            cap_frames = cap_frame_queue.get_nowait()
            new_phase_frames = {}
            for name, frame in sorted(cap_frames.items()):
                direction, k, i = name.split('_')
                new_phase_frames.setdefault(direction, {}).setdefault(int(k), [])
                new_phase_frames[direction][int(k)].append(frame)
            phase_frames = new_phase_frames
            break
        except queue.Empty:
            pass
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
            
    print("Initial capture complete.")

analyzer = calucullate_depth.DepthAnalyzer()

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if ENABLE_CAMERA:
        # プロジェクタの表示更新
   

        # 新しいキャプチャセットの取得
        try:
            cap_frames = cap_frame_queue.get_nowait()
            new_phase_frames = {}
            for name, frame in sorted(cap_frames.items()):
                direction, k, i = name.split('_')
                new_phase_frames.setdefault(direction, {}).setdefault(int(k), [])
                new_phase_frames[direction][int(k)].append(frame)
            phase_frames = new_phase_frames
            print("Captured new frames")
        except queue.Empty:
            pass
    else:
        # ファイルモードの場合（初回のみ読み込み）
        if not phase_frames.get("v"):
            for direction in ["v","h"]:
                for k in [2**i for i in range(PHASE_DETAIL)]:
                    phase_frames[direction][k]=[cv2.imread(f"phase_frames_4_none/{direction}_{k}_{i}.png") for i in range(3)]

    # 解析と表示を実行
    if phase_frames.get("v") and phase_frames.get("h"):
        analyzer.update(phase_frames, key)

    if not ENABLE_CAMERA and key == ord('q'):
        break

    if key == 27 or key == ord('q'):
        break
