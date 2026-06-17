import cv2
import numpy as np
import math
from pathlib import Path
import shutil, os

# フォルダごと削除して、すぐ空のフォルダを作り直す（1行ずつ）



PHASE_DETAIL=1

def phase_shifting_capture_loop(loop_max=-1):

    # 1. 普通のウィンドウとして作成する（ここがポイント）
    # ※最初からフルスクリーンで開くと moveWindow が効かないことがあります
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # # 2. 表示したいディスプレイの「左上端の座標」を指定して移動させる
    # # 例：メイン画面の右側に「1920x1080」のサブ画面があり、そこに表示したい場合
    # display_x = 1920  # サブ画面の開始X座標
    # display_y = 0     # サブ画面の開始Y座標
    # cv2.moveWindow(window_name, display_x, display_y)

    # # 3. 移動した後にフルスクリーン化する
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    for _ in range(10):
        ret, cap_frame = cap.read()
        cv2.imshow("camera", cap_frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


    phase_frames={}
    cap_frames={}

    for k in range(1, 2**PHASE_DETAIL):
        for direction in ["v","h"]:
            for i,shita in enumerate( [0,1/3,2/3]):
                name = f"{direction}_{k}_{i}"

                phase_frame=phase_shift(freq=k, shift=shita+1/6,direction=direction)
                phase_frame = cv2.cvtColor(phase_frame, cv2.COLOR_GRAY2BGR)
                cv2.putText(phase_frame, str(name), (70,120), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2, cv2.LINE_AA)
                phase_frames[name]=phase_frame


    frame_count=0
    while True:
        if loop_max==frame_count:
            break

        offset = 2

        name = list(phase_frames.keys())[(frame_count+offset) % len(phase_frames)]


        # x番目の値を取得する場合
        phase_frame = list(phase_frames.values())[frame_count % len(phase_frames)]

        cv2.imshow("phase", phase_frame)
        cv2.waitKey(1)

        for _ in range(1):
            ret, cap_frame = cap.read()
            cap_frames[name] = cap_frame

        


        frame_count += 1

    for name, cap_frame in cap_frames.items():
        cv2.imwrite(f"phase_frames/{name}.png", cap_frame)





    # for i in range(10):
        

    #     imgs=[]
    #     cams=[]
    #     for direction in ["v","h"]:
    #         for i,shita in enumerate( [0,1/3,2/3]):
    #             path = Path(f"phase_frames/{direction}_{k}_{i}.png")
                
    #             imgs.append(phase_shift(freq=k, shift=shita+1/6,direction=direction))
    #             imgs[-1] = cv2.cvtColor(imgs[-1], cv2.COLOR_GRAY2BGR)
    #                 # 表示
    #             cv2.putText(imgs[-1], str(path), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    #             cv2.imshow(window_name, imgs[-1])
    #             cv2.waitKey(1)

                    
    #             for _ in range(1):
    #                 ret, frame = cap.read()
    #                 cv2.imshow("camera", frame)
    #                 cv2.waitKey(1)
    #             cv2.imwrite(str(path), frame)
    #             cams.append(frame)
    #             path.parent.mkdir(parents=True, exist_ok=True)



# 画像サイズ
width = 800*2
height = 400*2




window_name = "Fullscreen_Window"

def phase_shift(freq, shift, direction="v"):
    # 指定サイズ





    if direction == "v":
        # 縦方向に変化（横縞）
        # 1. 縦の長さ（h=800）分、波を作る
        x = np.arange(height)
        wave = np.cos(math.tau * ((freq * x / height) + shift)) * 100 + 128
        gray = np.clip(wave, 0, 255).astype(np.uint8)
        
        # 2. 「縦1列(800, 1)」の形にしてから、横に1600個並べる
        # ※ここを reshape(-1, 1) にするのが最大のポイント
        return np.tile(gray.reshape(-1, 1), (1, width))

    else:
        # 横方向に変化（縦縞）
        # 1. 横の長さ（w=1600）分、波を作る
        x = np.arange(width)
        wave = np.cos(math.tau * ((freq * x / width) + shift)) * 100 + 128
        gray = np.clip(wave, 0, 255).astype(np.uint8)
        
        # 2. 「横1行(1, 1600)」を、縦に800個並べる
        return np.tile(gray, (height, 1))


if __name__ == "__main__":
    shutil.rmtree("./phase_frames", ignore_errors=True)
    os.makedirs("./phase_frames", exist_ok=True)
    phase_shifting_capture_loop(loop_max=10)