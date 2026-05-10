import cv2
import numpy as np
import math





# 画像サイズ
width = 800*2
height = 400*2

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

window_name = "sin grayscale"

cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

# 2. フルスクリーンプロパティを設定する
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def capture_phase_shift_set(  direction, k):
    imgs=[]
    cams=[]
    for i,shita in enumerate( [0,1/3,2/3]):
        imgs.append(phase_shift(freq=k, shift=shita+1/6,direction=direction))
            # 表示
        cv2.imshow(window_name, imgs[-1])
        cv2.waitKey(1)

            
        for _ in range(10):
            ret, frame = cap.read()
            cv2.imshow("camera", frame)
            cv2.waitKey(1)
        cams.append(frame)

    return imgs, cams

def phase_shift_set(  direction, k):
   imgs= [cv2.imread(f"img/{direction}_{k}_{i}.png") for i in range(3)]
   cams=[cv2.imread(f"cam/{direction}_{k}_{i}.png") for i in range(3)]
   return imgs, cams


if __name__ == "__main__":
    for direction in ["v","h"]:
        for k in [2**i for i in range(8)]:
            imgs, cams = capture_phase_shift_set(direction, k)
            for i, img in enumerate(imgs):
                cv2.imwrite(f"img/{direction}_{k}_{i}.png", img)
            for i, frame in enumerate(cams):
                cv2.imwrite(f"cam/{direction}_{k}_{i}.png", frame)

