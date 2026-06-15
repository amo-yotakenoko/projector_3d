import phase_shifting
import cv2
import pos_ditect


PHASE_DETAIL=2

ENABLE_CAMERA=False


phase_frames={"v":{},"h":{}}



# カメラらの取得

if ENABLE_CAMERA:

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    for direction in ["v","h"]:
        for k in [2**i for i in range(PHASE_DETAIL)]:
            imgs, cams = phase_shifting.capture_phase_shift_set(cap,direction, k)
            phase_frames[direction][k]=cams

            for i, frame in enumerate(cams):
                cv2.imwrite(f"phase_frames/{direction}_{k}_{i}.png", frame)

else:
    for direction in ["v","h"]:
        for k in [2**i for i in range(PHASE_DETAIL)]:
            phase_frames[direction][k]=[cv2.imread(f"phase_frames/{direction}_{k}_{i}.png") for i in range(3)]


# uv座標を取得


uv_img=pos_ditect.get_projection_img(phase_frames)

cv2.imshow("uv", uv_img)

cv2.waitKey(0)