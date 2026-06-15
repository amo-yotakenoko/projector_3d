import phase_shifting
import cv2
import pos_ditect
import numpy as np


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
            phase_frames[direction][k]=[cv2.imread(f"phase_frames_2/{direction}_{k}_{i}.png") for i in range(3)]


# uv座標を取得


uv_img=pos_ditect.get_projection_img(phase_frames)

# Interactive warping
points = np.array([
    [100, 100],
    [uv_img.shape[1] - 100, 100],
    [uv_img.shape[1] - 100, uv_img.shape[0] - 100],
    [100, uv_img.shape[0] - 100]
], dtype=np.float32)

dragging_idx = -1

def mouse_callback(event, x, y, flags, param):
    global points, dragging_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if np.linalg.norm(points[i] - [x, y]) < 15:
                dragging_idx = i
                break
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_idx != -1:
            points[dragging_idx] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_idx = -1

cv2.namedWindow("uv")
cv2.setMouseCallback("uv", mouse_callback)

while True:
    # Drawing on a 8-bit version for visualization
    display_img = (uv_img * 255).astype(np.uint8)
    
    # Draw points and lines
    for i in range(4):
        pt1 = tuple(points[i].astype(int))
        pt2 = tuple(points[(i + 1) % 4].astype(int))
        cv2.circle(display_img, pt1, 8, (0, 0, 255), -1)
        cv2.line(display_img, pt1, pt2, (0, 255, 0), 2)
    
    cv2.imshow("uv", display_img)

    # Perspective Warp settings
    dst_w, dst_h = 800, 480
    dst_pts = np.array([
        [0, 0],
        [dst_w, 0],
        [dst_w, dst_h],
        [0, dst_h]
    ], dtype=np.float32)

    # Calculate 1.5x expanded points
    center = np.mean(points, axis=0)
    expanded_points = center + (points - center) * 1.5
    
    M = cv2.getPerspectiveTransform(expanded_points, dst_pts)
    warped = cv2.warpPerspective(uv_img, M, (dst_w, dst_h))
    
    cv2.imshow("warped", warped)

    # Create gradient image (matching 800x480)
    grid_x, grid_y = np.meshgrid(np.linspace(0, 255, dst_w), np.linspace(0, 255, dst_h))
    gradient_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    gradient_img[:, :, 1] = grid_x.astype(np.uint8) # Green channel (x gradient)
    gradient_img[:, :, 0] = grid_y.astype(np.uint8) # Blue channel (y gradient)
    
    cv2.imshow("gradient", gradient_img)

    # Calculate error (distance and angle)
    warped_float = warped.astype(np.float32) * 255.0
    ideal_float = gradient_img.astype(np.float32)

    dx = warped_float[:, :, 1] - ideal_float[:, :, 1]
    dy = warped_float[:, :, 0] - ideal_float[:, :, 0]

    error_dist = np.sqrt(dx**2 + dy**2)
    dist_display = np.clip(error_dist * (255.0 / 50.0), 0, 255).astype(np.uint8)
    cv2.imshow("error_distance", dist_display)

    error_angle = np.arctan2(dy, dx)
    hsv_error = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    hsv_error[:, :, 0] = ((error_angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv_error[:, :, 1] = 255
    hsv_error[:, :, 2] = 255
    
    angle_display = cv2.cvtColor(hsv_error, cv2.COLOR_HSV2BGR)
    cv2.imshow("error_angle", angle_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()