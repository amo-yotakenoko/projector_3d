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

# Normalize uv_img for better visualization (prevent overexposure)
valid_mask = np.any(uv_img > 0, axis=2)
if np.any(valid_mask):
    for i in range(2): # Normalize G and B channels separately
        channel = uv_img[:, :, i]
        c_min = np.min(channel[valid_mask])
        c_max = np.max(channel[valid_mask])
        if c_max > c_min:
            uv_img[:, :, i] = np.where(valid_mask, (channel - c_min) / (c_max - c_min), 0)

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

    M = cv2.getPerspectiveTransform(points, dst_pts)
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
    
    # Mask for valid pixels in the warped image
    valid_warped_mask = np.any(warped > 0, axis=2)

    # Robust dynamic normalization: Use a percentile (e.g., 98th) to avoid being skewed by outliers
    if np.any(valid_warped_mask):
        max_val = np.percentile(error_dist[valid_warped_mask], 80)
    else:
        max_val = 0

    if max_val > 0:
        error_intensity = np.clip(error_dist / max_val * 255.0, 0, 255).astype(np.uint8)
        # Apply a colormap for better visibility (Jet: Blue=Low, Red=High)
        dist_display = cv2.applyColorMap(error_intensity, cv2.COLORMAP_JET)
        # Force masked areas to be black
        dist_display[~valid_warped_mask] = 0
    else:
        error_intensity = np.zeros((dst_h, dst_w), dtype=np.uint8)
        dist_display = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    cv2.imshow("error_distance", dist_display)

    error_angle = np.arctan2(dy, dx)
    hsv_error = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    hsv_error[:, :, 0] = ((error_angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv_error[:, :, 1] = 255
    hsv_error[:, :, 2] = error_intensity # Brightness depends on error magnitude (grayscale intensity)
    
    angle_display = cv2.cvtColor(hsv_error, cv2.COLOR_HSV2BGR)
    cv2.imshow("error_angle", angle_display)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Check if 'o' is held down for continuous optimization
    if key == ord('o'):
        # One iteration of Gradient Descent per frame
        learning_rate = 5.0
        eps = 1.0

        grad = np.zeros_like(points)
        
        def get_loss(pts):
            Mat = cv2.getPerspectiveTransform(pts, dst_pts)
            W = cv2.warpPerspective(uv_img, Mat, (dst_w, dst_h))
            
            # Mask valid pixels (where any channel is non-zero)
            valid_warped_mask = np.any(W > 0, axis=2)
            
            if not np.any(valid_warped_mask):
                return 1e10 # Return large loss if no valid pixels in frame

            W_f = W.astype(np.float32) * 255.0
            grad_f = gradient_img.astype(np.float32)
            
            DX = W_f[:, :, 1] - grad_f[:, :, 1]
            DY = W_f[:, :, 0] - grad_f[:, :, 0]
            
            # Calculate mean squared error only for valid pixels
            mse = np.mean((DX[valid_warped_mask]**2 + DY[valid_warped_mask]**2))
            
            # Add a penalty for missing coverage to encourage larger valid area
            coverage_penalty = np.mean(~valid_warped_mask) * 1000.0
            
            return mse + coverage_penalty

        current_loss = get_loss(points)
        
        # Numerical gradient
        for i in range(4):
            for j in range(2):
                points_plus = points.copy()
                points_plus[i, j] += eps
                loss_plus = get_loss(points_plus)
                grad[i, j] = (loss_plus - current_loss) / eps
        
        # Update points (only when not dragging)
        if dragging_idx == -1:
            step = learning_rate * grad
            
            # Limit movement to 2 pixels per frame for each point
            for i in range(4):
                mag = np.linalg.norm(step[i])
                if mag > 2.0:
                    step[i] = (step[i] / mag) * 2.0
            
            points -= step
            
            # Keep points within image boundaries
            points[:, 0] = np.clip(points[:, 0], 0, uv_img.shape[1] - 1)
            points[:, 1] = np.clip(points[:, 1], 0, uv_img.shape[0] - 1)
            # print(f"Loss: {current_loss:.4f}")

    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()