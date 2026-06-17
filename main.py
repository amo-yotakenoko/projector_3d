import phase_shifting
import cv2
import pos_ditect
import numpy as np
import os
import threading
import queue


PHASE_DETAIL=1

ENABLE_CAMERA=False


phase_frames={"v":{},"h":{}}



# カメラらの取得

if ENABLE_CAMERA:

    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # for direction in ["v","h"]:
    #     for k in [2**i for i in range(PHASE_DETAIL)]:
    #         imgs, cams = phase_shifting.capture_phase_shift_set(cap,direction, k)
    #         phase_frames[direction][k]=cams

    #         for i, frame in enumerate(cams):
    #             cv2.imwrite(f"phase_frames/{direction}_{k}_{i}.png", frame)

else:
    for direction in ["v","h"]:
        for k in [2**i for i in range(PHASE_DETAIL)]:
            phase_frames[direction][k]=[cv2.imread(f"phase_frames_4_none/{direction}_{k}_{i}.png") for i in range(3)]


# uv座標を取得


uv_img=pos_ditect.get_projection_img(phase_frames)

# Apply median filter to reduce noise and fill small gaps in the UV map
# Note: Converting to uint8 for compatibility with some OpenCV builds
uv_img_uint8 = (uv_img * 255).astype(np.uint8)
ksize = 1 # Adjust this (3, 5, 7, 9...) to change blur strength
for i in range(3):
    uv_img_uint8[:, :, i] = cv2.medianBlur(uv_img_uint8[:, :, i], ksize)
uv_img = uv_img_uint8.astype(np.float32) / 255.0


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
if os.path.exists("points.npy"):
    points = np.load("points.npy")
else:
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

    # Mask for valid pixels in the warped image
    valid_warped_mask = np.any(warped > 0, axis=2)

    # Remove small noise from mask using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    valid_warped_mask = cv2.morphologyEx(valid_warped_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    # Calculate error (distance and angle) only for valid pixels
    warped_float = warped.astype(np.float32) * 255.0
    ideal_float = gradient_img.astype(np.float32)

    dx = np.zeros((dst_h, dst_w), dtype=np.float32)
    dy = np.zeros((dst_h, dst_w), dtype=np.float32)
    dx[valid_warped_mask] = warped_float[valid_warped_mask, 1] - ideal_float[valid_warped_mask, 1]
    dy[valid_warped_mask] = warped_float[valid_warped_mask, 0] - ideal_float[valid_warped_mask, 0]

    error_dist = np.zeros((dst_h, dst_w), dtype=np.float32)
    error_dist[valid_warped_mask] = np.sqrt(dx[valid_warped_mask]**2 + dy[valid_warped_mask]**2)
    
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

    error_angle = np.zeros((dst_h, dst_w), dtype=np.float32)
    error_angle[valid_warped_mask] = np.arctan2(dy[valid_warped_mask], dx[valid_warped_mask])
    
    hsv_error = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    hsv_error[valid_warped_mask, 0] = ((error_angle[valid_warped_mask] + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv_error[valid_warped_mask, 1] = 255
    hsv_error[valid_warped_mask, 2] = error_intensity[valid_warped_mask] # Brightness depends on error magnitude (grayscale intensity)
    
    angle_display = cv2.cvtColor(hsv_error, cv2.COLOR_HSV2BGR)
    cv2.imshow("error_angle", angle_display)
    
    # Draw error vectors as arrows
    vector_display = (uv_img * 255).astype(np.uint8) # Or use a blank image, but warped size matches dst_w, dst_h
    vector_display = cv2.cvtColor(cv2.warpPerspective(display_img, M, (dst_w, dst_h)), cv2.COLOR_BGR2GRAY)
    vector_display = cv2.cvtColor(vector_display, cv2.COLOR_GRAY2BGR)
    
    step = 20
    count_under_10 = 0
    total_valid = np.sum(valid_warped_mask)
    
    for y in range(0, dst_h, step):
        for x in range(0, dst_w, step):
            if valid_warped_mask[y, x]:
                vx = int(dx[y, x])
                vy = int(dy[y, x])
                err = np.sqrt(vx**2 + vy**2)
                if err > 0.5: # Only draw if there is a noticeable error
                    start_point = (x, y)
                    end_point = (x + vx, y + vy)
                    # Color based on error threshold
                    color = (0, 255, 0) if err < 10 else (0, 255, 255) # Green if < 10, else Yellow
                    cv2.arrowedLine(vector_display, start_point, end_point, color, 1, tipLength=0.3)
    
    # Calculate and display stats
    if total_valid > 0:
        prec_under_10 = np.sum(error_dist[valid_warped_mask] < 10) / total_valid * 100
        cv2.putText(vector_display, f"Under 10px: {prec_under_10:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("error_vectors", vector_display)
    
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
            
            # Remove small noise from mask for loss calculation consistency
            kernel_l = np.ones((3, 3), np.uint8)
            valid_warped_mask = cv2.morphologyEx(valid_warped_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_l).astype(bool)

            if not np.any(valid_warped_mask):
                return 1e10 # Return large loss if no valid pixels in frame

            W_f = W.astype(np.float32) * 255.0
            grad_f = gradient_img.astype(np.float32)
            
            DX = W_f[:, :, 1] - grad_f[:, :, 1]
            DY = W_f[:, :, 0] - grad_f[:, :, 0]
            
            dist_sq = DX[valid_warped_mask]**2 + DY[valid_warped_mask]**2
            dist = np.sqrt(dist_sq)

            # Prioritize increasing the count of pixels with error < 10px
            # Use a loss that is flat for large errors but steep for small ones to encourage precision
            # Or a modified Charbonnier/Huber loss that favors small residuals
            
            # High precision reward: pixels under 10px get a "steep" gradient to improve further
            # Pixels far away are still pulled in, but we prioritize the "near-perfect" ones
            threshold = 10.0
            
            # Custom loss: 
            # - For dist < threshold, use MSE to drive them to 0
            # - For dist >= threshold, use linear loss to reduce outlier influence 
            #   AND add a penalty for being above the threshold
            loss_precision = np.where(dist < threshold, 
                                     dist_sq, 
                                     threshold * 2 * dist - threshold**2 + 500.0)
            
            mse = np.mean(loss_precision)
            
            # Add a penalty for missing coverage to encourage larger valid area
            coverage_penalty = np.mean(~valid_warped_mask) * 5000.0
            
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

    if key == ord('s'):
        np.save("points.npy", points)
        print("Points saved to points.npy")

    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()