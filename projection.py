import numpy as np
import cv2
import pos_ditect 

def get_virtual_projection_matrices(width=1600, height=800):
    # --- 1. 内部パラメータ K (Intrinsic) ---
    # 焦点距離 (f) を適当に1000画素、中心 (cx, cy) を画像中心に設定
    f = 1000
    cx, cy = width / 2, height / 2
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # --- 2. カメラの外部パラメータ (左側の視点) ---
    # 世界座標の原点に配置 (回転なし、移動なし)
    R_cam = np.eye(3) # 単位行列
    t_cam = np.array([[0, 0, 0]], dtype=np.float32).T
    P_cam = K @ np.hstack((R_cam, t_cam))

    # --- 3. プロジェクターの外部パラメータ (右側の視点) ---
    # カメラから右に 200mm (0.2m) 離れた場所に配置
    # (x方向にマイナス移動 = カメラから見て右に物体があるように見える)
    R_proj = np.eye(3)
    t_proj = np.array([[-200, 0, 0]], dtype=np.float32).T 
    P_proj = K @ np.hstack((R_proj, t_proj))

    return P_cam, P_proj




def calculate_depth_from_merged(merged_img, P_cam, P_proj, proj_size=(1600, 800)):
    # 1. チャンネル分解 (B=V座標, G=H座標)
    v_phase = merged_img[:, :, 0]
    h_phase = merged_img[:, :, 1]
    
    h, w = v_phase.shape
    p_w, p_h = proj_size

    # 2. カメラ画素のグリッド生成 (u, v)
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    cam_pts = np.stack([u_coords.ravel(), v_coords.ravel()], axis=0).astype(np.float32)

    # 3. 位相(0-255)をプロジェクターのピクセル座標(x, y)に変換
    # 0 -> 0, 255 -> 最大画素数
    proj_x = (h_phase.ravel().astype(np.float32) / 255.0) * (p_w - 1)
    proj_y = (v_phase.ravel().astype(np.float32) / 255.0) * (p_h - 1)
    proj_pts = np.stack([proj_x, proj_y], axis=0)

    # 4. 三角測量 (4D同次座標が返る)
    points_4d = cv2.triangulatePoints(P_cam, P_proj, cam_pts, proj_pts)

    # 5. 3D座標へ変換 (wで割る)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    
    # 6. Z成分（深度）を取り出し、画像形状に戻す
    depth_map = points_3d[2, :].reshape(h, w)
    return depth_map

# --- 実行セクション ---

# 仮想カメラ行列の準備
P_cam, P_proj = get_virtual_projection_matrices()

# 画像の取得
merged = pos_ditect.get_projection_img()

# 深度計算
depth = calculate_depth_from_merged(merged, P_cam, P_proj)

depth = cv2.normalize(depth, None, alpha=np.min(depth), beta=np.max(depth), norm_type=cv2.NORM_MINMAX)

cv2.imshow("Original Merged (V/H Phase)", depth)
# cv2.imshow("Reconstructed Depth Map", depth_color)
cv2.waitKey(0)