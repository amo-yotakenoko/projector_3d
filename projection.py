import numpy as np
import cv2
import pos_ditect 
import phase_shifting as ps
import open3d as o3d

from cam_param import camera_matrix, projector_matrix


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






# --- 実行セクション ---

# 仮想カメラ行列の準備
P_cam, P_proj = get_virtual_projection_matrices()

# 画像の取得
merged = pos_ditect.get_projection_img()




depth_map = np.zeros((merged.shape[0], merged.shape[1], 4), dtype=np.float32)

for y in range(merged.shape[0]):
    for x in range(merged.shape[1]):
        v_phase, h_phase, _ = merged[y, x]
        
        # 位放値が有効な場合のみ計算（例: 0より大きい場合）
        if v_phase > 0 and h_phase > 0:
            # 1. プロジェクター上の画素座標に変換
            # h_phase (横方向の変化) -> X座標
            # v_phase (縦方向の変化) -> Y座標
            proj_x = h_phase * ps.width
            proj_y = v_phase * ps.height
            
            # 2. 三角測量用に座標を整形
            # points_cam: カメラ (x, y)
            # points_proj: プロジェクター (proj_x, proj_y)
            pts_cam = np.array([[x, y]], dtype=np.float32).T
            pts_proj = np.array([[proj_x, proj_y]], dtype=np.float32).T
            
            # 3. 三角測量の実行 (4D同次座標 [X, Y, Z, W] が返る)
            points_4d = cv2.triangulatePoints(camera_matrix, projector_matrix, pts_cam, pts_proj)
            
            # depth_mapに保存 (生の同次座標を保存)
            depth_map[y, x] = points_4d.flatten()

print(f"{depth_map=}")

def show_normalized_map(name, data):
    # 0 (背景) 以外の値を取り出す
    mask = (data != 0)
    if not np.any(mask):
        img = np.zeros_like(data, dtype=np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imshow(name, img)
        return img

    valid_data = data[mask]
    
    # 外れ値に強くするため、パーセンタイル（2%〜98%）で最小・最大を決定する
    min_val = np.percentile(valid_data, 2)
    max_val = np.percentile(valid_data, 98)

    # スケーリング (0.0 ~ 1.0)
    if max_val - min_val > 0:
        normalized = np.zeros_like(data)
        # クリップして範囲外を 0 or 1 に固定
        clipped = np.clip(data[mask], min_val, max_val)
        normalized[mask] = (clipped - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(data)

    # 255倍して uint8 に変換
    gray_img = (normalized * 255).astype(np.uint8)
    
    # カラーマップを適用 (JET: 青 -> 緑 -> 赤)
    color_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    
    # マスク外（背景）を黒にする
    color_img[~mask] = 0
    
    cv2.imshow(name, color_img)
    return color_img

# 各チャンネルを表示

# 同次座標から 3D 空間座標 (X, Y, Z) への変換 (除算処理)
W = depth_map[:, :, 3]
mask = np.abs(W) > 1e-6

X = np.zeros_like(W)
Y = np.zeros_like(W)
Z = np.zeros_like(W)

X[mask] = depth_map[mask, 0] / W[mask]
Y[mask] = depth_map[mask, 1] / W[mask]
Z[mask] = depth_map[mask, 2] / W[mask]

x_img = show_normalized_map("x_map", X)
y_img = show_normalized_map("y_map", Y)
z_img = show_normalized_map("z_map", Z)
w_img = show_normalized_map("w_map", W)

# 画像として保存
cv2.imwrite("output_x_map.png", x_img)
cv2.imwrite("output_y_map.png", y_img)
cv2.imwrite("output_z_map.png", z_img)
cv2.imwrite("output_w_map.png", w_img)
print("画像を保存しました: output_x_map.png, output_y_map.png, output_z_map.png, output_w_map.png")

# Open3D を使った 3D 表示 (いったん無効化)
# print("Open3D で 3D 表示を開始します...")
# points = np.stack((X[mask], Y[mask], Z[mask]), axis=-1)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([pcd, axes], window_name="3D Point Cloud (Rotatable)")

cv2.waitKey(0)