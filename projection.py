import numpy as np
import cv2
import pos_ditect 
import phase_shifting as ps

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
        
        # 位相値が有効な場合のみ計算（例: 0より大きい場合）
        if v_phase > 0 and h_phase > 0:
            # 1. プロジェクター上の画素座標に変換
            proj_x = v_phase * ps.width
            proj_y = h_phase * ps.height
            
            # 2. 三角測量用に座標を整形
            # points_cam: カメラの (x, y)
            # points_proj: プロジェクターの (proj_x, proj_y)
            pts_cam = np.array([[x, y]], dtype=np.float32).T
            pts_proj = np.array([[proj_x, proj_y]], dtype=np.float32).T
            
            # 3. 三角測量の実行 (4D同次座標が返る)
            points_4d = cv2.triangulatePoints(camera_matrix, projector_matrix, pts_cam, pts_proj)
            # print(f"{points_4d=}")
            # 4. 3D座標に変換 (x, y, z, w) -> (X, Y, Z)
            points_3d = points_4d[:3, :] / points_4d[3, :]
            
            # depth_mapに保存 (Z座標が奥行き)
            depth_map[y, x] = points_4d.flatten()

print(f"{depth_map=}")
    # 1行終わるごとに進捗を表示したい場合はここ
def show_normalized_map(name, data):
    # 最小値と最大値を取得
    min_val = np.min(data)
    max_val = np.max(data)
    
    # 0除算を防ぎつつ 0.0 ~ 1.0 にスケーリング
    if max_val - min_val > 0:
        normalized = (data - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(data)
    
    # 255倍して uint8 に変換
    display_img = (normalized * 255).astype(np.uint8)
    cv2.imshow(name, display_img)

# 各チャンネルを表示
show_normalized_map("x_map", depth_map[:,:,0])
show_normalized_map("y_map", depth_map[:,:,1])
show_normalized_map("z_map", depth_map[:,:,2])
show_normalized_map("w_map", depth_map[:,:,3])
# 最終的な奥行き(Z)を取り出す
z_coords = depth_map[:, :, 2]
# print(f"中心付近の奥行き: {z_coords[h//2, w//2]} mm")

cv2.waitKey(0)