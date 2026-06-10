import numpy as np
import phase_shifting as ps

np.set_printoptions(precision=3, suppress=True)

def get_projector_internal_matrix():
    # --- 1. 物理パラメータ (単位: m でも mm でも比率は同じ) ---
    distance = 1260.0     # 距離
    width_size = 810.0    # 投影面の幅
    height_size = 480.0   # 投影面の高さ

    # --- 2. 解像度 (単位: px) ---
    pixel_w = ps.width
    pixel_h = ps.height

    # --- 3. 内部行列 K の計算 ---
    # fx = (画素幅 / 実サイズ幅) * 距離
    f_x = (pixel_w / width_size) * distance
    f_y = (pixel_h / height_size) * distance

    # 中心座標 (画像サイズの中央)
    c_x = pixel_w / 2.0
    c_y = pixel_h / 2.0

    # 3x3 の内部行列 K
    K = np.array([
        [f_x, 0,   c_x],
        [0,   f_y, c_y],
        [0,   0,   1]
    ], dtype=np.float32)

    return K



def load_and_print_npz(file_path='cam.npz'):
    try:
        # npzファイルをロード
        data = np.load(file_path)

        camera_matrix,dist_coeffs=data['camera_matrix'], data['dist_coeffs']
        
       
            
        return camera_matrix, dist_coeffs
    except FileNotFoundError:
        print(f"Error: {file_path} が見つかりません。パスを確認してください。")
    except Exception as e:
        print(f"Error: {e}")

def get_projection_matrices(projector_t=[5.0, 0.0, 0.0], projector_r=[0.0, 0.0, 0.0]):
    # カメラ内部行列のロード
    camera_matrix_internal, _ = load_and_print_npz()
    
    # カメラのRt (基準なので単位行列)
    Rt_camera = np.hstack((np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)))
    camera_matrix = camera_matrix_internal @ Rt_camera

    # プロジェクター内部行列の取得
    projector_matrix_internal = get_projector_internal_matrix()

    # プロジェクターの回転行列 (Euler角から変換 - 簡易版)
    rx, ry, rz = projector_r
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    # プロジェクターのRt
    Rt_projector = np.hstack((R.astype(np.float32), np.array([projector_t], dtype=np.float32).T))
    projector_matrix = projector_matrix_internal @ Rt_projector

    return camera_matrix, projector_matrix

# 初期値での互換性維持
camera_matrix, projector_matrix = get_projection_matrices()
