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

# 実行
camera_matrix,_ = load_and_print_npz()
camera_matrix
print(camera_matrix)



Rt_camera = np.hstack((np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)))

# 4. 内部行列 K と掛けて投影行列 P を作成
camera_matrix = camera_matrix @ Rt_camera


projector_matrix= get_projector_internal_matrix()

Rt_projector = np.hstack((np.eye(3, dtype=np.float32), np.array([[ 120.0,0.0,0.0]], dtype=np.float32).T))

projector_matrix = projector_matrix  @ Rt_projector


print("Projector Internal Matrix:")
print(projector_matrix)