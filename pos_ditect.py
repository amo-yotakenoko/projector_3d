import cv2
import numpy as np
import math
import phase_shifting as ps


def normalized_phis(k,direction):
    
    # _, imgs = ps.phase_shift_set(direction, k)
    _, imgs = ps.phase_shift_set(direction, k)
    # print(  f"{k}_{i}.png")
    height, width, _ = imgs[0].shape

    new_img = np.zeros((height, width), dtype=np.float32)


    I1 = imgs[0].mean(axis=2)
    I2 = imgs[1].mean(axis=2)
    I3 = imgs[2].mean(axis=2)

    phi = np.arctan2(
        np.sqrt(3) * (I1 - I3),
        2 * I2 - I1 - I3
    )


    result=(phi + np.pi) / (2 * np.pi)



    
    cv2.imshow(f"phase", result)
    cv2.waitKey(10)
    return result




def get_phase_img(direction):
        

    result=normalized_phis(1, direction)

    # cv2.imshow(f"step0", normalized_phis[1])

    for low, high in [(1, 2**(i+1)) for i in range(8-1)]:
        ratio = high / low
        print(ratio)
        # cv2.imshow(f"normalized_phis[{low}]", normalized_phis[low])
        # cv2.imshow(f"normalized_phis[{high}]", normalized_phis[high])
        # normalized_phis[high]=cv2.imread(f"{1}_{0}.png")


        k = np.round(result * ratio - normalized_phis(high, direction))
        k /= ratio
        # cv2.imshow(f"k_{low}_{high}1" ,k)


        
        unwrap =  k+normalized_phis(high, direction) / ratio

        cv2.waitKey(10)
        # cv2.destroyAllWindows()
        result=unwrap
        cv2.imshow(f"result", unwrap)

    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result


def get_projection_img():
    v_img=get_phase_img("v")
    h_img=get_phase_img("h")
    merged_img = cv2.merge([v_img, h_img,np.full_like(v_img, 0)])
    return merged_img

if __name__ == "__main__":

    v_img=get_phase_img("v")
    h_img=get_phase_img("h")
    merged_img = cv2.merge([v_img, h_img,np.full_like(v_img, 0)])

    cv2.imshow("v_img", v_img)
    cv2.imshow("h_img", h_img)
    cv2.imshow("merged_img", merged_img)
    cv2.waitKey(0)




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



