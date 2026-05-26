import numpy as np
import cv2
import pos_ditect 
import phase_shifting as ps
import open3d as o3d

from cam_param import camera_matrix, projector_matrix




def normalize_view(img):
    mn = np.min(img)
    mx = np.max(img)

    # if mx == mn:
    #     return np.zeros_like(img, dtype=np.uint8)

    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)




if __name__ == "__main__":


    merged = pos_ditect.get_projection_img()

    print(f"{np.min(merged[:,:,0])=}")
    print(f"{np.max(merged[:,:,0])=}")
    print(f"{np.min(merged[:,:,1])=}")
    print(f"{np.max(merged[:,:,1])=}")

    # print(np.min(merged[:,:,0]), np.max(merged[:,:,0]))
    # print(np.min(merged[:,:,1]), np.max(merged[:,:,1]))

    while True:
        # --- 実行セクション ---

        # 仮想カメラ行列の準備
        # P_cam, P_proj = get_virtual_projection_matrices()
        C_cam, C_proj=camera_matrix, projector_matrix 



        # print(f"{camera_matrix=}")
        # print(f"{projector_matrix=}")

        # print(f"{P_cam=}")
        # print(f"{P_proj=}")

        # 画像の取得




        depth_map = np.zeros((merged.shape[0], merged.shape[1], 4), dtype=np.float32)

        for y in range(merged.shape[0]):
            for x in range(merged.shape[1]):
                v_phase, h_phase, _ = merged[y, x]
                
                # 位放値が有効な場合のみ計算（例: 0より大きい場合）
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
                points_4d = cv2.triangulatePoints( C_cam, C_proj, pts_cam, pts_proj)
                
                W = points_4d[3,0]
                X = (points_4d[0,0]/W+0.5)
                Y = (points_4d[1,0]/W+0.5)
                Z = (points_4d[2,0]/W+0.5)
                depth_map[y, x] = [X, Y, Z, W]
                    # points_3d = points_4d[:3] / points_4d[3]
                    




        cv2.imshow("Depth Map x",  normalize_view(depth_map[:,:,0]))
        cv2.imshow("Depth Map y",  normalize_view(depth_map[:,:,1]))
        cv2.imshow("Depth Map z",  normalize_view(depth_map[:,:,2]))
        cv2.imshow("Depth Map w",  normalize_view(depth_map[:,:,3]))
        # cv2.imshow("Depth Map w", normalize_view(depth_map[:,:,3]))
        # cv2.imshow("Depth Map asd", normalize_view(depth_map[:,:,3]/depth_map[:,:,2]))



        cv2.waitKey(0)
