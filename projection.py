import numpy as np
import cv2
import pos_ditect 
import phase_shifting as ps
import open3d as o3d
import tkinter as tk
from tkinter import ttk

from cam_param import get_projection_matrices

def apply_color_map(img):
    # 0-255に正規化
    mn = np.min(img)
    mx = np.max(img)
    if mx == mn:
        return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    # 正規化処理を明示的に行う (0.0 - 1.0 にしてから 255 倍)
    norm_img = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
    
    # カラーマップを適用 (JETは青から赤への変化)
    return cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)

class ParamControl:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Projector Param Control")
        
        self.params = {
            'tx': tk.DoubleVar(value=5.0),
            'ty': tk.DoubleVar(value=0.0),
            'tz': tk.DoubleVar(value=0.0),
            'rx': tk.DoubleVar(value=0.0),
            'ry': tk.DoubleVar(value=0.0),
            'rz': tk.DoubleVar(value=0.0)
        }
        
        self._create_widgets()
        
    def _create_widgets(self):
        ttk.Label(self.root, text="Translation").grid(row=0, column=0, columnspan=2)
        self._add_slider("X", 'tx', -100, 100, 1)
        self._add_slider("Y", 'ty', -100, 100, 2)
        self._add_slider("Z", 'tz', -100, 100, 3)
        
        ttk.Label(self.root, text="Rotation").grid(row=4, column=0, columnspan=2)
        self._add_slider("Rx", 'rx', -np.pi, np.pi, 5)
        self._add_slider("Ry", 'ry', -np.pi, np.pi, 6)
        self._add_slider("Rz", 'rz', -np.pi, np.pi, 7)

    def _add_slider(self, label, var_name, from_, to, row):
        ttk.Label(self.root, text=label).grid(row=row, column=0)
        ttk.Scale(self.root, from_=from_, to=to, variable=self.params[var_name], orient='horizontal', length=200).grid(row=row, column=1)
        ttk.Label(self.root, textvariable=self.params[var_name]).grid(row=row, column=2)

    def get_values(self):
        t = [self.params['tx'].get(), self.params['ty'].get(), self.params['tz'].get()]
        r = [self.params['rx'].get(), self.params['ry'].get(), self.params['rz'].get()]
        return t, r

    def update(self):
        try:
            self.root.update()
            return True
        except:
            return False

if __name__ == "__main__":
    merged_full = pos_ditect.get_projection_img()
    
    # 処理軽量化のためにサイズを半分に
    h_orig, w_orig = merged_full.shape[:2]
    h_small, w_small = h_orig // 2, w_orig // 2
    merged = cv2.resize(merged_full, (w_small, h_small), interpolation=cv2.INTER_NEAREST)
    
    ctrl = ParamControl()

    while True:
        if not ctrl.update():
            break
            
        t, r = ctrl.get_values()
        
        # 動的に行列を更新
        C_cam, C_proj = get_projection_matrices(projector_t=t, projector_r=r)

        # 高速化のためにベクトル化
        h, w = merged.shape[:2]
        v_phase = merged[:,:,0]
        h_phase = merged[:,:,1]
        
        # プロジェクター上の座標を計算
        proj_x = h_phase * ps.width
        proj_y = v_phase * ps.height
        
        # カメラ座標のグリッドを作成
        # 注意: 元のサイズに対する座標にする必要があるため、スケーリングを考慮
        yy, xx = np.mgrid[0:h, 0:w]
        xx_full = (xx * (w_orig / w_small)).astype(np.float32)
        yy_full = (yy * (h_orig / h_small)).astype(np.float32)
        
        # 三角測量用に形を整える (2, N)
        pts_cam = np.vstack([xx_full.ravel(), yy_full.ravel()]).astype(np.float32)
        pts_proj = np.vstack([proj_x.ravel(), proj_y.ravel()]).astype(np.float32)
        
        # 一括で三角測量 (4, N)
        points_4d = cv2.triangulatePoints(C_cam, C_proj, pts_cam, pts_proj)
        
        # 同次座標から3D座標へ変換
        W = points_4d[3, :]
        points_3d = points_4d[:3, :] / np.where(W != 0, W, 1.0)
        
        # 低解像度のマップを作成
        depth_map_small = np.zeros((h, w, 3), dtype=np.float32)
        depth_map_small[:,:,0] = points_3d[0, :].reshape(h, w)
        depth_map_small[:,:,1] = points_3d[1, :].reshape(h, w)
        depth_map_small[:,:,2] = points_3d[2, :].reshape(h, w)

        # 各軸を表示（カラーマップ適用）
        for i, axis in enumerate(['x', 'y', 'z']):
            colored = apply_color_map(depth_map_small[:,:,i])
            # 表示用に元のサイズに戻す
            display = cv2.resize(colored, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(f"Depth Map {axis} (Color)", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
