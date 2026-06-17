import cv2
import numpy as np
import pos_ditect
import os

class DepthAnalyzer:
    def __init__(self):
        self.points = None
        self.dragging_idx = -1
        self.uv_img = None
        self.last_phase_frames_id = None
        self.dst_w, self.dst_h = 800, 480
        self.dst_pts = np.array([
            [0, 0],
            [self.dst_w, 0],
            [self.dst_w, self.dst_h],
            [0, self.dst_h]
        ], dtype=np.float32)
        
        # Create gradient image (matching 800x480)
        grid_x, grid_y = np.meshgrid(np.linspace(0, 255, self.dst_w), np.linspace(0, 255, self.dst_h))
        self.gradient_img = np.zeros((self.dst_h, self.dst_w, 3), dtype=np.uint8)
        self.gradient_img[:, :, 1] = grid_x.astype(np.uint8) # Green channel (x gradient)
        self.gradient_img[:, :, 0] = grid_y.astype(np.uint8) # Blue channel (y gradient)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(4):
                if np.linalg.norm(self.points[i] - [x, y]) < 15:
                    self.dragging_idx = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_idx != -1:
                self.points[self.dragging_idx] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = -1

    def update(self, phase_frames, key):
        # phase_framesが変わった場合（新しいキャプチャ）、UV画像を再計算
        # 簡易的なIDチェックとして、辞書の長さや適当なハッシュを使いたいが、
        # ここでは単純に別オブジェクトなら更新とする
        current_id = id(phase_frames)
        if current_id != self.last_phase_frames_id:
            print("Updating UV map from new phase frames...")
            uv_img_raw = pos_ditect.get_projection_img(phase_frames)
            
            # フィルタリング
            uv_img_uint8 = (uv_img_raw * 255).astype(np.uint8)
            ksize = 1
            for i in range(3):
                uv_img_uint8[:, :, i] = cv2.medianBlur(uv_img_uint8[:, :, i], ksize)
            self.uv_img = uv_img_uint8.astype(np.float32) / 255.0

            # 正規化
            valid_mask = np.any(self.uv_img > 0, axis=2)
            if np.any(valid_mask):
                for i in range(2):
                    channel = self.uv_img[:, :, i]
                    c_min = np.min(channel[valid_mask])
                    c_max = np.max(channel[valid_mask])
                    if c_max > c_min:
                        self.uv_img[:, :, i] = np.where(valid_mask, (channel - c_min) / (c_max - c_min), 0)
            
            self.last_phase_frames_id = current_id

            # 初回のみポイントを設定
            if self.points is None:
                if os.path.exists("points.npy"):
                    self.points = np.load("points.npy")
                else:
                    self.points = np.array([
                        [100, 100],
                        [self.uv_img.shape[1] - 100, 100],
                        [self.uv_img.shape[1] - 100, self.uv_img.shape[0] - 100],
                        [100, self.uv_img.shape[0] - 100]
                    ], dtype=np.float32)
                
                cv2.namedWindow("uv")
                cv2.setMouseCallback("uv", self.mouse_callback)

        if self.uv_img is None:
            return

        # 1フレーム分の表示・処理
        display_img = (self.uv_img * 255).astype(np.uint8)
        for i in range(4):
            pt1 = tuple(self.points[i].astype(int))
            pt2 = tuple(self.points[(i + 1) % 4].astype(int))
            cv2.circle(display_img, pt1, 8, (0, 0, 255), -1)
            cv2.line(display_img, pt1, pt2, (0, 255, 0), 2)
        cv2.imshow("uv", display_img)

        M = cv2.getPerspectiveTransform(self.points, self.dst_pts)
        warped = cv2.warpPerspective(self.uv_img, M, (self.dst_w, self.dst_h))
        cv2.imshow("warped", warped)
        cv2.imshow("gradient", self.gradient_img)

        valid_warped_mask = np.any(warped > 0, axis=2)
        kernel = np.ones((3, 3), np.uint8)
        valid_warped_mask = cv2.morphologyEx(valid_warped_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

        warped_float = warped.astype(np.float32) * 255.0
        ideal_float = self.gradient_img.astype(np.float32)

        dx = np.zeros((self.dst_h, self.dst_w), dtype=np.float32)
        dy = np.zeros((self.dst_h, self.dst_w), dtype=np.float32)
        dx[valid_warped_mask] = warped_float[valid_warped_mask, 1] - ideal_float[valid_warped_mask, 1]
        dy[valid_warped_mask] = warped_float[valid_warped_mask, 0] - ideal_float[valid_warped_mask, 0]

        error_dist = np.zeros((self.dst_h, self.dst_w), dtype=np.float32)
        error_dist[valid_warped_mask] = np.sqrt(dx[valid_warped_mask]**2 + dy[valid_warped_mask]**2)
        
        if np.any(valid_warped_mask):
            max_val = np.percentile(error_dist[valid_warped_mask], 80)
        else:
            max_val = 0

        if max_val > 0:
            error_intensity = np.clip(error_dist / max_val * 255.0, 0, 255).astype(np.uint8)
            dist_display = cv2.applyColorMap(error_intensity, cv2.COLORMAP_JET)
            dist_display[~valid_warped_mask] = 0
        else:
            error_intensity = np.zeros((self.dst_h, self.dst_w), dtype=np.uint8)
            dist_display = np.zeros((self.dst_h, self.dst_w, 3), dtype=np.uint8)
        cv2.imshow("error_distance", dist_display)

        error_angle = np.zeros((self.dst_h, self.dst_w), dtype=np.float32)
        error_angle[valid_warped_mask] = np.arctan2(dy[valid_warped_mask], dx[valid_warped_mask])
        
        hsv_error = np.zeros((self.dst_h, self.dst_w, 3), dtype=np.uint8)
        hsv_error[valid_warped_mask, 0] = ((error_angle[valid_warped_mask] + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
        hsv_error[valid_warped_mask, 1] = 255
        hsv_error[valid_warped_mask, 2] = error_intensity[valid_warped_mask]
        
        angle_display = cv2.cvtColor(hsv_error, cv2.COLOR_HSV2BGR)
        cv2.imshow("error_angle", angle_display)
        
        vector_display = cv2.cvtColor(cv2.warpPerspective(display_img, M, (self.dst_w, self.dst_h)), cv2.COLOR_BGR2GRAY)
        vector_display = cv2.cvtColor(vector_display, cv2.COLOR_GRAY2BGR)
        
        step = 20
        total_valid = np.sum(valid_warped_mask)
        for y in range(0, self.dst_h, step):
            for x in range(0, self.dst_w, step):
                if valid_warped_mask[y, x]:
                    vx, vy = int(dx[y, x]), int(dy[y, x])
                    err = np.sqrt(vx**2 + vy**2)
                    if err > 0.5:
                        color = (0, 255, 0) if err < 10 else (0, 255, 255)
                        cv2.arrowedLine(vector_display, (x, y), (x + vx, y + vy), color, 1, tipLength=0.3)
        
        if total_valid > 0:
            prec_under_10 = np.sum(error_dist[valid_warped_mask] < 10) / total_valid * 100
            cv2.putText(vector_display, f"Under 10px: {prec_under_10:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("error_vectors", vector_display)

        # 最適化 ('o'キー)
        if key == ord('o'):
            learning_rate = 5.0
            eps = 1.0
            grad = np.zeros_like(self.points)
            
            def get_loss(pts):
                Mat = cv2.getPerspectiveTransform(pts, self.dst_pts)
                W = cv2.warpPerspective(self.uv_img, Mat, (self.dst_w, self.dst_h))
                v_m = np.any(W > 0, axis=2)
                v_m = cv2.morphologyEx(v_m.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
                if not np.any(v_m): return 1e10
                W_f = W.astype(np.float32) * 255.0
                g_f = self.gradient_img.astype(np.float32)
                DX, DY = W_f[:, :, 1] - g_f[:, :, 1], W_f[:, :, 0] - g_f[:, :, 0]
                dist = np.sqrt(DX[v_m]**2 + DY[v_m]**2)
                threshold = 10.0
                loss_p = np.where(dist < threshold, dist**2, threshold * 2 * dist - threshold**2 + 500.0)
                return np.mean(loss_p) + np.mean(~v_m) * 5000.0

            current_loss = get_loss(self.points)
            for i in range(4):
                for j in range(2):
                    points_plus = self.points.copy()
                    points_plus[i, j] += eps
                    grad[i, j] = (get_loss(points_plus) - current_loss) / eps
            
            if self.dragging_idx == -1:
                step_v = learning_rate * grad
                for i in range(4):
                    mag = np.linalg.norm(step_v[i])
                    if mag > 2.0: step_v[i] = (step_v[i] / mag) * 2.0
                self.points -= step_v
                self.points[:, 0] = np.clip(self.points[:, 0], 0, self.uv_img.shape[1] - 1)
                self.points[:, 1] = np.clip(self.points[:, 1], 0, self.uv_img.shape[0] - 1)

        if key == ord('s'):
            np.save("points.npy", self.points)
            print("Points saved to points.npy")

def calculate_depth(phase_frames):
    # 下位互換性のために残すが、メインループでの使用はクラス推奨
    analyzer = DepthAnalyzer()
    analyzer.update(phase_frames, -1)
