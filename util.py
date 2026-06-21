import cv2
import numpy as np


def safe_vstack(images):
    """横幅がバラバラの画像リストを、1枚目の横幅に合わせて安全に縦結合する関数"""
    # Noneや空のリストを除外
    valid_images = [img for img in images if img is not None]

    if not valid_images:
        return None

    def normalize_image(img):
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0 and img.min() >= 0.0:
                return np.clip(img * 255.0, 0, 255).astype(np.uint8)
            return np.clip(img, 0, 255).astype(np.uint8)
        if img.dtype != np.uint8:
            return np.clip(img, 0, 255).astype(np.uint8)
        return img

    # 1枚目の画像の横幅（width）を基準にする
    base_width = normalize_image(valid_images[0]).shape[1]

    resized_images = []
    for img in valid_images:
        img = normalize_image(img)
        # 横幅が基準と異なる場合だけリサイズを実行
        if img.shape[1] != base_width:
            # 縦横比は崩れますが、横幅をbase_widthに強制統一、縦幅は元のまま
            img_resized = cv2.resize(
                img, (base_width, img.shape[0]), interpolation=cv2.INTER_AREA
            )
            resized_images.append(img_resized)
        else:
            resized_images.append(img)

    # 横幅が完全に揃ったので、通常のnp.vstackで結合して返す
    return np.vstack(resized_images)