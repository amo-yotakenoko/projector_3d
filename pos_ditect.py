import cv2
import numpy as np
import math
import phase_shifting as ps


def normalized_phis(imgs):
    
    # _, imgs = ps.phase_shift_set(direction, k)
    # _, imgs = ps.phase_shift_set(direction, k)
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

    A = np.sqrt(3) * (I1 - I3)
    B = 2 * I2 - I1 - I3
    confidence = np.sqrt(A*A + B*B)
    confidence /= (I1 + I2 + I3 + 1e-6)

    
    cv2.imshow(f"phase", result)
    cv2.waitKey(1)
    return result,confidence




def get_phase_img(direction, phase_frames):
        

    result,confidence=normalized_phis(phase_frames[1])

    # cv2.imshow(f"step0", normalized_phis[1])
    confidences = []

    for low, high in [(1, 2**(i+1)) for i in range(8-1)]:

    # for low, high in zip(
    #     [1,2,4,8,16,32,64],
    #     [2,4,8,16,32,64,128]
    # ):
        # result, confidence = normalized_phis(1, direction)
        if not high in phase_frames:
            break


        phi, conf = normalized_phis(phase_frames[high])

        ratio = high / low

        k = np.round(result * ratio - phi)

        result = (k + phi) / ratio

        confidences.append(conf)

        
        cv2.imshow(f"result", result)
        
        # cv2.imshow(f"confidence", confidence)

        cv2.waitKey(1)
        # print(f"low={low}, high={high}, ratio={ratio}, k={k}")
        
        # cv2.destroyAllWindows()

    confidence = np.mean(confidences, axis=0)
    
    # Improved thresholding: use a combination of absolute and relative thresholds
    avg_conf = np.mean(confidence)
    std_conf = np.std(confidence)
    
    # Thresholding logic: pixels with confidence significantly lower than average are noise
    thresh = max(0.01, avg_conf - 0.5 * std_conf)
    mask = np.where(confidence > thresh, 255, 0).astype(np.uint8)

    # Apply morphological cleaning to remove small noise dots
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = np.where(mask > 0, result, 0)
    
    cv2.imshow(f"confidence_mask", mask)
    cv2.imshow(f"result", result)
    cv2.waitKey(1)
    cv2.destroyAllWindows()


    return result


def get_projection_img(phase_frames):
    v_img=get_phase_img("v",phase_frames["v"])
    h_img=get_phase_img("h",phase_frames["h"])
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





