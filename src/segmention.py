import cv2
import numpy as np
from kalman_tracker import kalman_filter

class segmentation:
    def __init__(self, lower_hsv=(0, 50, 50), upper_hsv=(10, 255, 255), min_area=200, blur_ksize=5, morph_kernel=5):
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.min_area = int(min_area)
        self.blur_ksize = int(blur_ksize) if blur_ksize % 2 == 1 else int(blur_ksize) + 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))

    def hsv_segment(self,frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (self.blur_ksize, self.blur_ksize), 0)
        
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours_data= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

        if not contours:
            return None
        
        best = max(contours, key=cv2.contourArea)
        area = int(cv2.contourArea(best))
        if area < self.min_area:
            return None
        
        M = cv2.moments(best)
        if M.get('m00', 0) == 0:
            return None
        cx = float(M['m10'] / M['m00'])
        cy = float(M['m01'] / M['m00'])
        x, y, w, h = cv2.boundingRect(best)
        return {'cx': cx, 'cy': cy, 'area': area, 'bbox': (x, y, w, h), 'mask': mask, 'contour': best}
    
def track_frame(frame, segmenter: segmentation, tracker: kalman_filter):
    det = segmenter.segment(frame)
    if det is None:
        x, y, used = tracker.update(None)
        return x, y, used, None
    
    cx, cy = det['cx'], det['cy']
    x, y, used = tracker.update((cx, cy))
    return x, y, used, det