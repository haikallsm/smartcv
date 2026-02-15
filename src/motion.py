import cv2
import numpy as np

class motionDetect:
    def __init__(self, history=500, var_threshold=50, detect_shadows=True, min_motion_area=1000, blur_ksize=5):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
        self.min_motion_area = int(min_motion_area)
        self.blur_ksize = int(blur_ksize) if blur_ksize % 2 == 1 else int(blur_ksize) + 1

    def detect(self, frame):
        fg_mask = self.subtractor.apply(frame)

        fg_mask = cv2.medianBlur(fg_mask, self.blur_ksize)
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        motion_pixels = cv2.countNonZero(fg_mask)
        motion_detected = motion_pixels > self.min_motion_area

        contours_data = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

        motion_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

        return motion_detected, fg_mask, motion_contours