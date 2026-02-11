import cv2
from src.motion import motionDetect
from src.segmentation import segmentation
from src.kalman_tracker import kalman_filter

def main():
    motion_detector = motionDetect(
        history=500,
        var_threshold=50,
        detect_shadows=True,
        min_motion_area=1000,
        blur_ksize=5
    )

    segmenter = segmentation(
        lower_hsv=(0, 50, 50),
        upper_hsv=(10, 255, 255),
        min_area=200,
        blur_ksize=5,
        morph_kernel=5
    )

    tracker =kalman_filter(
        dt=0.1,
        process_noise=1e-2,
        meas_noise=1.0,
        gate_threshold=9.21
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Error: Could not open camera')
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame =cap.read()
        if not ret:
            break

        motion_mask, motion_detected, motion_contours = motion_detector.detect_motion(frame)

        det = segmenter.hsv_segment(frame)

        if motion_detected and det is not None:
            if not tracker.initialized:
                tracker.init(det['cx'], det['cy'])
                x, y = det['cx'], det['cy']
                used = True
            else:
                x, y, used = tracker.update((det['cx'], det['cy']))

            x_bb, y_bb, w_bb, h_bb = det['bbox']
            cv2.rectangle(frame, (x_bb, y_bb), (x_bb + w_bb, y_bb + h_bb), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"motion + HSV | area: {det['area']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        else: 
            if tracker.initialized:
                x, y, used = tracker.update(None)
            else:
                x, y, used = 0, 0, False

        cv2.imshow("motion mask", motion_mask)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    