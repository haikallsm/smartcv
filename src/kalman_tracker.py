import cv2
import numpy as np

class kalman_filter:
    def __init__(self, dt=0.005, process_noise=1e-2, meas_noise=1.0, gate_threshold=9.21):
        self.dt = float(dt)
        self.gate_threshold = float(gate_threshold) 
        self.kalman = cv2.KalmanFilter(4, 2)

        dt = self.dt

        self.kalman.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * np.float32(process_noise)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * np.float32(meas_noise)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False

    def init(self, cx, cy, vx=0, vy=0):
        st= np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(vx)], [np.float32(vy)]], np.float32)
        self.kalman.statePost = st.copy()
        self.kalman.statePre = st.copy()
        self.initialized = True

    def predict(self):
        pred = self.kalman.predict()
        return int(pred[0]), int(pred[1])
    
    def mahalanobis_sq(self, cx, cy):
        Z = np.array([[np.float32(cx)], [np.float32(cy)]], np.float32)
        H = self.kalman.measurementMatrix
        x_pre = self.kalman.statePre
        innov = Z - H.dot(x_pre)
        P_pre = self.kalman.errorCovPre
        R = self.kalman.measurementNoiseCov
        S = H.dot(P_pre).dot(H.T) + R
        try:
            s_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(S)
        d2 = float((innov.T.dot(s_inv).dot(innov))[0, 0])
        return d2
    
    def correct(self, cx, cy):
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]], np.float32)
        corrected = self.kalman.correct(measurement)
        return float(corrected[0]), float(corrected[1])
    
    def update(self, detection=None):
        pred_x, pred_y = self.predict()
        if detection is None:
            return pred_x, pred_y, False
        
        cx, cy = float(detection[0]), float(detection[1])
        d2 = self.mahalanobis_sq(cx, cy)
        if d2 <= self.gate_threshold:
            x, y = self.correct(cx, cy)
            return x, y, True
        else:
            return pred_x, pred_y, False