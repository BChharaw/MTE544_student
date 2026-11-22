import numpy as np

# Extended Kalman Filter
# state x = [x, y, th, w, v, vdot], measurement z = [v, w, ax, ay] (odom v,w; IMU ax,ay)

class kalman_filter:

    # P: state covariance, Q: process noise covariance, R: measurement noise covariance
    def __init__(self, P,Q,R, x, dt):

        self.P=np.array(P, dtype=float)
        self.Q=np.array(Q, dtype=float)
        self.R=np.array(R, dtype=float)
        self.x=np.array(x, dtype=float)
        self.dt = dt

    # Prediction, propagate the state with f(x) and the covariance
    def predict(self):

        self.A = self.jacobian_A()
        self.motion_model()
        self.C = self.jacobian_H()

        self.P= np.dot( np.dot(self.A, self.P), self.A.T) + self.Q

    # Update correct the prediction with the measurement
    def update(self, z):

        z=np.array(z, dtype=float)

        S = np.dot( np.dot(self.C, self.P), self.C.T) + self.R  # innovation covariance
        kalman_gain= np.dot( np.dot(self.P, self.C.T), np.linalg.inv(S))  # weight of measurement vs prediction

        surprise_error = z - self.measurement_model()

        self.x= self.x + np.dot(kalman_gain, surprise_error)
        self.P= np.dot( (np.eye(self.A.shape[0]) - np.dot(kalman_gain, self.C)), self.P)


    # h(x): expected sensor readings; ay is the centripetal acceleration v*w
    def measurement_model(self):
        x, y, th, w, v, vdot = self.x
        return np.array([
            v,# v
            w,# w
            vdot, # ax
            v*w, # ay
        ])

    # f(x): unicycle kinematics, constant w and vdot over dt
    def motion_model(self):

        x, y, th, w, v, vdot = self.x
        dt = self.dt

        self.x = np.array([
            x + v * np.cos(th) * dt,
            y + v * np.sin(th) * dt,
            np.arctan2(np.sin(th + w * dt), np.cos(th + w * dt)),  # wrapped to [-pi, pi]
            w,
            v  + vdot*dt,
            vdot,
        ])



    # A = df/dx
    def jacobian_A(self):
        x, y, th, w, v, vdot = self.x
        dt = self.dt

        return np.array([
            #x, y,               th, w,             v, vdot
            [1, 0, -v*np.sin(th)*dt, 0, np.cos(th)*dt,  0],
            [0, 1,  v*np.cos(th)*dt, 0, np.sin(th)*dt,  0],
            [0, 0,                1, dt,           0,  0],
            [0, 0,                0, 1,            0,  0],
            [0, 0,                0, 0,            1,  dt],
            [0, 0,                0, 0,            0,  1 ]
        ])


    # C = dh/dx, only the ay row is nonlinear
    def jacobian_H(self):
        x, y, th, w, v, vdot=self.x
        return np.array([
            #x, y,th, w, v,vdot
            [0,0,0  , 0, 1, 0], # v
            [0,0,0  , 1, 0, 0], # w
            [0,0,0  , 0, 0, 1], # ax
            [0,0,0  , v, w, 0], # ay
        ])

    def get_states(self):
        return self.x
