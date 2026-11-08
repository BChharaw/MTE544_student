import numpy as np

# Extended Kalman Filter (EKF) for the TurtleBot.
#
# State vector:        x = [x, y, th, w, v, vdot]
#   x, y  -> position of the robot in the odom frame
#   th    -> heading (yaw) of the robot
#   w     -> angular velocity
#   v     -> linear (forward) velocity
#   vdot  -> linear (forward) acceleration
#
# Measurement vector:  z = [v, w, ax, ay]
#   v, w   -> linear and angular velocity from the odometry (/odom)
#   ax, ay -> linear acceleration in the robot x and y direction from the IMU (/imu)

class kalman_filter:

    # Store the filter quantities
    #   P  -> state covariance (uncertainty of the estimate), 6x6
    #   Q  -> process noise covariance (how much we trust the motion model), 6x6
    #   R  -> measurement noise covariance (how much we trust the sensors), 4x4
    #   x  -> initial state estimate, 6 values
    #   dt -> time step between filter iterations (s)
    def __init__(self, P,Q,R, x, dt):

        self.P=np.array(P, dtype=float)
        self.Q=np.array(Q, dtype=float)
        self.R=np.array(R, dtype=float)
        self.x=np.array(x, dtype=float)
        self.dt = dt

    # Prediction step:
    #   x_bar = f(x)            (nonlinear motion model)
    #   P_bar = A P A^T + Q     (A = Jacobian of f, evaluated at the previous state)
    def predict(self):

        # Linearize the motion model around the previous state estimate
        self.A = self.jacobian_A()

        # Propagate the state forward through the nonlinear motion model
        self.motion_model()

        # Linearize the measurement model around the predicted state
        self.C = self.jacobian_H()

        # Predicted covariance: previous uncertainty propagated through A plus process noise
        self.P= np.dot( np.dot(self.A, self.P), self.A.T) + self.Q

    # Update (correction) step:
    #   S = C P C^T + R              (innovation covariance)
    #   K = P C^T S^-1               (Kalman gain)
    #   x = x + K (z - h(x))         (correct the state with the measurement residual)
    #   P = (I - K C) P              (reduce the uncertainty after the correction)
    def update(self, z):

        z=np.array(z, dtype=float)

        # Innovation covariance: predicted measurement uncertainty plus sensor noise
        S = np.dot( np.dot(self.C, self.P), self.C.T) + self.R

        # Kalman gain: decides how much we rely on the measurements vs. the prediction
        kalman_gain= np.dot( np.dot(self.P, self.C.T), np.linalg.inv(S))

        # Measurement residual: real measurement minus the expected measurement
        surprise_error = z - self.measurement_model()

        # Corrected state estimate
        self.x= self.x + np.dot(kalman_gain, surprise_error)

        # Corrected covariance
        self.P= np.dot( (np.eye(self.A.shape[0]) - np.dot(kalman_gain, self.C)), self.P)


    # Measurement model h(x): what the sensors should read given the current state
    #   v  -> odom linear velocity is the state v
    #   w  -> odom angular velocity is the state w
    #   ax -> IMU forward acceleration is the tangential acceleration vdot
    #   ay -> IMU lateral acceleration is the centripetal acceleration v*w
    def measurement_model(self):
        x, y, th, w, v, vdot = self.x
        return np.array([
            v,# v
            w,# w
            vdot, # ax
            v*w, # ay
        ])

    # Motion model f(x): unicycle kinematics with constant angular velocity and
    # constant linear acceleration over one time step dt
    def motion_model(self):

        x, y, th, w, v, vdot = self.x
        dt = self.dt

        self.x = np.array([
            x + v * np.cos(th) * dt,  # move along the heading direction
            y + v * np.sin(th) * dt,
            # integrate the angular velocity, wrapped to [-pi, pi] so the
            # angular error in utilities.py stays valid after many turns
            np.arctan2(np.sin(th + w * dt), np.cos(th + w * dt)),
            w,                        # angular velocity assumed constant
            v  + vdot*dt,             # integrate the linear acceleration
            vdot,                     # linear acceleration assumed constant
        ])



    # Jacobian of the motion model A = df/dx (6x6), evaluated at the current state.
    #   dx/dth = -v sin(th) dt     dx/dv = cos(th) dt
    #   dy/dth =  v cos(th) dt     dy/dv = sin(th) dt
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


    # Jacobian of the measurement model C = dh/dx (4x6): sensors in the rows,
    # states in the columns. Only ay = v*w is nonlinear:
    #   d(ay)/dw = v      d(ay)/dv = w
    def jacobian_H(self):
        x, y, th, w, v, vdot=self.x
        return np.array([
            #x, y,th, w, v,vdot
            [0,0,0  , 0, 1, 0], # v
            [0,0,0  , 1, 0, 0], # w
            [0,0,0  , 0, 0, 1], # ax
            [0,0,0  , v, w, 0], # ay
        ])

    # Return the current state estimate [x, y, th, w, v, vdot]
    def get_states(self):
        return self.x
