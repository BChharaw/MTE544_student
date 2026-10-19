# You can use this file to plot the loged sensor data
# Note that you need to modify/adapt it to your own files
# Feel free to make any modifications/additions here
import os
import math

import numpy as np
import matplotlib.pyplot as plt

from utilities import FileReader

MOTIONS = ["line", "circle", "spiral"]
OUT_DIR = "plots"

LASER_ANGLE_MIN = -math.pi


def load(filename):
    headers, rows = FileReader(filename).read_file()
    data = np.array(rows, dtype=float)
    return headers, data


def time_axis(stamps, t0):
    return (stamps - t0) * 1e-9

TIME_WINDOWS = {"line": (2.0, None), "spiral": (None, 15.0)}


def clip_time(data, stamp_col, t0, t_min=None, t_max=None):
    t = (data[:, stamp_col] - t0) * 1e-9
    keep = np.ones(len(data), dtype=bool)
    if t_min is not None:
        keep &= t >= t_min
    if t_max is not None:
        keep &= t <= t_max
    return data[keep]


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)


def plot_odom_xy(motion, data):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(data[:, 0], data[:, 1], "-", color="tab:blue", label="odom path")
    ax.plot(data[0, 0], data[0, 1], "o", color="green", label="start")
    ax.plot(data[-1, 0], data[-1, 1], "s", color="red", label="end")
    ax.axis("equal")
    style_axes(ax, f"Odometry: x vs y ({motion})", "x [m]", "y [m]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"odom_xy_{motion}.png"), dpi=150)
    plt.close(fig)


def plot_odom_time(motion, data, t0):
    t = time_axis(data[:, 3], t0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, data[:, 0], "o-", markersize=2, label="x")
    ax.plot(t, data[:, 1], "s-", markersize=2, label="y")
    ax.plot(t, data[:, 2], "^-", markersize=2, label="th (yaw)")
    style_axes(ax, f"Odometry: x, y, th vs time ({motion})", "time [s]", "x [m], y [m], th [rad]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"odom_time_{motion}.png"), dpi=150)
    plt.close(fig)


def plot_imu(motion, data, t0):
    t = time_axis(data[:, 3], t0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, data[:, 0], "-", label="a_x")
    ax.plot(t, data[:, 1], "--", label="a_y")
    ax.plot(t, data[:, 2], "-.", label="w_z")
    style_axes(ax, f"IMU: a_x, a_y, w_z vs time ({motion})", "time [s]", "acceleration [m/s²], angular velocity [rad/s]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"imu_time_{motion}.png"), dpi=150)
    plt.close(fig)


def plot_laser(motion, data):
    row = data[len(data) // 2]
    ranges = row[:-2]
    angle_increment = row[-2]
    angles = LASER_ANGLE_MIN + np.arange(len(ranges)) * angle_increment

    valid = np.isfinite(ranges) & (ranges > 0)
    x = ranges[valid] * np.cos(angles[valid])
    y = ranges[valid] * np.sin(angles[valid])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, ".", markersize=3, color="tab:purple", label="scan points")
    ax.plot(0, 0, "k^", markersize=10, label="lidar")
    ax.axis("equal")
    style_axes(ax, f"Laser scan in Cartesian frame ({motion})", "x [m]", "y [m]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"laser_xy_{motion}.png"), dpi=150)
    plt.close(fig)


def spiral_start(odom):
    centre = odom[-100:, :2].mean(axis=0)
    radius = np.hypot(odom[:, 0] - centre[0], odom[:, 1] - centre[1])
    i = int(np.argmax(radius < 0.2))
    while i > 0 and radius[i - 1] > radius[i]:
        i -= 1
    return i


def trim_spiral(odom, imu, laser):
    cut = spiral_start(odom)
    cut_stamp = odom[cut, 3]
    print(f"spiral: dropped {cut} odom samples ({(odom[cut, 3] - odom[0, 3]) * 1e-9:.2f} s)")
    odom = odom[cut:]
    imu = imu[imu[:, 3] >= cut_stamp]
    laser = laser[laser[:, -1] >= cut_stamp]
    return odom, imu, laser


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for motion in MOTIONS:
        _, odom = load(f"odom_content_{motion}.csv")
        _, imu = load(f"imu_content_{motion}.csv")
        _, laser = load(f"laser_content_{motion}.csv")

        if motion == "spiral":
            odom, imu, laser = trim_spiral(odom, imu, laser)

        t0 = odom[0, 3]
        t_min, t_max = TIME_WINDOWS.get(motion, (None, None))
        odom = clip_time(odom, 3, t0, t_min, t_max)
        imu = clip_time(imu, 3, t0, t_min, t_max)
        laser = clip_time(laser, -1, t0, t_min, t_max)
        print(f"{motion}: {len(odom)} odom, {len(imu)} imu, {len(laser)} laser samples kept")

        plot_odom_xy(motion, odom)
        plot_odom_time(motion, odom, t0)
        plot_imu(motion, imu, t0)
        plot_laser(motion, laser)
        print(f"Saved plots for {motion}")


if __name__ == "__main__":
    main()
