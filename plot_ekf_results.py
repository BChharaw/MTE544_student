"""
Lab 3 report figures: measured vs. EKF-estimated signals for each Q/R case.

Run from the lab folder (the one holding the "Q0.5 R0.5", "Q0.5_R0.15", ... run folders):
    python3 plot_ekf_results.py            # writes report/*.png and prints the metrics table
    python3 plot_ekf_results.py <lab_dir>  # or point it at the folder

Each run folder holds robotPose.csv with columns
    imu_ax, imu_ay, kf_ax, kf_ay, kf_vx, kf_w, kf_x, kf_y, stamp
where imu_* are the IMU measurements and kf_* are the EKF outputs
(kf_ax = vdot state, kf_ay = v*w predicted centripetal acceleration).
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FIG_OUT", os.path.join(LAB, "report"))
os.makedirs(OUT, exist_ok=True)

# folder name -> (Q scalar, R scalar); Q = q*I6, R = r*I4
RUNS = {
    "Q0.5_R0.05": (0.5, 0.05),
    "Q0.5_R0.15": (0.5, 0.15),
    "Q0.5 R0.25": (0.5, 0.25),
    "Q0.5 R0.5": (0.5, 0.5),
    "Q0.5 R0.75": (0.5, 0.75),
    "Q0.25_R0.15": (0.25, 0.15),
    "Q0.6 R0.15": (0.6, 0.15),
    "Q=0.75_R=0.15": (0.75, 0.15),
    "Q0.25 R0.75": (0.25, 0.75),
    "Q0.1_R1.0": (0.1, 1.0),
}
FINAL = "Q0.6 R0.15"
T_SS = 10.0          # metrics use t > 10 s (spiral has settled into its orbit)
WIN = (21.0, 27.0)   # time window shown in the time-series panels

MEAS = dict(color="#8a8984", marker="o", ms=3.2, lw=0.8, ls=":", mfc="white", mew=0.8)
EST = dict(color="#2a78d6", marker="s", ms=3.0, lw=1.6, ls="-", markevery=3)
EST2 = dict(color="#eb6834", marker="^", ms=3.4, lw=1.6, ls="-", markevery=3)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.35,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 220,
})


def load(name):
    d = np.genfromtxt(os.path.join(LAB, name, "robotPose.csv"), delimiter=",", skip_header=1)[:, :9]
    t = (d[:, 8] - d[0, 8]) * 1e-9
    keys = ["ax", "ay", "kax", "kay", "v", "w", "x", "y"]
    return t, {k: d[:, i] for i, k in enumerate(keys)}


def steady_gain(q, r, v, w, dt=0.1):
    """Steady-state Kalman gain of the [w, v, vdot] block (x, y, th do not feed back into it)."""
    A = np.array([[1, 0, 0], [0, 1, dt], [0, 0, 1]])
    C = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [v, w, 0]])   # z = [v, w, ax, ay]
    P, Q, R = np.eye(3), q * np.eye(3), r * np.eye(4)
    for _ in range(3000):
        P = A @ P @ A.T + Q
        K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
        P = (np.eye(3) - K @ C) @ P
    return K, C


def metrics(name):
    q, r = RUNS[name]
    t, s = load(name)
    m = t > T_SS
    jitter = lambda a: np.mean(np.abs(np.diff(a[m])))
    K, C = steady_gain(q, r, s["v"][m].mean(), s["w"][m].mean())
    return dict(
        q=q, r=r, qr=q / r, T=t[-1],
        rmse_ax=np.sqrt(np.mean((s["ax"] - s["kax"])[m] ** 2)),
        rel_ax=np.sqrt(np.mean((s["ax"] - s["kax"])[m] ** 2)) / s["ax"][m].std(),
        pass_ax=jitter(s["kax"]) / jitter(s["ax"]),
        K_ax=K[2, 2], K_ay=(C @ K)[3, 3],
        rmse_ay=np.sqrt(np.mean((s["ay"] - s["kay"])[m] ** 2)),
        std_ay=s["ay"][m].std(), std_kay=s["kay"][m].std(),
        mean_ay=s["ay"][m].mean(), mean_kay=s["kay"][m].mean(),
        v=s["v"][m].mean(), w=s["w"][m].mean(),
    )


def ax_panel(a, name, title, legend=False):
    t, s = load(name)
    k = (t >= WIN[0]) & (t <= WIN[1])
    a.plot(t[k], s["ax"][k], label=r"Measured $a_x$ (IMU)", **MEAS)
    a.plot(t[k], s["kax"][k], label=r"Estimated $\hat{a}_x=\hat{\dot v}$ (EKF)", **EST)
    a.set_title(title)
    a.set_xlabel("Time (s)")
    a.set_ylabel(r"$a_x$ (m/s$^2$)")
    a.set_xlim(*WIN)
    a.set_ylim(-2.2, 2.0)


# ---- Figure 1: R sweep (Q = 0.5) on top, Q sweep (R = 0.15) below ----
ROW_R = ["Q0.5_R0.05", "Q0.5 R0.5", "Q0.5 R0.75"]
ROW_Q = ["Q0.25_R0.15", "Q0.5_R0.15", "Q=0.75_R=0.15"]
fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.2), sharey=True)
for a, name in zip(axs[0], ROW_R):
    m = metrics(name)
    ax_panel(a, name, f"$R={m['r']:g}$ (Q/R = {m['qr']:.2g})\nRMSE {m['rmse_ax']:.2f} m/s$^2$, $K_{{a_x}}$ = {m['K_ax']:.2f}")
for a, name in zip(axs[1], ROW_Q):
    m = metrics(name)
    ax_panel(a, name, f"$Q={m['q']:g}$ (Q/R = {m['qr']:.2g})\nRMSE {m['rmse_ax']:.2f} m/s$^2$, $K_{{a_x}}$ = {m['K_ax']:.2f}")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=7.5, framealpha=0.95)
for a in axs[:, 1:].flat:
    a.set_ylabel(r"$a_x$ (m/s$^2$)")
fig.text(0.004, 0.66, r"Vary $R$, $Q=0.5\,I_6$", rotation=90, va="center", fontsize=8, fontweight="bold")
fig.text(0.004, 0.24, r"Vary $Q$, $R=0.15\,I_4$", rotation=90, va="center", fontsize=8, fontweight="bold")
fig.suptitle(r"Spiral: measured vs. estimated longitudinal acceleration $a_x$ for each covariance case",
             fontsize=9.5, fontweight="bold")
fig.tight_layout(rect=(0.02, 0, 1, 0.925), h_pad=0.5, w_pad=0.5)
fig.savefig(os.path.join(OUT, "fig1_ax_vs_QR.png"))
plt.close(fig)

# ---- Figure 3: final tuning (ay + path) and gain vs Q/R for every run ----
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.55), gridspec_kw=dict(width_ratios=[1.3, 1.0, 1.1]))
t, s = load(FINAL)
k = (t >= WIN[0]) & (t <= WIN[1])
a = axs[0]
a.plot(t[k], s["ay"][k], label=r"Measured $a_y$ (IMU)", **MEAS)
a.plot(t[k], s["kay"][k], label=r"Estimated $\hat v\hat\omega$ (EKF)", **EST2)
a.set_title(r"(a) $a_y$, final tuning")
a.set_xlabel("Time (s)")
a.set_ylabel(r"$a_y$ (m/s$^2$)")
a.set_xlim(*WIN)
a.set_ylim(-1.0, 1.75)
a.legend(loc="upper center", ncol=1, fontsize=6.3, framealpha=0.9, borderpad=0.3)

a = axs[1]
a.plot(s["x"], s["y"], color="#2a78d6", lw=1.4, label=r"Estimated path $(\hat x,\hat y)$")
a.plot(s["x"][0], s["y"][0], "o", color="#1baf7a", ms=6, mec="white", label="Start")
a.plot(s["x"][-1], s["y"][-1], "X", color="#e34948", ms=7, mec="white", label="End")
a.set_title("(b) Estimated path, final tuning")
a.set_xlabel("x (m)")
a.set_ylabel("y (m)")
a.set_xlim(-1.4, 2.4)
a.set_ylim(-3.2, 2.25)
a.set_aspect("equal", adjustable="box")
a.legend(loc="lower center", fontsize=6.0, ncol=1, framealpha=0.9, borderpad=0.3)

a = axs[2]
rows = [metrics(n) for n in RUNS]
qr = np.logspace(-1.1, 1.1, 120)
mf = metrics(FINAL)
a.plot(qr, [steady_gain(x, 1.0, mf["v"], mf["w"])[0][2, 2] for x in qr], "-", color="#2a78d6", lw=1.6,
       label=r"Steady-state gain $K_{a_x}$")
a.plot([m["qr"] for m in rows], [m["pass_ax"] for m in rows], "^", color="#eb6834", ms=5,
       label="Jitter passed (meas.)")
a.plot([m["qr"] for m in rows], [m["rel_ax"] for m in rows], "o", color="#6250d6", ms=4.5, mfc="white", mew=1.2,
       label=r"RMSE / $\sigma(a_x)$ (meas.)")
a.set_xscale("log")
a.set_xlim(0.08, 13)
a.set_ylim(0, 1.5)
a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
a.set_title(r"(c) $a_x$ tracking vs. Q/R (10 runs)")
a.set_xlabel("Q/R (log scale)")
a.set_ylabel("Ratio (–)")
a.legend(loc="upper center", fontsize=6.0, framealpha=0.9, borderpad=0.3)
fig.suptitle(r"Final tuning ($Q=0.6\,I_6$, $R=0.15\,I_4$) and gain trend across all covariance cases", fontsize=9.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=0.8)
fig.savefig(os.path.join(OUT, "fig2_final_tuning.png"))
plt.close(fig)

# ---- metrics table ----
print(f"{'run':15s} {'Q':>5s} {'R':>5s} {'Q/R':>5s} {'RMSEax':>7s} {'RMSE/sd':>8s} {'pass':>5s} {'Kax':>5s} "
      f"{'Kay':>5s} {'RMSEay':>7s} {'sd_ay':>6s} {'sd_kay':>6s} {'m_ay':>6s} {'m_kay':>6s} {'v':>5s} {'w':>5s} {'v/w':>5s}")
for n in sorted(RUNS, key=lambda n: RUNS[n][0] / RUNS[n][1]):
    m = metrics(n)
    print(f"{n:15s} {m['q']:5.2f} {m['r']:5.2f} {m['qr']:5.2f} {m['rmse_ax']:7.3f} {m['rel_ax']:8.2f} {m['pass_ax']:5.2f} "
          f"{m['K_ax']:5.2f} {m['K_ay']:5.2f} {m['rmse_ay']:7.3f} {m['std_ay']:6.3f} {m['std_kay']:6.3f} "
          f"{m['mean_ay']:6.3f} {m['mean_kay']:6.3f} {m['v']:5.2f} {m['w']:5.2f} {m['v']/m['w']:5.2f}")
print("figures written to", OUT)
