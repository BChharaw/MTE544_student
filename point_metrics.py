import csv, math, os

# Computes agility / accuracy / overshoot metrics for the point-controller runs
# (goal = [-1.0, -1.0]) from the logged linear.csv, angular.csv and robot_pose.csv

GOAL=[-1.0, -1.0]
BAND=0.05   # 5% settling band (the robot stops at 0.05 m, i.e. 3.5% of the 1.414 m step, so a 2% band is never reached)

RUNS=[
    ("P_005",   "P",   "Kp=0.05 (lin & ang)", "P_005"),
    ("P_065",   "P",   "Kp=0.65 (lin & ang)", "P_065"),
    ("P_point", "P",   "Kp_lin=0.2, Kp_ang=0.8", "P_point"),
    ("PID",     "PID", "lin 0.2/0.5/0.2, ang 0.8/0.6/0.2", "PID"),
    ("PID_06",  "PID", "lin 0.6/0.5/0.2, ang 1.8/0.6/0.2", "PID_self.controller=controller(klp=0.6, klv=0.5, kap=1.8, kav=0.6)"),
    ("PID_08",  "PID", "lin 0.8/0.3/0.2, ang 2.2/0.6/0.2", "PID self.controller=controller(klp=0.8, klv=0.3, kap=2.2, kav=0.6)"),
]


def load(path):
    rows=[]
    with open(path) as file:
        reader=csv.reader(file)
        next(reader)    # skip the header
        for line in reader:
            vals=[float(v) for v in line if v.strip()!=""]
            if vals:
                rows.append(vals)
    return rows


def first_time_below(t, e, level):
    for ti, ei in zip(t, e):
        if abs(ei) <= level:
            return ti
    return None


def settling_time(t, e, level):
    # last time the error is outside the band; None if it never ends inside it
    if abs(e[-1]) > level:
        return None
    for i in range(len(e)-1, -1, -1):
        if abs(e[i]) > level:
            return t[i+1]
    return t[0]


def rise_time(t, e):
    # error goes from 90% to 10% of its initial value (10-90% rise of the response)
    e0=abs(e[0])
    t90=None
    for ti, ei in zip(t, e):
        if t90 is None and abs(ei) <= 0.9*e0:
            t90=ti
        if t90 is not None and abs(ei) <= 0.1*e0:
            return ti - t90
    return None


def analyse(folder):
    lin=load(os.path.join(folder, "linear.csv"))
    ang=load(os.path.join(folder, "angular.csv"))
    pose=load(os.path.join(folder, "robot_pose.csv"))

    t0=lin[0][3]
    t=[(row[3]-t0)/1e9 for row in lin]
    e_lin=[row[0] for row in lin]
    e_ang=[row[0] for row in ang]

    m={}
    # Agility
    m["tr_lin"]=rise_time(t, e_lin)
    m["tr_ang"]=rise_time(t, e_ang)
    m["ts_lin"]=settling_time(t, e_lin, BAND*e_lin[0])
    m["ts_ang"]=settling_time(t, e_ang, BAND*abs(e_ang[0]))
    m["t_goal"]=(pose[-1][3]-t0)/1e9

    # Accuracy (steady-state error at the moment the robot stops)
    x, y=pose[-1][0], pose[-1][1]
    m["ess_lin"]=math.hypot(GOAL[0]-x, GOAL[1]-y)
    m["ess_ang"]=e_ang[-1]

    # Overshoot
    # angular: the error starts negative, so any positive value means the heading went past the target
    m["os_ang"]=max(0.0, max(e_ang))/abs(e_ang[0])*100
    # position: how far x and y went past the -1 m goal, as % of the 1 m step in each axis
    m["os_x"]=max(0.0, GOAL[0]-min(p[0] for p in pose))/abs(GOAL[0])*100
    m["os_y"]=max(0.0, GOAL[1]-min(p[1] for p in pose))/abs(GOAL[1])*100
    # how much the distance to the goal grew while the robot was still turning
    m["e_lin_peak"]=max(e_lin)-e_lin[0]
    return m


def fmt(v, spec, none="n/a"):
    return none if v is None else format(v, spec)


if __name__=="__main__":

    here=os.path.dirname(os.path.abspath(__file__))

    cols=["Run", "Type", "Gains (Kp/Kd/Ki)",
          "t_r lin (s)", "t_r ang (s)", "t_s lin 5% (s)", "t_s ang 5% (s)", "Time to goal (s)",
          "e_ss lin (m)", "e_ss ang (rad)",
          "%OS ang", "%OS x", "%OS y", "Peak e_lin rise (m)"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|"*len(cols))

    for name, type_, gains, folder in RUNS:
        m=analyse(os.path.join(here, folder))
        row=[name, type_, gains,
             fmt(m["tr_lin"], ".2f", "not reached"), fmt(m["tr_ang"], ".2f", "not reached"),
             fmt(m["ts_lin"], ".2f", "not settled"), fmt(m["ts_ang"], ".2f", "not settled"), fmt(m["t_goal"], ".2f"),
             fmt(m["ess_lin"], ".3f"), fmt(m["ess_ang"], "+.3f"),
             fmt(m["os_ang"], ".1f")+"%", fmt(m["os_x"], ".1f")+"%", fmt(m["os_y"], ".1f")+"%",
             fmt(m["e_lin_peak"], ".3f")]
        print("| " + " | ".join(row) + " |")
