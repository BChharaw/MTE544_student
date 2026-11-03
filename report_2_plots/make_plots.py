import os
from math import exp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)

BLUE="#2a78d6"; ORANGE="#eb6834"; GREEN="#008300"; INK="#0b0b0b"; MUTED="#52514e"; GRID="#e2e1dc"

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":9.5,"axes.titlesize":9.5,"axes.titleweight":"bold",
    "axes.labelsize":9.5,"xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":9.5,
    "axes.edgecolor":MUTED,"axes.labelcolor":INK,"text.color":INK,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,"axes.axisbelow":True,
    "axes.spines.top":False,"axes.spines.right":False,"lines.linewidth":1.8,
    "legend.frameon":False,"savefig.dpi":300,"figure.dpi":100,
})

P_RUN="P_point"
PID_RUN="PID self.controller=controller(klp=0.8, klv=0.3, kap=2.2, kav=0.6)"
PID_MID="PID_self.controller=controller(klp=0.6, klv=0.5, kap=1.8, kav=0.6)"
P_STYLE={"color":BLUE,"linestyle":"-"}
PID_STYLE={"color":ORANGE,"linestyle":"--"}
P_LABEL="P controller: linear Kp=0.2 | angular Kp=0.8"
PID_LABEL="PID controller: linear Kp=0.8, Kd=0.3, Ki=0.2 | angular Kp=2.2, Kd=0.6, Ki=0.2"

def load_csv(folder,name):
    rows=[]
    for line in open(os.path.join(ROOT,folder,name)).readlines()[1:]:
        values=[float(v) for v in line.split(",") if v.strip()]
        if values:
            rows.append(values)
    return rows

def load_run(folder):
    pose=load_csv(folder,"robot_pose.csv")
    lin=load_csv(folder,"linear.csv")
    ang=load_csv(folder,"angular.csv")
    t0=pose[0][3]
    return {"pose":pose,"lin":lin,"ang":ang,
            "tp":[(r[3]-t0)/1e9 for r in pose],
            "tl":[(r[3]-t0)/1e9 for r in lin],
            "ta":[(r[3]-t0)/1e9 for r in ang]}

def col(rows,i):
    return [r[i] for r in rows]

def label_axes(ax,title,xlabel,ylabel):
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

def save(fig,name):
    fig.savefig(os.path.join(HERE,name+".png"))
    fig.savefig(os.path.join(HERE,name+".pdf"))
    plt.close(fig)

def top_legend(fig,handles,ncol):
    fig.legend(handles=handles,loc="outside upper center",ncol=ncol)

def compare_handles():
    return [plt.Line2D([],[],label=P_LABEL,**P_STYLE),plt.Line2D([],[],label=PID_LABEL,**PID_STYLE)]

def tidy(ax):
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

def point_figure(p,pid):
    fig,ax=plt.subplots(3,3,figsize=(7.2,6.7),layout="constrained")
    for run,style in [(p,P_STYLE),(pid,PID_STYLE)]:
        ax[0][0].plot(run["tl"],col(run["lin"],0),**style)
        ax[0][1].plot(run["ta"],col(run["ang"],0),**style)
        ax[0][2].plot(col(run["pose"],0),col(run["pose"],1),**style)
        ax[1][0].plot(run["tl"],col(run["lin"],1),**style)
        ax[1][1].plot(run["ta"],col(run["ang"],1),**style)
        ax[1][2].plot(run["tp"],col(run["pose"],2),**style)
        ax[2][0].plot(col(run["lin"],0),col(run["lin"],1),**style)
        ax[2][1].plot(col(run["ang"],0),col(run["ang"],1),**style)
        ax[2][2].plot(run["tp"],col(run["pose"],0),**style)
        ax[2][2].plot(run["tp"],col(run["pose"],1),**style)
    ax[0][2].plot(0,0,"o",color=INK,markersize=5,zorder=4)
    ax[0][2].plot(-1,-1,"*",color=INK,markersize=10,zorder=4)
    ax[2][2].text(3.6,0.02,"x",fontsize=10,fontweight="bold")
    ax[2][2].text(0.9,-0.55,"y",fontsize=10,fontweight="bold")
    label_axes(ax[0][0],"Distance to goal","time [s]","e [m]")
    label_axes(ax[0][1],"Heading error","time [s]","e [rad]")
    label_axes(ax[0][2],"Robot path (x vs y)","x [m]","y [m]")
    label_axes(ax[1][0],"Distance error rate","time [s]","ė [m/s]")
    label_axes(ax[1][1],"Heading error rate","time [s]","ė [rad/s]")
    label_axes(ax[1][2],"Robot heading θ","time [s]","θ [rad]")
    label_axes(ax[2][0],"Distance: e vs ė","e [m]","ė [m/s]")
    label_axes(ax[2][1],"Heading: e vs ė","e [rad]","ė [rad/s]")
    label_axes(ax[2][2],"Position x and y","time [s]","position [m]")
    for row in ax:
        for a in row:
            tidy(a)
    for a in [ax[0][0],ax[0][1],ax[1][0],ax[1][1],ax[1][2],ax[2][2]]:
        a.set_xlim(0,17)
    handles=compare_handles()
    handles.append(plt.Line2D([],[],color=INK,marker="o",linestyle="",markersize=5,label="start (0, 0)   and   ★ goal (-1, -1), Robot path panel only"))
    fig.legend(handles=handles,loc="outside upper center",ncol=1,fontsize=9)
    save(fig,"fig1_point_P_vs_PID")

def tuning_row(sub,runs,title):
    ax=sub.subplots(1,3)
    handles=[]
    for folder,label,color,style in runs:
        run=load_run(folder)
        ax[0].plot(col(run["pose"],0),col(run["pose"],1),color=color,linestyle=style)
        keep=[i for i,t in enumerate(run["tl"]) if t<=20]
        ax[1].plot([run["tl"][i] for i in keep],[run["lin"][i][0] for i in keep],color=color,linestyle=style)
        keep=[i for i,t in enumerate(run["ta"]) if t<=20]
        ax[2].plot([run["ta"][i] for i in keep],[run["ang"][i][0] for i in keep],color=color,linestyle=style)
        handles.append(plt.Line2D([],[],color=color,linestyle=style,label=label))
    ax[0].plot(-1,-1,"*",color=INK,markersize=10,zorder=4)
    label_axes(ax[0],"Robot path (★ = goal)","x [m]","y [m]")
    label_axes(ax[1],"Distance to goal","time [s]","e [m]")
    label_axes(ax[2],"Heading error","time [s]","e [rad]")
    ax[1].set_xlim(0,20); ax[2].set_xlim(0,20)
    for a in ax:
        tidy(a)
    sub.legend(handles=handles,loc="outside upper center",ncol=1,fontsize=8.5,title=title,title_fontproperties={"weight":"bold","size":9.5})

def tuning_figure():
    fig=plt.figure(figsize=(7.2,5.6),layout="constrained")
    top,bottom=fig.subfigures(2,1)
    tuning_row(top,[("P_005","linear Kp=0.05 | angular Kp=0.05: reached goal in 101 s (first 20 s shown)",GREEN,":"),
                    (P_RUN,"linear Kp=0.2 | angular Kp=0.8: reached goal in 16.6 s",BLUE,"-"),
                    ("P_065","linear Kp=0.65 | angular Kp=0.65: reached goal in 8.8 s",ORANGE,"--")],"Tuning the P controller: three gain sets")
    tuning_row(bottom,[("PID","linear Kp=0.2, Kd=0.5 | angular Kp=0.8, Kd=0.6: reached goal in 18.1 s",GREEN,":"),
                       (PID_MID,"linear Kp=0.6, Kd=0.5 | angular Kp=1.8, Kd=0.6: reached goal in 9.8 s",BLUE,"-"),
                       (PID_RUN,"linear Kp=0.8, Kd=0.3 | angular Kp=2.2, Kd=0.6: reached goal in 8.1 s",ORANGE,"--")],"Tuning the PID controller: three gain sets (Ki=0.2 in all)")
    save(fig,"fig2_tuning")

def trajectory_figure():
    para=load_run("parabola test"); sig=load_run("sigmoid test")
    ref_p=[[1.5*i/300,(1.5*i/300)**2] for i in range(301)]
    ref_s=[[2.5*i/300,2/(1+exp(-2*2.5*i/300))-1] for i in range(301)]
    fig,ax=plt.subplots(2,2,figsize=(7.2,4.2),layout="constrained")
    for c,(run,ref,title) in enumerate([(para,ref_p,"Parabola"),(sig,ref_s,"Sigmoid")]):
        ax[0][c].plot(col(run["pose"],0),col(run["pose"],1),color=BLUE,linewidth=3,solid_capstyle="round")
        ax[0][c].plot(col(ref,0),col(ref,1),color=INK,linestyle="--",linewidth=1.1)
        label_axes(ax[0][c],title+": path followed (x vs y)","x [m]","y [m]")
        ax[1][c].plot(run["tl"],col(run["lin"],0),color=GREEN,linestyle="-")
        ax[1][c].plot(run["ta"],col(run["ang"],0),color=ORANGE,linestyle="--")
        label_axes(ax[1][c],title+": logged errors","time [s]","error [m or rad]")
        ax[1][c].set_xlim(0,run["tl"][-1])
    handles=[plt.Line2D([],[],color=BLUE,linewidth=3,label="robot path (odometry)"),
             plt.Line2D([],[],color=INK,linestyle="--",linewidth=1.1,label="desired path"),
             plt.Line2D([],[],color=GREEN,label="distance to end of path [m]"),
             plt.Line2D([],[],color=ORANGE,linestyle="--",label="heading error to next path point [rad]")]
    fig.legend(handles=handles,loc="outside upper center",ncol=2,fontsize=9,title="PID controller: linear Kp=0.8, Kd=0.3, Ki=0.2 | angular Kp=2.2, Kd=0.6, Ki=0.2",title_fontproperties={"weight":"bold","size":9.5})
    save(fig,"fig3_trajectories")

if __name__=="__main__":
    p=load_run(P_RUN); pid=load_run(PID_RUN)
    point_figure(p,pid)
    tuning_figure()
    trajectory_figure()
