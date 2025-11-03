import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

#   DC MOTOR
class DCMotor:
    def __init__(self, K=2.0, T=0.08, dt=0.01):
        self.K = K
        self.T = T
        self.dt = dt
        self.omega = 0.0

    def step(self, u):
        domega = (self.dt / self.T) * (-self.omega + self.K * u)
        self.omega += domega
        return self.omega

#   MF
def trimf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if a < x < b: return (x - a) / (b - a)
    if b <= x < c: return (c - x) / (c - b)
    return 0.0

def trapmf(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if a < x < b: return (x - a) / (b - a)
    if b <= x <= c: return 1.0
    if c < x < d: return (d - x) / (d - c)
    return 0.0

#   FUZZY CONTROLLER
class FuzzySpeedController:
    def __init__(self):
        self.labels = ["NB","NS","Z","PS","PB"]
        self.rule_table = [
            ["NB","NB","NS","Z","Z"],
            ["NB","NS","NS","Z","PS"],
            ["NS","NS","Z","PS","PS"],
            ["Z","Z","PS","PS","PB"],
            ["Z","PS","PS","PB","PB"]
        ]
        self.du_range = (-200,200)

    def fuzzify_error(self,e):
        NB=trapmf(e,-150,-150,-80,-40)
        NS=trimf(e,-80,-40,0)
        Z =trimf(e,-10,0,10)
        PS=trimf(e,0,40,80)
        PB=trapmf(e,40,80,150,150)
        return {"NB":NB,"NS":NS,"Z":Z,"PS":PS,"PB":PB}

    def fuzzify_derror(self,de):
        NB=trapmf(de,-80,-80,-40,-15)
        NS=trimf(de,-40,-20,0)
        Z =trimf(de,-5,0,5)
        PS=trimf(de,0,20,40)
        PB=trapmf(de,15,40,80,80)
        return {"NB":NB,"NS":NS,"Z":Z,"PS":PS,"PB":PB}

    def output_mf(self,label,x):
        if label=="NB": return trapmf(x,-200,-200,-120,-60)
        if label=="NS": return trimf(x,-120,-60,0)
        if label=="Z":  return trimf(x,-10,0,10)
        if label=="PS": return trimf(x,0,60,120)
        if label=="PB": return trapmf(x,60,120,200,200)
        return 0.0

    def infer(self,e,de):
        e_fs=self.fuzzify_error(e)
        de_fs=self.fuzzify_derror(de)
        xs=np.arange(-200,201,1)
        aggregated=np.zeros_like(xs,dtype=float)

        for i,el in enumerate(self.labels):
            for j,dl in enumerate(self.labels):
                fire=min(e_fs[el],de_fs[dl])
                if fire==0: continue
                out=self.rule_table[i][j]
                for k,x in enumerate(xs):
                    mu=min(self.output_mf(out,x),fire)
                    aggregated[k]=max(aggregated[k],mu)
        num=np.sum(xs*aggregated)
        den=np.sum(aggregated)
        return 0 if den==0 else num/den

#   SIMULATION FUNCTION
def run_sim(ref_speed=100,gain=0.05):
    motor=DCMotor()
    controller=FuzzySpeedController()
    u=0
    prev_e=0
    speeds=[]
    for _ in range(200):
        w=motor.step(u)
        e=ref_speed-w
        de=e-prev_e
        du=controller.infer(e,de)
        u=np.clip(u+du*gain,0,100)
        prev_e=e
        speeds.append(w)
    return np.array(speeds)

#   GRAPHIC AND SLIDER
speeds=run_sim()
fig,ax=plt.subplots()
plt.subplots_adjust(bottom=0.3)
(line,)=plt.plot(speeds,label="Speed (rad/s)")
plt.axhline(100,color='r',ls='--',label="Target=100")
plt.legend()
ax.set_xlabel("Step"); ax.set_ylabel("rad/s")

# Slider areas
ax_ref=plt.axes([0.25,0.15,0.65,0.03])
ax_gain=plt.axes([0.25,0.1,0.65,0.03])
s_ref=Slider(ax_ref,"Ref Speed",20,140,valinit=100)
s_gain=Slider(ax_gain,"Gain",0.01,0.2,valinit=0.05)

# Update function
def update(val):
    speeds=run_sim(s_ref.val,s_gain.val)
    line.set_ydata(speeds)
    ax.relim(); ax.autoscale_view()
    fig.canvas.draw_idle()

def update_after_release(event):
    if event.name == "button_release_event":
        speeds = run_sim(s_ref.val, s_gain.val)
        line.set_ydata(speeds)
        ax.relim(); ax.autoscale_view()
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_release_event', update_after_release)

plt.show()

# FUZZY SURFACE ANALYSIS
from mpl_toolkits.mplot3d import Axes3D

def plot_fuzzy_surface():
    controller = FuzzySpeedController()
    e_vals = np.linspace(-150, 150, 50)
    de_vals = np.linspace(-80, 80, 50)

    E, DE = np.meshgrid(e_vals, de_vals)
    DU = np.zeros_like(E)

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            DU[i, j] = controller.infer(E[i, j], DE[i, j])

    fig2 = plt.figure(figsize=(8,6))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf = ax2.plot_surface(E, DE, DU, cmap='viridis', edgecolor='none')
    ax2.set_title("Fuzzy Control Surface (Error vs dError → dU)")
    ax2.set_xlabel("Error (e)")
    ax2.set_ylabel("dError (Δe)")
    ax2.set_zlabel("Output (dU)")
    fig2.colorbar(surf, shrink=0.6, aspect=10)
    plt.show()

plot_fuzzy_surface()