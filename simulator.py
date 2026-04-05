"""
窜天猴（持续动力型）飞行轨迹模拟
二维平面，考虑变质量、推力、重力和空气阻力（速度平方模型）
支持自由设置发射角度（度）
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys

# ========== 参数设置 ==========
m0 = 0.015          # 初始总质量 [kg]
mp = 0.004          # 推进剂质量 [kg]
tb = 1.0            # 燃烧时间 [s]
mu = mp / tb        # 质量流率 [kg/s]
u_exhaust = 600     # 排气相对速度 [m/s]
k = 3.46e-5         # 空气阻力常数 [kg/m]
g = 9.8             # 重力加速度 [m/s²]

# ========== 角度获取 ==========
def get_angle():
    """从命令行参数或用户输入获取发射角度（度）"""
    if len(sys.argv) > 1:
        # 从命令行参数获取，例如：python rocket.py 45
        try:
            theta_deg = float(sys.argv[1])
            print(f"使用命令行参数：发射角度 = {theta_deg}°")
            return theta_deg
        except ValueError:
            print("命令行参数无效，将采用交互输入。")
    
    # 交互输入
    while True:
        try:
            theta_deg = float(input("请输入发射角度（度，例如 45）："))
            if 0 <= theta_deg <= 90:
                return theta_deg
            else:
                print("角度应在 0~90 度之间，请重新输入。")
        except ValueError:
            print("请输入数字。")

theta_deg = get_angle()
theta = np.radians(theta_deg)   # 转为弧度

# 初始状态 [x, y, vx, vy, m]
init_state = [0.0, 0.0, 0.0, 0.0, m0]
t_span = (0.0, 10.0)

# ========== 微分方程定义 ==========
def rocket_ode(t, state):
    x, y, vx, vy, m = state
    v = np.hypot(vx, vy)

    # 质量变化率
    dmdt = -mu if t <= tb else 0.0

    # 推力方向：沿瞬时速度方向（速度极小时按初始发射角）
    if v < 1e-6:
        thrust_dir_x = np.cos(theta)
        thrust_dir_y = np.sin(theta)
    else:
        thrust_dir_x = vx / v
        thrust_dir_y = vy / v

    thrust_mag = u_exhaust * abs(dmdt)
    thrust_x = thrust_mag * thrust_dir_x
    thrust_y = thrust_mag * thrust_dir_y

    # 空气阻力
    drag_x = -k * v * vx
    drag_y = -k * v * vy

    ax = (drag_x + thrust_x) / m
    ay = (drag_y + thrust_y) / m - g

    return [vx, vy, ax, ay, dmdt]

# ========== 落地事件 ==========
def ground_event(t, state):
    return state[1]   # y = 0
ground_event.terminal = True
ground_event.direction = -1

# ========== 求解 ==========
sol = solve_ivp(rocket_ode, t_span, init_state,
                method='RK45', events=ground_event,
                dense_output=True, max_step=0.01)

# ========== 输出结果 ==========
if sol.t_events[0].size > 0:
    tf = sol.t_events[0][0]
    xf = sol.sol(tf)[0]
    vxf = sol.sol(tf)[2]
    vyf = sol.sol(tf)[3]
    vf = np.hypot(vxf, vyf)
    print(f"\n发射角度: {theta_deg}°")
    print(f"落地时间: {tf:.2f} s")
    print(f"落地水平距离: {xf:.2f} m")
    print(f"落地速度: {vf:.2f} m/s")
else:
    print("未在模拟时间内落地")

# ========== 绘图（支持中文） ==========
plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体，Windows
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示

t = sol.t
x, y, vx, vy, m = sol.y
v = np.hypot(vx, vy)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, y)
plt.xlabel("水平距离 x (m)")
plt.ylabel("高度 y (m)")
plt.title(f"轨迹 (发射角 {theta_deg}°)")
plt.grid(True)
plt.axis("equal")

plt.subplot(1, 3, 2)
plt.plot(t, v)
plt.xlabel("时间 t (s)")
plt.ylabel("速度 v (m/s)")
plt.title("速度变化")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(t, m)
plt.xlabel("时间 t (s)")
plt.ylabel("质量 m (kg)")
plt.title("质量变化")
plt.grid(True)

plt.tight_layout()
plt.show()