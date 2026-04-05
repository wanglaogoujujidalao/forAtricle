"""
读取 C++ 模拟生成的 trajectory.csv 文件，绘制窜天猴飞行轨迹。
CSV 格式：t,x,y,v
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

def read_csv(filename):
    """读取 CSV，返回 t, x, y, v 数组"""
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # 跳过表头
    t = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    v = data[:, 3]
    return t, x, y, v

def find_landing(t, y):
    """根据 y>=0 的最后一个点近似落地时间和位置"""
    # 找出 y>=0 的索引
    idx = np.where(y >= 0)[0]
    if len(idx) == 0:
        return None, None, None
    last = idx[-1]
    t_land = t[last]
    x_land = x[last]
    v_land = v[last]
    return t_land, x_land, v_land

def plot_trajectory(t, x, y, v, landing_info):
    """绘图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']   # 使用黑体（Windows）
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 轨迹 (x-y)
    axes[0].plot(x, y, 'b-', linewidth=1.5)
    if landing_info[0] is not None:
        axes[0].scatter(landing_info[1], 0, color='red', zorder=5, label=f"落地 ({landing_info[1]:.1f} m)")
        axes[0].legend()
    axes[0].set_xlabel("水平距离 x (m)")
    axes[0].set_ylabel("高度 y (m)")
    axes[0].set_title("飞行轨迹")
    axes[0].grid(True)
    axes[0].axis("equal")

    # 速度随时间变化
    axes[1].plot(t, v, 'r-', linewidth=1.5)
    if landing_info[0] is not None:
        axes[1].axvline(x=landing_info[0], color='gray', linestyle='--', alpha=0.7)
        axes[1].text(landing_info[0], max(v)*0.9, f"落地 t={landing_info[0]:.2f}s", rotation=90, va='top')
    axes[1].set_xlabel("时间 t (s)")
    axes[1].set_ylabel("速度 v (m/s)")
    axes[1].set_title("速度变化")
    axes[1].grid(True)

    # 高度随时间变化
    axes[2].plot(t, y, 'g-', linewidth=1.5)
    if landing_info[0] is not None:
        axes[2].axvline(x=landing_info[0], color='gray', linestyle='--', alpha=0.7)
    axes[2].set_xlabel("时间 t (s)")
    axes[2].set_ylabel("高度 y (m)")
    axes[2].set_title("高度变化")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 允许通过命令行参数指定 CSV 文件名，默认为 trajectory.csv
    filename = sys.argv[1] if len(sys.argv) > 1 else "trajectory.csv"
    try:
        t, x, y, v = read_csv(filename)
    except Exception as e:
        print(f"读取文件失败: {e}")
        sys.exit(1)

    landing = find_landing(t, y)
    if landing[0] is not None:
        t_land, x_land, v_land = landing
        print(f"从数据中检测到落地: t = {t_land:.3f} s, x = {x_land:.3f} m, v = {v_land:.3f} m/s")
    else:
        print("未检测到落地（可能所有 y < 0）")

    plot_trajectory(t, x, y, v, landing)