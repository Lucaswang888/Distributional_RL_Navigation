import os, sys, json

# === 1) 计算项目根目录（无论从哪里运行都稳定）===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) \
    if "__file__" in globals() else os.path.abspath(".")
sys.path.insert(0, ROOT)

import env_visualizer

# === 2) 指定你的实验 JSON（用绝对路径，避免 FileNotFound）===
filename = os.path.join(ROOT, "experiment_data", "exp_data_2025-12-19-19-52-02.json")

episode_id = 300
agent = "APF"

with open(filename, "r") as f:
    exp_data = json.load(f)

episode = exp_data[agent]["ep_data"][episode_id]

# === 3) 静态绘制整条轨迹并保存图像 ===
ev = env_visualizer.EnvVisualizer(draw_traj=True)
ev.load_episode(episode)
ev.draw_trajectory()  # 默认会保存 trajectory_test.png（在运行脚本的工作目录下）
print("Saved: trajectory_test.png")