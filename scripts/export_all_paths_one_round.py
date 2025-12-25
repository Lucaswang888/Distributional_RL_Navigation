import os, sys, json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import env_visualizer

json_file = os.path.join(ROOT, "experiment_data", "exp_data_2025-12-19-19-52-02.json")
episode_id = 7

with open(json_file, "r") as f:
    exp = json.load(f)

# 选最多4个方法
agents = ["DQN", "adaptive_IQN", "BA", "APF"]

base_episode = exp[agents[0]]["ep_data"][episode_id]  # 用第一个方法的episode加载环境
ev = env_visualizer.EnvVisualizer(draw_traj=True)
ev.load_episode(base_episode)

all_actions = {}
for a in agents:
    ep = exp[a]["ep_data"][episode_id]
    all_actions[a] = ep["robot"]["action_history"]

# 关键：only_ep_actions=False，并传 all_actions
ev.draw_trajectory(only_ep_actions=False, all_actions=all_actions)
print("Saved: trajectory_test.png")