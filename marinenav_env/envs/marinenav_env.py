import numpy as np
import scipy.spatial
import marinenav_env.envs.utils.robot as robot
import gym
import json
import copy


class Core:
    def __init__(self, x: float, y: float, clockwise: bool, Gamma: float, rd_state=None):
        self.x = x  # 涡旋核心x坐标
        self.y = y  # 涡旋核心y坐标
        self.clockwise = clockwise  # 旋转方向
        self.Gamma = Gamma  # 环量强度

        # --- 论文特性：动态漂移 ---
        # 为每个涡旋分配一个随机的漂移速度矢量 (vx, vy)
        if rd_state:
            angle = rd_state.uniform(0, 2 * np.pi)
            speed = rd_state.uniform(0.05, 0.15)  # 模拟洋流缓慢演化 [cite: 157]
            self.vx = speed * np.cos(angle)
            self.vy = speed * np.sin(angle)
        else:
            self.vx, self.vy = 0.0, 0.0


class Obstacle:
    def __init__(self, x: float, y: float, r: float):
        self.x = x
        self.y = y
        self.r = r


class MarineNavEnv(gym.Env):
    def __init__(self, seed: int = 0, schedule: dict = None):
        self.seed(seed)
        self.robot = robot.Robot()

        # 定义动作和观测空间
        self.action_space = gym.spaces.Discrete(self.robot.compute_actions_dimension())
        obs_len = 2 + 2 + 2 * self.robot.sonar.num_beams
        self.observation_space = gym.spaces.Box(low=-np.inf * np.ones(obs_len),
                                                high=np.inf * np.ones(obs_len),
                                                dtype=np.float32)

        # 参数初始化
        self.width, self.height = 50, 50
        self.r = 0.5  # 涡核半径 [cite: 137]
        self.v_range = [5, 10]
        self.obs_r_range = [1, 3]
        self.clear_r = 10.0
        self.goal_dis = 2.0
        self.collision_penalty = -50.0
        self.goal_reward = 100.0
        self.timestep_penalty = -1.0
        self.num_cores, self.num_obs = 8, 5
        self.min_start_goal_dis = 25.0
        self.cores, self.obstacles = [], []
        self.total_timesteps, self.episode_timesteps = 0, 0
        self.reset_start_and_goal = True
        self.random_reset_state = True
        self.set_boundary = False

    def seed(self, seed):
        self.sd = seed
        self.rd = np.random.RandomState(seed)
        return [seed]

    def _update_kdtree(self):
        """当涡旋移动后，刷新空间搜索索引"""
        if self.cores:
            centers = np.array([[c.x, c.y] for c in self.cores])
            self.core_centers = scipy.spatial.KDTree(centers)

    def reset(self):
        self.episode_timesteps = 0
        self.cores.clear()
        self.obstacles.clear()

        # 随机起点终点生成
        if self.reset_start_and_goal:
            iteration = 500
            max_dist = 0.0
            while True:
                start = self.rd.uniform(low=2.0, high=self.width - 2.0, size=2)
                goal = self.rd.uniform(low=2.0, high=self.width - 2.0, size=2)
                iteration -= 1
                if np.linalg.norm(goal - start) > max_dist:
                    max_dist = np.linalg.norm(goal - start)
                    self.start, self.goal = start, goal
                if max_dist > self.min_start_goal_dis or iteration == 0:
                    break

        # 生成动态涡流核心 [cite: 177, 185]
        for _ in range(self.num_cores):
            center = self.rd.uniform(low=0, high=self.width, size=2)
            direction = self.rd.binomial(1, 0.5)
            v_edge = self.rd.uniform(self.v_range[0], self.v_range[1])
            Gamma = 2 * np.pi * self.r * v_edge
            self.cores.append(Core(center[0], center[1], direction, Gamma, rd_state=self.rd))
        self._update_kdtree()

        # 生成障碍物
        for _ in range(self.num_obs):
            center = self.rd.uniform(low=5.0, high=self.width - 5.0, size=2)
            r = self.rd.uniform(low=self.obs_r_range[0], high=self.obs_r_range[1])
            self.obstacles.append(Obstacle(center[0], center[1], r))

        if self.obstacles:
            centers = np.array([[obs.x, obs.y] for obs in self.obstacles])
            self.obs_centers = scipy.spatial.KDTree(centers)

        self.reset_robot()
        return self.get_observation()

    def step(self, action):
        self.robot.action_history.append(action)
        dis_before = self.dist_to_goal()
        is_collision = False

        # --- 核心改进：高频更新与子步碰撞检测 ---
        # 即使洋流猛烈，也要在每一小步检测碰撞，防止“穿模”
        for _ in range(self.robot.N):
            # 1. 更新动态洋流位置
            for core in self.cores:
                core.x += core.vx * self.robot.dt
                core.y += core.vy * self.robot.dt
            self._update_kdtree()

            # 2. 获取当前洋流速度并更新机器人 [cite: 272, 308]
            current_velocity = self.get_velocity(self.robot.x, self.robot.y)
            self.robot.update_state(action, current_velocity)
            self.robot.trajectory.append([self.robot.x, self.robot.y])

            # 3. 子步即时检测：一旦触碰立即终止
            if self.check_collision():
                is_collision = True
                break
                # ------------------------------------

        dis_after = self.dist_to_goal()
        obs = self.get_observation()
        reward = self.timestep_penalty + (dis_before - dis_after)

        if is_collision:
            reward += self.collision_penalty
            done, info = True, {"state": "collision"}
        elif self.check_reach_goal():
            reward += self.goal_reward
            done, info = True, {"state": "reach goal"}
        elif self.episode_timesteps >= 1000:
            done, info = True, {"state": "timeout"}
        else:
            done, info = False, {"state": "normal"}

        self.episode_timesteps += 1
        return obs, reward, done, info

    def get_velocity(self, x: float, y: float):
        """基于 Lamb 涡旋模型的矢量合成 [cite: 163, 178]"""
        if not self.cores: return np.zeros(2)
        d, idx = self.core_centers.query(np.array([x, y]), k=len(self.cores))
        if isinstance(idx, (int, np.int64)): idx = [idx]

        v_vel = np.zeros((2, 1))
        for i in idx:
            core = self.cores[i]
            dx, dy = x - core.x, y - core.y
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist < 1e-6: continue
            speed = self.compute_speed(core.Gamma, dist)
            # 切向矢量计算
            v_unit = np.array([[-dy / dist], [dx / dist]])
            if core.clockwise: v_unit *= -1
            v_vel += v_unit * speed
        return v_vel.flatten()

    def compute_speed(self, Gamma: float, d: float):
        # 遵循论文中的涡核模型
        if d <= self.r:
            return (Gamma / (2 * np.pi * self.r ** 2)) * d
        return Gamma / (2 * np.pi * d)

    def check_collision(self):
        """修复穿模：检测机器人边缘与障碍物边缘"""
        if not self.obstacles: return False
        d, idx = self.obs_centers.query(np.array([self.robot.x, self.robot.y]))
        # 这里的 self.robot.r 取自 robot.py 中的 0.8
        return d <= (self.obstacles[idx].r + self.robot.r + 0.05)

    def dist_to_goal(self):
        return np.linalg.norm(self.goal - np.array([self.robot.x, self.robot.y]))

    def check_reach_goal(self):
        return self.dist_to_goal() <= self.goal_dis

    def reset_robot(self):
        if self.random_reset_state:
            self.robot.init_theta = self.rd.uniform(low=0.0, high=2 * np.pi)
            self.robot.init_speed = self.rd.uniform(low=0.0, high=self.robot.max_speed)
        current_v = self.get_velocity(self.start[0], self.start[1])
        self.robot.reset_state(self.start[0], self.start[1], current_velocity=current_v)

    def get_observation(self, for_visualize=False):
        # 集成 robot.py 的声呐检测
        self.robot.sonar_reflection(self.obstacles)
        R_wr, t_wr = self.robot.get_robot_transform()
        R_rw = np.transpose(R_wr)
        t_rw = -R_rw * t_wr
        abs_v = R_rw * np.reshape(self.robot.velocity, (2, 1))
        abs_v.resize((2,));
        abs_v = np.array(abs_v)
        goal_w = np.reshape(self.goal, (2, 1))
        goal_r = R_rw * goal_w + t_rw
        goal_r.resize((2,));
        goal_r = np.array(goal_r)

        sonar_points_r = []
        for point in self.robot.sonar.reflections:
            p = np.reshape(point, (3, 1))
            if p[2] == 0:
                p_r = np.zeros(2)
            else:
                p_r = R_rw * p[:2] + t_rw
                p_r.resize((2,));
                p_r = np.array(p_r)
            sonar_points_r.append(p_r)

        if for_visualize: return abs_v, np.array(sonar_points_r).T, goal_r
        return np.hstack((abs_v, goal_r, np.concatenate(sonar_points_r)))