import numpy as np
import scipy.spatial
import marinenav_env.envs.utils.robot as robot
import gym
import json
import copy


class Core:
    def __init__(self, x: float, y: float, clockwise: bool, D: float, R: float, rd_state=None):
        self.x = x
        self.y = y
        self.clockwise = clockwise
        # Lamb vortex strength (same role as Gamma in paper1 Rankine model)
        self.D = D
        # Lamb vortex core radius parameter
        self.R = R
        if rd_state:
            angle = rd_state.uniform(0, 2 * np.pi)
            speed = rd_state.uniform(0.05, 0.15)
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

        self.action_space = gym.spaces.Discrete(self.robot.compute_actions_dimension())
        obs_len = 2 + 2 + 2 * self.robot.sonar.num_beams
        self.observation_space = gym.spaces.Box(low=-np.inf * np.ones(obs_len),
                                                high=np.inf * np.ones(obs_len),
                                                dtype=np.float32)

        self.width, self.height = 50, 50
        # vortex core radius parameter range (R in paper2)
        self.vortex_R_range = [1.0, 3.0]
        # keep legacy radius attribute for compatibility when saving/loading
        self.r = self.vortex_R_range[0]
        # flow strength range, interpreted as edge speed used to compute D
        self.v_range = [0.0, 0.5]
        self.obs_r_range = [1, 3]
        self.v_rel_max = 1.0
        self.p = 0.8
        self.clear_r, self.goal_dis = 10.0, 2.0
        self.timestep_penalty, self.collision_penalty = -1.0, -50.0
        self.goal_reward, self.discount = 100.0, 0.99
        self.num_cores, self.num_obs = 8, 5
        self.min_start_goal_dis = 25.0
        self.cores, self.obstacles = [], []
        self.total_timesteps, self.episode_timesteps = 0, 0
        # time-varying flow: change vortex layout every N environment steps
        self.flow_change_interval_steps = 20
        self.last_flow_change_step = 0
        self.reset_start_and_goal, self.random_reset_state = True, True
        self.set_boundary = False
        self.schedule = schedule

    def seed(self, seed):
        self.sd = seed
        self.rd = np.random.RandomState(seed)
        return [seed]

    def get_state_space_dimension(self):
        # 返回观测向量的长度：2(速度) + 2(目标点) + 2*声呐线数
        return 2 + 2 + 2 * self.robot.sonar.num_beams

    def get_action_space_dimension(self):
        # 返回机器人可执行动作的总数
        return self.robot.compute_actions_dimension()

    def _update_kdtree(self):
        if self.cores:
            self.core_centers = scipy.spatial.KDTree(np.array([[c.x, c.y] for c in self.cores]))
        if self.obstacles:
            self.obs_centers = scipy.spatial.KDTree(np.array([[o.x, o.y] for o in self.obstacles]))

    def reset(self):
        if self.schedule is not None:
            idx = len(
                np.array(self.schedule["timesteps"])[np.array(self.schedule["timesteps"]) <= self.total_timesteps]) - 1
            self.num_cores = self.schedule["num_cores"][idx]
            self.num_obs = self.schedule["num_obstacles"][idx]
            self.min_start_goal_dis = self.schedule["min_start_goal_dis"][idx]

        self.episode_timesteps = 0
        self.cores.clear();
        self.obstacles.clear()

        if self.reset_start_and_goal:
            iteration, max_dist = 500, 0.0
            while iteration > 0:
                start = self.rd.uniform(low=2.0, high=self.width - 2.0, size=2)
                goal = self.rd.uniform(low=2.0, high=self.width - 2.0, size=2)
                if np.linalg.norm(goal - start) > max_dist:
                    max_dist = np.linalg.norm(goal - start);
                    self.start, self.goal = start, goal
                iteration -= 1
                if max_dist > self.min_start_goal_dis: break

        self._resample_cores()

        num_o = self.num_obs
        iteration = 500
        while num_o > 0 and iteration > 0:
            center = self.rd.uniform(5.0, self.width - 5.0, 2)
            obs = Obstacle(center[0], center[1], self.rd.uniform(self.obs_r_range[0], self.obs_r_range[1]))
            if self.check_obstacle(obs):
                self.obstacles.append(obs)
                num_o -= 1
            iteration -= 1
        self._update_kdtree()

        self.reset_robot()
        self.last_flow_change_step = self.total_timesteps
        return self.get_observation()

    def step(self, action):
        self._maybe_refresh_flow_field()
        self.robot.action_history.append(action)
        dis_before = self.dist_to_goal()
        is_collision = False
        step_energy_cost = 0.0

        for _ in range(self.robot.N):
            for core in self.cores:
                core.x += core.vx * self.robot.dt
                core.y += core.vy * self.robot.dt
            self._update_kdtree()

            va = self.robot.speed
            step_energy_cost += 0.5 * (np.abs(va) ** 3) * self.robot.dt

            current_v = self.get_velocity(self.robot.x, self.robot.y)
            self.robot.update_state(action, current_v)
            self.robot.trajectory.append([self.robot.x, self.robot.y])

            if self.check_collision():
                is_collision = True
                break

        reward = -step_energy_cost + (dis_before - self.dist_to_goal())

        if is_collision:
            reward += self.collision_penalty
            done, info = True, {"state": "collision"}
        elif self.check_reach_goal():
            reward += self.goal_reward
            done, info = True, {"state": "reach goal"}
        elif self.set_boundary and self.out_of_boundary():
            done, info = True, {"state": "out of boundary"}
        else:
            done = self.episode_timesteps >= 1000
            info = {"state": "normal"}

        self.episode_timesteps += 1
        self.total_timesteps += 1
        return self.get_observation(), reward, done, info

    def get_velocity(self, x, y):
        """
        Lamb-Oseen vortex model (paper2):
        v = (D / (2*pi*r^2)) * (1 - exp(-r^2 / R^2)) * tangential_direction
        Dynamic flow uses current core positions and strength.
        """
        if not self.cores: return np.zeros(2)
        d, idx = self.core_centers.query(np.array([x, y]), k=len(self.cores))
        if isinstance(idx, (int, np.int64)): idx = [idx]

        v_vel = np.zeros((2, 1))
        for i in list(idx):
            core = self.cores[i]
            dx, dy = x - core.x, y - core.y
            dis = np.sqrt(dx ** 2 + dy ** 2)
            if dis < 1e-6: continue

            v_unit = np.array([[-dy / dis], [dx / dis]])
            if core.clockwise: v_unit *= -1

            coef = core.D / (2 * np.pi * (dis ** 2))
            decay = 1.0 - np.exp(- (dis ** 2) / (core.R ** 2))
            speed = coef * decay
            v_vel += v_unit * speed

        return v_vel.flatten()

    def check_core(self, core_j):
        margin = core_j.R
        if core_j.x - margin < 0 or core_j.x + margin > self.width: return False
        if core_j.y - margin < 0 or core_j.y + margin > self.height: return False
        pos = np.array([core_j.x, core_j.y])
        if np.linalg.norm(pos - self.start) < margin + self.clear_r: return False
        if np.linalg.norm(pos - self.goal) < margin + self.clear_r: return False
        for core_i in self.cores:
            dx, dy = core_i.x - core_j.x, core_i.y - core_j.y
            dis = np.sqrt(dx ** 2 + dy ** 2)
            if dis < (core_i.R + core_j.R): return False
        return True

    def check_obstacle(self, obs):
        if obs.x - obs.r < 0 or obs.x + obs.r > self.width or obs.y - obs.r < 0 or obs.y + obs.r > self.height: return False
        pos = np.array([obs.x, obs.y])
        if np.linalg.norm(pos - self.start) < obs.r + self.clear_r or np.linalg.norm(
            pos - self.goal) < obs.r + self.clear_r: return False
        for c in self.cores:
            if np.linalg.norm(np.array([c.x, c.y]) - pos) <= c.R + obs.r: return False
        for o in self.obstacles:
            if np.linalg.norm(np.array([o.x, o.y]) - pos) <= o.r + obs.r: return False
        return True

    def _generate_core(self):
        center = self.rd.uniform(0, self.width, 2)
        v_edge = self.rd.uniform(self.v_range[0], self.v_range[1])
        core_R = self.rd.uniform(self.vortex_R_range[0], self.vortex_R_range[1])
        D = 2 * np.pi * core_R * v_edge
        core = Core(center[0], center[1], self.rd.binomial(1, 0.5), D, core_R, rd_state=self.rd)
        return core

    def _resample_cores(self):
        self.cores.clear()
        num_c = self.num_cores
        iteration = 500
        while num_c > 0 and iteration > 0:
            core = self._generate_core()
            if self.check_core(core):
                self.cores.append(core)
                num_c -= 1
            iteration -= 1
        self._update_kdtree()

    def _maybe_refresh_flow_field(self):
        if self.flow_change_interval_steps is None or self.flow_change_interval_steps <= 0:
            return
        if (self.total_timesteps - self.last_flow_change_step) >= self.flow_change_interval_steps:
            self._resample_cores()
            self.last_flow_change_step = self.total_timesteps

    def out_of_boundary(self):
        return self.robot.x < 0 or self.robot.x > self.width or self.robot.y < 0 or self.robot.y > self.height

    def reset_with_eval_config(self, config):
        self.episode_timesteps = 0
        self.sd = config["env"]["seed"]
        self.width, self.height = config["env"]["width"], config["env"]["height"]
        self.r = config["env"].get("r", self.vortex_R_range[0])
        self.start, self.goal = np.array(config["env"]["start"]), np.array(config["env"]["goal"])
        self.cores.clear()
        for i in range(len(config["env"]["cores"]["positions"])):
            p = config["env"]["cores"]["positions"][i]
            # backward compatibility: Gamma may be stored instead of D/R
            core_D = config["env"]["cores"].get("D", config["env"]["cores"]["Gamma"])[i]
            core_R_list = config["env"]["cores"].get("R", None)
            core_R = core_R_list[i] if core_R_list is not None else self.vortex_R_range[0]
            self.cores.append(
                Core(p[0], p[1], config["env"]["cores"]["clockwise"][i], core_D, core_R))
        self.obstacles.clear()
        for i in range(len(config["env"]["obstacles"]["positions"])):
            p = config["env"]["obstacles"]["positions"][i]
            self.obstacles.append(Obstacle(p[0], p[1], config["env"]["obstacles"]["r"][i]))
        self._update_kdtree()

        r_conf = config["robot"]
        self.robot.dt, self.robot.N = r_conf["dt"], r_conf["N"]
        self.robot.length, self.robot.width, self.robot.r = r_conf["length"], r_conf["width"], r_conf["r"]
        self.robot.max_speed = r_conf["max_speed"]
        self.robot.a, self.robot.w = np.array(r_conf["a"]), np.array(r_conf["w"])
        self.robot.compute_actions()

        s_conf = r_conf["sonar"]
        self.robot.sonar.range, self.robot.sonar.angle, self.robot.sonar.num_beams = s_conf["range"], s_conf["angle"], \
        s_conf["num_beams"]
        self.robot.sonar.compute_phi();
        self.robot.sonar.compute_beam_angles()

        self.reset_robot()
        self.last_flow_change_step = self.total_timesteps
        return self.get_observation()

    def episode_data(self):
        ep = {"env": {"seed": self.sd, "width": self.width, "height": self.height, "r": self.r,
                      "start": list(self.start.astype(float)), "goal": list(self.goal.astype(float)),
                      "v_rel_max": self.v_rel_max, "p": self.p, "discount": self.discount}}
        ep["env"]["cores"] = {"positions": [[float(c.x), float(c.y)] for c in self.cores],
                              "clockwise": [bool(c.clockwise) for c in self.cores],
                              "Gamma": [float(c.D) for c in self.cores],
                              "D": [float(c.D) for c in self.cores],
                              "R": [float(c.R) for c in self.cores]}
        ep["env"]["obstacles"] = {"positions": [[float(o.x), float(o.y)] for o in self.obstacles],
                                  "r": [float(o.r) for o in self.obstacles]}

        ep["robot"] = {"dt": self.robot.dt, "N": self.robot.N, "length": self.robot.length, "width": self.robot.width,
                       "r": self.robot.r, "max_speed": self.robot.max_speed,
                       "a": list(self.robot.a), "w": list(self.robot.w), "init_theta": float(self.robot.init_theta),
                       "init_speed": float(self.robot.init_speed)}
        ep["robot"]["sonar"] = {"range": self.robot.sonar.range, "angle": self.robot.sonar.angle,
                                "num_beams": self.robot.sonar.num_beams}
        ep["robot"]["action_history"] = copy.deepcopy(self.robot.action_history)
        ep["robot"]["trajectory"] = copy.deepcopy(self.robot.trajectory)
        return ep

    def check_collision(self):
        if not self.obstacles: return False
        d, idx = self.obs_centers.query(np.array([self.robot.x, self.robot.y]))
        return d <= (self.obstacles[idx].r + self.robot.r)

    def save_episode(self, filename):
        with open(filename, "w") as f: json.dump(self.episode_data(), f)

    def dist_to_goal(self):
        return np.linalg.norm(self.goal - np.array([self.robot.x, self.robot.y]))

    def check_reach_goal(self):
        return self.dist_to_goal() <= self.goal_dis

    def reset_robot(self):
        self.robot.init_theta = self.rd.uniform(0.0, 2 * np.pi) if self.random_reset_state else 0.0
        self.robot.init_speed = self.rd.uniform(0.0, self.robot.max_speed) if self.random_reset_state else 0.0
        self.robot.reset_state(self.start[0], self.start[1],
                               current_velocity=self.get_velocity(self.start[0], self.start[1]))

    def get_observation(self, for_visualize=False):
        self.robot.sonar_reflection(self.obstacles)
        R_wr, t_wr = self.robot.get_robot_transform()
        R_rw, t_rw = np.transpose(R_wr), -np.transpose(R_wr) @ t_wr
        abs_v = np.array(R_rw @ np.reshape(self.robot.velocity, (2, 1))).flatten()
        goal_r = np.array(R_rw @ np.reshape(self.goal, (2, 1)) + t_rw).flatten()
        sonar_r = [np.array(R_rw @ np.reshape(p[:2], (2, 1)) + t_rw).flatten() if p[2] != 0 else np.zeros(2) for p in
                   self.robot.sonar.reflections]
        if for_visualize: return abs_v, np.array(sonar_r).T, goal_r
        return np.hstack((abs_v, goal_r, np.concatenate(sonar_r)))
