import gym
import numpy as np
import pygame
import random
from gym import spaces
from sympy.printing.pretty.pretty_symbology import center

from constants import (
    CHECKPOINT_RADIUS,
    ENV_HEIGHT,
    ENV_WIDTH,
    MAX_WHEEL_VELOCITY,
    SCALE_FACTOR,
)
from robots.checkpoint import Checkpoint
from robots.robot import Robot
from robots.walls import Wall, walls_mapping
from utils.drawing_utils import draw_robot


class EscapeRoomEnv(gym.Env):
    def __init__(self, max_steps_per_episode=2000, delta=15):
        super().__init__()
        self.obs_dis = [0.0 , 0.0 , 0.0 , 0.0]
        self.reward = 0
        # 设置机器人的初始生成位置
        self.spawn_x = int(70 * SCALE_FACTOR)
        self.spawn_y = int(70 * SCALE_FACTOR)
        self.goal = [0,0]
        self.goal[0] = random.randint(100, 530)
        self.goal[1] = random.randint(100, 430)
        # 设置目标位置
        self.goal_position = np.array(
            [int(self.goal[0] * SCALE_FACTOR), int(self.goal[1] * SCALE_FACTOR)]
        )

        # 初始化墙壁
        self.walls = [Wall(**wall_data) for wall_data in walls_mapping]
        self.walls_realm = []

        # 设置允许的偏差
        self.delta = delta

        # 创建目标检查点
        self.goal = Checkpoint(self.goal_position, CHECKPOINT_RADIUS, (0, 128, 0), "G")

        # 设置观察空间（状态空间）
        # why -1.5 * ENV_WIDTH and -1.5 * ENV_HEIGHT
        # why 1.5 * ENV_WIDTH and 1.5 * ENV_HEIGHT
        low = np.array([-1.5 * ENV_WIDTH, -1.5 * ENV_HEIGHT, -np.pi, -5.0, -5.0, -5.0, 0.0 ,0.0 ,0.0 , 0.0 , -10000])
        high = np.array([1.5 * ENV_WIDTH, 1.5 * ENV_HEIGHT, np.pi, 5.0, 5.0, 5.0, 600.0, 600.0 , 600.0 , 600.0 , 100000])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 设置动作空间（控制机器人的行为）
        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]),
                                       dtype=np.float32)

        # 初始化机器人
        self.robot = Robot((self.spawn_x, self.spawn_y))
        self.max_steps_per_episode = max_steps_per_episode
        self.t = 0  # 计时器

        # 初始化与目标的距离
        self.old_distance = np.linalg.norm(
            np.array([self.robot.x, self.robot.y]) - np.array(self.goal.center_pos)
        )
        self.screen = None
        self.clock = None

    def not_in_obstacle(self):
        x,y = self.goal_position[0],self.goal_position[1]
        for wall in self.walls:
            start_x = wall.start_pos[0]
            start_y = wall.start_pos[1]
            end_x = start_x + wall.width
            end_y = start_y + wall.height
            if (x > start_x and x < end_x) and (y > start_y and y < end_y):
                return False
        return True


    def step(self, action):
        # 确保动作在预定范围内
        action = np.clip(action, -1, +1).astype(np.float32)

        # 调试信息：打印动作形状和类型
        # print("action shape:", action.shape)
        # print("action type:", type(action))

        # 确保 action 是一个一维数组，且至少包含两个元素
        if action.ndim != 1 or action.shape[0] < 2:
            raise ValueError("Action must be a 1D array with at least two elements.")

        # 从动作中获取左轮和右轮速度
        left_vel = action[0] * MAX_WHEEL_VELOCITY
        right_vel = action[1] * MAX_WHEEL_VELOCITY

        # 更新机器人位置并检查碰撞
        penalty, out_of_bounds = self.robot.update_and_check_collisions(
            left_vel, right_vel, self.walls, dt=1
        )
        reward = 0

        # 计算新的位置并更新距离
        new_pos = np.array([self.robot.x, self.robot.y])
        new_distance = np.linalg.norm(new_pos - np.array(self.goal.center_pos))

        # 更新距离差
        alpha = 0.1
        distance_improvement = float(self.old_distance - new_distance)
        self.old_distance = new_distance

        # 计算机器人的朝向与目标方向的差异
        goal_direction = np.array(self.goal.center_pos) - new_pos
        goal_angle = np.arctan2(goal_direction[1], goal_direction[0])
        heading_difference = self.robot.theta - goal_angle

        # 规范化朝向差异
        heading_difference = (heading_difference + np.pi) % (2 * np.pi) - np.pi

        # 根据朝向差异应用奖励
        # if heading_difference > np.pi / 6:  # 如果偏离超过30度
        #    reward += -np.log1p(heading_difference)
        #   if self.robot.omega > np.pi / 6:
        #       reward += -alpha
        #计算距离障碍物的奖励
        i = 0
        for wall in self.walls:
            center_pos = [0,0]
            center_pos[0] = wall.start_pos[0] + wall.width / 2
            center_pos[1] = wall.start_pos[1] + wall.height / 2
            radius =  np.linalg.norm(np.array(wall.start_pos)-np.array(center_pos))
            new_dis = np.linalg.norm(np.array(center_pos) - new_pos)
            if(new_dis < radius + 10):
                reward -= 5 * (abs(new_dis - radius))
            else:
                reward += 0.05 * (abs(new_dis - radius))
            self.obs_dis[i] = new_dis
            i = i + 1
        # reward += -np.log1p(np.min(obs_dis))

        # 计算距离改善的奖励
        if distance_improvement > 0:
            reward += +np.log1p(distance_improvement)
            if np.abs(left_vel) + np.abs(right_vel) < MAX_WHEEL_VELOCITY:
                reward += +alpha  # 促进高效运动
        else:
            reward += -np.log1p(-distance_improvement)  # 奖励减少

        reward += penalty
        reward += -alpha  # 步数惩罚

        self.reward = reward

        # 返回状态
        state = np.array(
            [
                self.robot.x,
                self.robot.y,
                self.robot.theta,
                self.robot.vx,
                self.robot.vy,
                self.robot.omega,
                self.obs_dis[0],
                self.obs_dis[1],
                self.obs_dis[2],
                self.obs_dis[3],
                self.reward

            ]
        )

        self.t += 1
        terminated = False
        truncated = False
        info = {}

        # 检查是否达到目标
        if self.goal.check_goal_reached((self.robot.x, self.robot.y), delta=self.delta):
            base_reward = +10_000
            efficiency_bonus = (np.log1p(self.max_steps_per_episode / self.t)) * base_reward * alpha  # 奖励根据步数
            reward += base_reward + efficiency_bonus
            print(
                f"Goal '{self.goal.label}' reached in {self.t} steps with cumulative reward {reward} for this episode."
            )
            self.old_distance = np.linalg.norm(np.array([self.robot.x, self.robot.y]) - np.array(self.goal.center_pos))
            terminated = True
            info["reason"] = "Goal_reached"
        elif out_of_bounds:
            terminated = True
            reward += -50
            info["reason"] = "out_of_bounds"
        elif self.t >= self.max_steps_per_episode:
            truncated = True
            reward += -5
            info["reason"] = "max_steps_reached"

        return state, reward, terminated, truncated, info

    def reset(self):
        # 重置机器人位置和状态
        self.robot = Robot([self.spawn_x, self.spawn_y], init_angle=0)
        self.t = 0
        self.old_distance = np.linalg.norm(
            np.array([self.robot.x, self.robot.y]) - np.array(self.goal.center_pos)
        )
        self.screen = None
        self.clock = None
        info = {"message": "Environment reset."}
        self.goal.reset()
        return (
            np.array(
                [
                    self.robot.x,
                    self.robot.y,
                    self.robot.theta,
                    self.robot.vx,
                    self.robot.vy,
                    self.robot.omega,
                    self.obs_dis[0],
                    self.obs_dis[1],
                    self.obs_dis[2],
                    self.obs_dis[3],
                    self.reward
                ]
            ),
            info,
        )

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # 清空屏幕，设置白色背景
        self.screen.fill((255, 255, 255))

        # 绘制所有墙壁
        for wall in self.walls:
            wall.draw(self.screen)

        # 绘制目标
        self.goal.draw(self.screen)

        # 绘制机器人
        draw_robot(self.screen, self.robot)

        if mode == "human":
            # 更新显示
            pygame.display.flip()
            self.clock.tick(30)
        elif mode == "rgb_array":
            # 获取当前渲染帧作为RGB数组
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))  # 转换形状
            return frame

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = EscapeRoomEnv()
    assert (
            isinstance(env.observation_space, gym.spaces.Box)
            and len(env.observation_space.shape) == 1
    )
    try:
        for _ in range(5):
            action = env.action_space.sample()
            # print("action1", action)
            env.step(action)
            env.render()
    except KeyboardInterrupt:
        print("Simulation stopped manually.")
    finally:
        env.close()
