import gym
from gym import spaces
import pygame
import numpy as np
import math

# Constants for environment and robot dimensions
ENV_WIDTH, ENV_HEIGHT = 1000, 800
ROBOT_RADIUS = 30
WHEEL_WIDTH = 10
WHEEL_HEIGHT = 20
WHEEL_BASE = ROBOT_RADIUS * 2 + WHEEL_WIDTH  # Distance between the two wheels
LINK_LENGTH_MIN, LINK_LENGTH_MAX = 50, 120  # Extensible link range



class DifferentialDriveRobot:
    def __init__(self, init_x, init_y, init_theta):
        # 初始化机器人位置和方向
        self.x = init_x
        self.y = init_y
        self.theta = init_theta  # 方向（弧度）

    def update_position(self, v1, v2, dt):
        # 根据两个轮子的速度更新机器人位置和方向
        v = (v1 + v2) / 2  # 线速度
        omega = (v1 - v2) / WHEEL_BASE  # 角速度
        self.theta += omega * dt  # 更新方向
        self.theta %= 2 * math.pi  # 确保方向在0到2π之间
        self.x += v * math.cos(self.theta) * dt  # 更新x坐标
        self.y += v * math.sin(self.theta) * dt  # 更新y坐标

class RobotArm:
    def __init__(self, base_theta, init_theta, init_q):
        # 初始化机械臂的基本角度、初始角度和初始伸展长度
        self.base_theta = base_theta  # 相对于机器人身体的基本方向
        self.theta = init_theta  # 机械臂自身的方向
        self.q = init_q  # 伸展长度
        self.gripper_openness = 0.5  # 夹爪的开合度

    def rotate_base(self, d_theta):
        # 旋转机械臂基础
        self.base_theta = (self.base_theta + d_theta) % (2 * math.pi)

    def rotate_joint(self, alpha):
        # 旋转机械臂关节
        self.theta = (self.theta + alpha) % (2 * math.pi)

    def extend_joint(self, dq):
        # 延伸机械臂关节
        self.q = np.clip(self.q + dq, LINK_LENGTH_MIN, LINK_LENGTH_MAX)

    def adjust_gripper(self, openness):
        # 调整夹爪的开合度
        self.gripper_openness = np.clip(openness, 0, 1)

class EscapeRoomEnv(gym.Env):
    def __init__(self):
        super(EscapeRoomEnv, self).__init__()
        # 初始化环境中的机器人和机械臂
        self.robot = DifferentialDriveRobot(400, 300, math.pi / 2)
        self.robot_arm = RobotArm(0, 0, 50)  # 初始与机器人身体对齐
        pygame.init()
        self.screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))  # 设置显示窗口
        self.clock = pygame.time.Clock()
        # 动作空间：轮子速度、臂基础旋转、关节旋转、伸展、夹爪开合度
        self.action_space = spaces.Box(
            low=np.array([-10, -10, -math.pi, -math.pi, -10, 0], dtype=np.float32),
            high=np.array([10, 10, math.pi, math.pi, 10, 1], dtype=np.float32),
        )

    def step(self, action):
        # 执行动作并更新环境状态
        v1, v2, d_base_theta, alpha, dq, gripper_openness = action
        self.robot.update_position(v1, v2, 1)  # 更新机器人位置
        self.robot_arm.rotate_base(d_base_theta)  # 更新机械臂基础
        self.robot_arm.rotate_joint(alpha)  # 更新关节
        self.robot_arm.extend_joint(dq)  # 更新伸展
        self.robot_arm.adjust_gripper(gripper_openness)  # 更新夹爪
        return np.array([self.robot.x, self.robot.y, self.robot.theta]), 0, False, {}

    def reset(self):
        # 重置环境状态
        self.robot = DifferentialDriveRobot(400, 300, math.pi / 2)
        self.robot_arm = RobotArm(0, 0, 50)
        return np.array([self.robot.x, self.robot.y, self.robot.theta])

    def render(self, mode="human"):
        # 渲染环境状态
        if mode == "human":
            self.screen.fill((255, 255, 255))  # 清空屏幕
            self.draw_robot()  # 绘制机器人
            pygame.display.flip()  # 更新显示
            self.clock.tick(60)  # 控制帧率

    def close(self):
        # 关闭环境
        pygame.quit()

    def draw_robot(self):
        # 绘制机器人的形状和位置
        robot_center = (int(self.robot.x), int(self.robot.y))  # 机器人的中心
        pygame.draw.circle(self.screen, (128, 128, 128), robot_center, ROBOT_RADIUS)  # 画机器人
        robot_theta_rad = math.radians(self.robot.theta)  # 转换角度为弧度
        arm_theta_rad = (
            robot_theta_rad + self.robot_arm.base_theta + self.robot_arm.theta  # 计算机械臂角度
        )
        # 绘制轮子
        self.draw_wheels(robot_center, robot_theta_rad)
        # 绘制机械臂
        arm_end = (
            robot_center[0] + int(self.robot_arm.q * math.cos(arm_theta_rad)),
            robot_center[1] + int(self.robot_arm.q * math.sin(arm_theta_rad)),
        )
        pygame.draw.line(self.screen, (0, 0, 255), robot_center, arm_end, 5)  # 画机械臂
        gripper_color = (
            255 * self.robot_arm.gripper_openness,
            0,
            255 * (1 - self.robot_arm.gripper_openness),  # 根据开合度调整颜色
        )
        pygame.draw.circle(self.screen, gripper_color, arm_end, 10)  # 画夹爪

    def draw_wheels(self, robot_center, angle_rad):
        # 绘制机器人的轮子
        left_wheel_center = (
            robot_center[0] - (WHEEL_BASE / 2) * math.cos(angle_rad),
            robot_center[1] + (WHEEL_BASE / 2) * math.sin(angle_rad),
        )
        right_wheel_center = (
            robot_center[0] + (WHEEL_BASE / 2) * math.cos(angle_rad),
            robot_center[1] - (WHEEL_BASE / 2) * math.sin(angle_rad),
        )
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            pygame.Rect(
                left_wheel_center[0] - WHEEL_WIDTH / 2,
                left_wheel_center[1] - WHEEL_HEIGHT / 2,
                WHEEL_WIDTH,
                WHEEL_HEIGHT,
            ),
        )  # 画左轮
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            pygame.Rect(
                right_wheel_center[0] - WHEEL_WIDTH / 2,
                right_wheel_center[1] - WHEEL_HEIGHT / 2,
                WHEEL_WIDTH,
                WHEEL_HEIGHT,
            ),
        )  # 画右轮

# 初始化并运行环境
env = EscapeRoomEnv()
env.reset()

try:
    for _ in range(500):
        action = env.action_space.sample()  # 随机采样动作
        env.step(action)  # 执行动作
        env.render()  # 渲染环境
except KeyboardInterrupt:
    print("Simulation stopped manually.")  # 手动停止模拟
finally:
    env.close()  # 关闭环境

