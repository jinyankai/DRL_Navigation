import math
import time

import gym
import numpy as np
import pygame
from gym import spaces
from constants import (
    AXEL_LENGTH,  # 轮轴长度
    ENV_HEIGHT,  # 环境高度
    ENV_WIDTH,  # 环境宽度
    ROBOT_RADIUS,  # 机器人半径
    WHEEL_WIDTH,  # 轮子宽度
    WHEEL_HEIGHT,  # 轮子高度
)

# 环境和机器人尺寸的常量
WHEEL_RADIUS = WHEEL_HEIGHT / 2
WHEEL_BASE = WHEEL_WIDTH / 2 + AXEL_LENGTH * 2 + ROBOT_RADIUS
WHEEL_OFFSET = (
    ROBOT_RADIUS + WHEEL_HEIGHT / 2
)  # 从机器人中心到轮心的偏移
LINK_LENGTH_MIN, LINK_LENGTH_MAX = 50, 120  # 链接的最小和最大长度


class DifferentialDriveRobot:
    def __init__(self, init_x, init_y, init_theta):
        self.x = init_x  # 初始x坐标
        self.y = init_y  # 初始y坐标
        self.theta = init_theta  # 方向（以弧度表示）

    def update_position(self, v1, v2, dt=1):
        # 通过轮子速度计算机器人速度和角速度
        # 线速度: V = (VL + VR) / 2
        # 角速度: ω = (VR - VL) / W

        v = (v1 + v2) / 2  # 计算线速度
        omega = WHEEL_RADIUS * (v1 - v2) / WHEEL_BASE  # 计算角速度

        # 更新机器人的方向
        self.theta += omega * dt
        self.theta %= 2 * math.pi  # 规范化方向

        # 更新机器人的位置
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt


class RobotArm:
    def __init__(self, init_theta, init_q):
        self.theta = init_theta  # 初始关节角度
        self.q = init_q  # 链接长度
        self.gripper_open = True  # 手爪状态，初始为打开

    def rotate_joint(self, d_theta):
        """旋转关节，改变关节角度。"""
        self.theta += d_theta
        self.theta %= 2 * math.pi  # 规范化角度

    def extend_joint(self, dq):
        """延伸关节，改变链接长度。"""
        self.q += dq
        self.q = max(LINK_LENGTH_MIN, min(self.q, LINK_LENGTH_MAX))  # 限制链接长度

    def toggle_gripper(self):
        """切换手爪的开合状态。"""
        self.gripper_open = not self.gripper_open


class EscapeRoomEnv(gym.Env):
    def __init__(self):
        super(EscapeRoomEnv, self).__init__()
        self.robot = DifferentialDriveRobot(400, 300, 0)  # 初始化机器人
        self.robot_arm = RobotArm(0, 50)  # 初始化机器人手臂
        pygame.init()  # 初始化pygame
        self.screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))  # 设置屏幕
        self.clock = pygame.time.Clock()  # 创建时钟
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -0.1, -10, 0]),  # 动作空间的下界
            high=np.array([1, 1, 0.1, 10, 1]),  # 动作空间的上界
            dtype=np.float32,
        )

    def step(self, action):
        """执行一步操作，并更新环境状态。

        Args:
            action: 动作数组，包括左轮速度、右轮速度、关节旋转角度、链接延伸和手爪动作。

        Returns:
            tuple: 新状态、奖励、完成标志和附加信息。
        """
        v1, v2, d_theta, dq, gripper_action = action  # 解包动作
        self.robot.update_position(v1, v2, 0.1)  # 更新机器人位置，dt = 0.1秒
        self.robot_arm.rotate_joint(d_theta)  # 旋转手臂关节
        self.robot_arm.extend_joint(dq)  # 延伸手臂
        if gripper_action > 0.5:  # 如果手爪动作大于0.5，则切换手爪状态
            self.robot_arm.toggle_gripper()
        return (
            np.array(
                [
                    self.robot.x,  # 机器人的x坐标
                    self.robot.y,  # 机器人的y坐标
                    self.robot.theta,  # 机器人的方向
                    self.robot_arm.q,  # 手臂链接长度
                    self.robot_arm.gripper_open,  # 手爪状态
                ]
            ),
            0,  # 奖励
            False,  # 是否完成
            {},
        )

    def reset(self):
        """重置环境到初始状态。"""
        self.robot = DifferentialDriveRobot(400, 300, 0)  # 重新初始化机器人
        self.robot_arm = RobotArm(0, 50)  # 重新初始化手臂
        return np.array(
            [
                self.robot.x,
                self.robot.y,
                self.robot.theta,
                self.robot_arm.q,
                self.robot_arm.gripper_open,
            ]
        )

    def render(self, mode="human"):
        """渲染环境。

        Args:
            mode: 渲染模式，"human"表示人类可视化。
        """
        if mode == "human":
            self.screen.fill((255, 255, 255))  # 清屏，填充白色
            self.draw_robot(self.screen)  # 绘制机器人
            pygame.display.flip()  # 刷新屏幕
            self.clock.tick(60)  # 控制帧率

    def close(self):
        """关闭环境。"""
        pygame.quit()  # 退出pygame

    def draw_robot(self, screen):
        """在屏幕上绘制机器人及其手臂。

        Args:
            screen: 渲染的屏幕。
        """
        # 机器人主体中心
        robot_center = (int(self.robot.x), int(self.robot.y))

        # 绘制机器人主体
        pygame.draw.circle(screen, (128, 128, 128), robot_center, ROBOT_RADIUS)

        # 绘制轮子
        wheel_angle = math.radians(self.robot.theta)  # 将角度转为弧度
        left_wheel_center = (
            robot_center[0] - WHEEL_OFFSET * math.sin(wheel_angle),
            robot_center[1] + WHEEL_OFFSET * math.cos(wheel_angle),
        )
        right_wheel_center = (
            robot_center[0] + WHEEL_OFFSET * math.sin(wheel_angle),
            robot_center[1] - WHEEL_OFFSET * math.cos(wheel_angle),
        )

        # 左轮
        pygame.draw.rect(
            screen,
            (0, 0, 0),
            (
                left_wheel_center[0] - WHEEL_WIDTH // 2,
                left_wheel_center[1] - WHEEL_HEIGHT // 2,
                WHEEL_WIDTH,
                WHEEL_HEIGHT,
            ),
        )
        # 右轮
        pygame.draw.rect(
            screen,
            (0, 0, 0),
            (
                right_wheel_center[0] - WHEEL_WIDTH // 2,
                right_wheel_center[1] - WHEEL_HEIGHT // 2,
                WHEEL_WIDTH,
                WHEEL_HEIGHT,
            ),
        )

        # 绘制手臂和手爪
        arm_end_x = robot_center[0] + self.robot_arm.q * math.cos(
            math.radians(self.robot.theta + self.robot_arm.theta)
        )
        arm_end_y = robot_center[1] + self.robot_arm.q * math.sin(
            math.radians(self.robot.theta + self.robot_arm.theta)
        )
        arm_end = (int(arm_end_x), int(arm_end_y))
        pygame.draw.line(screen, (0, 0, 255), robot_center, arm_end, 5)  # 绘制手臂

        # 手爪颜色
        gripper_color = (255, 0, 0) if self.robot_arm.gripper_open else (0, 255, 0)
        pygame.draw.circle(screen, gripper_color, arm_end, 10)  # 绘制手爪


# 初始化并运行环境
env = EscapeRoomEnv()  # 创建环境实例
env.reset()  # 重置环境

try:
    for _ in range(500):  # 执行500次操作
        action = env.action_space.sample()  # 随机采样动作
        env.step(action)  # 执行动作
        env.render()  # 渲染环境
        time.sleep(0.05)  # 暂停0.05秒
except KeyboardInterrupt:
    print("Simulation stopped manually.")  # 手动停止模拟
finally:
    env.close()  # 关闭环境
