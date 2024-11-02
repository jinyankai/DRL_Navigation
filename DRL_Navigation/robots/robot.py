import math
import pygame

from constants import (
    AXEL_LENGTH,  # 轮轴长度
    ENV_HEIGHT,  # 环境高度
    ENV_WIDTH,  # 环境宽度
    ROBOT_RADIUS,  # 机器人半径
    WHEEL_WIDTH,  # 轮子宽度
)


class Robot:
    def __init__(self, init_position: tuple, init_angle: float = 0):
        self.x, self.y = init_position  # 机器人初始位置
        self.theta = init_angle  # 机器人的朝向（以弧度表示）
        self.vx, self.vy, self.omega = 0, 0, 0  # 初始速度

    def normalize_angle(self, angle):
        """将角度规范化到[-pi, pi]的范围内。"""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def get_collision_circle(self, robot_pos):
        """返回机器人碰撞圆的中心位置和半径。

        Args:
            robot_pos (tuple): 机器人的当前位置 (x, y)。

        Returns:
            tuple: 包含机器人的圆心坐标和半径的元组 (cx, cy, radius)。
        """
        return (robot_pos[0], robot_pos[1], ROBOT_RADIUS + WHEEL_WIDTH)

    def circle_rect_collision(self, circle, rect: pygame.Rect):
        """检查圆与矩形之间的碰撞。

        Args:
            circle (tuple): 一个元组 (cx, cy, radius) 表示圆。
            rect (pygame.Rect): 矩形对象。

        Returns:
            bool: 如果发生碰撞返回True，否则返回False。
        """
        cx, cy, radius = circle
        # 找到矩形中离圆心最近的点
        closest_x = max(rect.left, min(cx, rect.right))
        closest_y = max(rect.top, min(cy, rect.bottom))

        # 计算圆心与这个最近点之间的距离
        distance_x = cx - closest_x
        distance_y = cy - closest_y

        # 如果距离小于圆的半径，则发生碰撞
        return (distance_x ** 2 + distance_y ** 2) <= (radius ** 2)

    def check_boundary_collision(self, circle):
        """检查机器人是否与环境边界发生碰撞。

        Args:
            circle (tuple): 一个元组 (cx, cy, radius) 表示机器人的碰撞圆。

        Returns:
            bool: 如果发生边界碰撞返回True，否则返回False。
        """
        cx, cy, radius = circle
        if cx - radius < 0 or cx + radius > ENV_WIDTH:
            return True  # 与左或右边界碰撞
        if cy - radius < 0 or cy + radius > ENV_HEIGHT:
            return True  # 与上或下边界碰撞
        return False

    def update_and_check_collisions(self, left_vel, right_vel, walls, dt=1):
        """更新机器人的位置并检查与墙壁的碰撞。

        Args:
            left_vel (float): 左轮的速度。
            right_vel (float): 右轮的速度。
            dt (int): 时间增量。
            walls (list): 表示墙壁的pygame.Rect对象列表。

        Returns:
            tuple: (penalty, collision_flag)，其中penalty是表示碰撞惩罚的数值，
                   collision_flag是一个布尔值，表示是否发生了碰撞。
        """
        # 存储更新前的状态
        old_x, old_y, old_theta = self.x, self.y, self.theta

        # 计算新位置
        v = (left_vel + right_vel) / 2  # 线速度
        omega = (right_vel - left_vel) / AXEL_LENGTH  # 角速度
        new_theta = old_theta + omega * dt  # 新的角度
        new_theta = self.normalize_angle(new_theta)  # 规范化角度

        vx = v * math.cos(new_theta)  # x方向的速度
        vy = v * math.sin(new_theta)  # y方向的速度
        new_x = old_x + vx * dt  # 新的x位置
        new_y = old_y + vy * dt  # 新的y位置

        # 创建一个新的圆用于碰撞检测
        robot_circle = self.get_collision_circle((new_x, new_y))
        penalty = 0
        collision_flag = False

        # 检查与边界的碰撞
        if self.check_boundary_collision(robot_circle):
            collision_flag = True
            penalty = -10  # 边界碰撞的惩罚

        # 检查与墙壁的碰撞
        for wall in walls:
            if self.circle_rect_collision(robot_circle, wall.rect):
                penalty += -5  # 墙壁碰撞的惩罚
                collision_flag = True  # 发生碰撞
                break  # 一旦发生碰撞，退出循环

        if not collision_flag and penalty < 0:
            # 如果没有碰撞且惩罚为负，回退到之前的状态
            self.x, self.y, self.theta = old_x, old_y, old_theta
        else:
            # 否则更新为新的状态
            self.x, self.y, self.theta = new_x, new_y, new_theta

        self.vx = vx  # 更新x方向速度
        self.vy = vy  # 更新y方向速度
        self.omega = omega  # 更新角速度

        return (penalty, collision_flag)  # 返回惩罚和碰撞标志
