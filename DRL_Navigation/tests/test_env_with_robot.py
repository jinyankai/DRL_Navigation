import pygame
import numpy as np
from envs.escape_room_continuous_space_env import EscapeRoomEnv


def run_environment():
    # 初始化pygame
    pygame.init()
    env = EscapeRoomEnv()
    env.reset()

    running = True
    while running:
        # 渲染环境的视觉显示
        env.render()
        pygame.event.pump()  # 更新事件内部缓冲区以检测连续按键

        # 检查连续按键
        keys = pygame.key.get_pressed()

        # 初始化动作为停止（2）
        action = np.zeros(5, dtype=np.float32)  # 创建一个5维的动作数组
        action[2] = 0  # 默认线速度为0（停止）

        if keys[pygame.K_UP]:
            action[0] = 1  # 前进
        elif keys[pygame.K_DOWN]:
            action[1] = -1  # 后退
        elif keys[pygame.K_LEFT]:
            action[3] = -1  # 向左旋转
        elif keys[pygame.K_RIGHT]:
            action[4] = 1  # 向右旋转

        # 将动作传递给环境
        state, reward, terminated, truncated, info = env.step(action)

        # 检查是否终止
        if terminated:
            print(f"Action: {action}, State: {state}, Reward: {reward}, Info: {info}")
            print("Episode terminated")
            env.reset()

        # 检查退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()  # 更新整个显示表面到屏幕
        env.clock.tick(30)  # 限制帧率

    env.close()


if __name__ == "__main__":
    run_environment()
