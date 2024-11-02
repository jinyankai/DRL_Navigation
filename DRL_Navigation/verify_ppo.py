import gym
import numpy as np

from envs.for_ppo import EscapeRoomEnv
from ppo_torch.ppo_agent import PPOAgent


def load_and_simulate(env, agent, n_episodes=5, max_steps=500):
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            env.render()  # Render the environment to visualize the agent's behavior
            action = agent.select_action(
                state
            )  # Agent selects an action based on the current state
            state, reward, terminated, truncated, info = env.step(
                action
            )  # Execute the action in the environment
            done = terminated or truncated
            total_reward += reward
            steps += 1

        print(f"Episode {episode + 1}: Total reward = {total_reward}, Steps = {steps}")
        if "next_goal" in info:
            print(f"Moving to Next Goal: {info['next_goal']}")

        if steps >= max_steps:
            print(f"Episode {episode + 1} reached the maximum of {max_steps} steps.")

    env.close()  # Close the environment when done


def main():
    env = EscapeRoomEnv(max_steps_per_episode=500, goal=(550, 350))
    n_actions = env.action_space.shape[0]
    input_dims = env.observation_space.shape

    # Parameters used during training, for consistency
    alpha = 0.0001
    beta = 0.001
    tau = 0.001
    fc1_dims = 400
    fc2_dims = 300
    n_episodes = 100
    update_interval = 50

    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=40,
        eps_clip=0.2,
        total_updates=n_episodes // update_interval,
        action_std_init=0.6

    )

        # Load models from the appropriate file paths
    agent.load("E:/Enhancing_Autonomous_Robot_Navigation_with_DRL-main/tmp/ppo/checkpoint_100.pth")

    load_and_simulate(env, agent, n_episodes=n_episodes, max_steps=500)
if __name__ == "__main__":
    main()

