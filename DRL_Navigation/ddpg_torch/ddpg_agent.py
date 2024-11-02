import os
import numpy as np
import torch as T
import torch.nn.functional as F
from .network import ActorNetwork, CriticNetwork
from .noise import OUActionNoise
from .replay_buffer import ReplayBuffer

class Agent:
    def __init__(
        self,
        alpha,
        beta,
        input_dims,
        tau,
        n_actions,
        gamma=0.99,
        max_size=1000000,
        fc1_dims=400,
        fc2_dims=300,
        batch_size=64,
    ):
        # 初始化超参数和网络
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 目标网络更新速率
        self.batch_size = batch_size  # 每次学习的样本数量
        self.alpha = alpha  # Actor 网络的学习率
        self.beta = beta  # Critic 网络的学习率

        # 初始化经验回放缓冲区
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        # 添加噪声以促进探索
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # 初始化 Actor 和 Critic 网络
        self.actor = ActorNetwork(
            alpha, input_dims, fc1_dims, fc2_dims, n_actions=n_actions, name="actor"
        )
        self.critic = CriticNetwork(
            beta, input_dims, fc1_dims, fc2_dims, n_actions=n_actions, name="critic"
        )

        # 初始化目标网络
        self.target_actor = ActorNetwork(
            alpha,
            input_dims,
            fc1_dims,
            fc2_dims,
            n_actions=n_actions,
            name="target_actor",
        )

        self.target_critic = CriticNetwork(
            beta,
            input_dims,
            fc1_dims,
            fc2_dims,
            n_actions=n_actions,
            name="target_critic",
        )
        self.actor.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        # 更新目标网络参数
        self.update_network_parameters(tau=0.001)

    def choose_action(self, observation):
        # 根据当前观察选择动作
        observation = np.array(observation)  # 转换为 numpy 数组
        self.actor.eval()  # 设置为评估模式
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)  # 转为张量并移至设备
        mu = self.actor.forward(state).to(self.actor.device)  # 计算动作均值
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)  # 添加噪声
        self.actor.train()  # 设置为训练模式

        return mu_prime.cpu().detach().numpy()[0]  # 返回动作

    def remember(self, state, action, reward, state_, done):
        # 存储经历到回放缓冲区
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        # 保存模型
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        # 加载模型
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        # 从经验回放中学习
        if self.memory.mem_cntr < self.batch_size:
            return  # 如果经验不足，则返回

        # 从回放缓冲区随机采样
        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)
        print(f"Sampled states: {states}, actions: {actions}, rewards: {rewards}, states_: {states_}, done: {done}")

        # 转换为张量
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        # 计算目标值
        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0  # 对于完成的状态，目标值为 0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma * critic_value_  # 计算目标
        target = target.view(self.batch_size, 1)

        # 更新 Critic 网络
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)  # 计算损失
        critic_loss.backward()  # 反向传播
        self.critic.optimizer.step()  # 更新参数

        # 更新 Actor 网络
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))  # 计算损失
        actor_loss = T.mean(actor_loss)  # 平均化损失
        actor_loss.backward()  # 反向传播
        self.actor.optimizer.step()  # 更新参数

        self.update_network_parameters()  # 更新目标网络
        print(f"Critic Loss: {critic_loss.item()}, Actor Loss: {actor_loss.item()}")  # 打印损失以确认
        return critic_loss.item(), actor_loss.item()  # 返回损失值

    def update_network_parameters(self, tau=None):
        # 更新目标网络参数
        if tau is None:
            tau = self.tau  # 默认使用初始化的 tau

        # 获取网络参数
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        # 转换为字典
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        # 更新目标网络参数
        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_state_dict[name].clone()
            )

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_state_dict[name].clone()
            )

        # 加载更新后的参数到目标网络
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)