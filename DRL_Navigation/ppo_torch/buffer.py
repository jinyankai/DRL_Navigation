import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, device):
        # 初始化存储列表
        self.states = []
        self.actions = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.is_terminals = []
        self.device = device  # 设备设置（CPU或GPU）

    def store(self, state, action, logprob, state_value, reward, done):
        """存储一个时间步的所有信息"""
        action = np.asarray(action)  # 确保 action 是 numpy 数组
        action_dim = action.shape[-1]  # 获取动作的维度

        # 转换输入为张量并移动到指定设备
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).view(-1, action_dim).to(self.device)  # 确保正确的形状
        logprob_tensor = torch.tensor(logprob, dtype=torch.float32).to(self.device)
        state_value_tensor = torch.tensor(state_value, dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done_tensor = torch.tensor(done, dtype=torch.bool).to(self.device)

        # 存储到各自的列表中
        self.states.append(state_tensor)
        self.actions.append(action_tensor)
        self.logprobs.append(logprob_tensor)
        self.state_values.append(state_value_tensor)
        self.rewards.append(reward_tensor)
        self.is_terminals.append(done_tensor)

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.is_terminals = []

    def to_tensor(self):
        # Only convert lists to tensors if not already done in store method
        self.states = torch.stack(self.states).to(self.device) if self.states else torch.empty((0, self.state_dim),
                                                                                               device=self.device)
        self.actions = torch.stack(self.actions).to(self.device) if self.actions else torch.empty((0, self.action_dim),
                                                                                                  device=self.device)
        self.logprobs = torch.stack(self.logprobs).to(self.device) if self.logprobs else torch.empty((0,),
                                                                                                     device=self.device)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device) if self.rewards else torch.empty(
            (0,), device=self.device)

        # 确保只有有效的状态值进行堆叠
        if self.state_values:
            valid_state_values = [val for val in self.state_values if val.numel() > 0]  # 过滤掉空的张量
            self.state_values = torch.stack(valid_state_values).to(self.device) if valid_state_values else torch.empty(
                (0,), device=self.device)
        else:
            self.state_values = torch.empty((0,), device=self.device)

        self.is_terminals = torch.tensor(self.is_terminals, dtype=torch.bool).to(
            self.device) if self.is_terminals else torch.empty((0,), device=self.device)

    def get_data(self):
        """转换所有存储的列表为张量并返回"""
        self.to_tensor()  # 确保所有列表是张量
        return (
            self.states, self.actions, self.logprobs,
            self.state_values, self.rewards, self.is_terminals
        )
