import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(
            self,
            beta,  # 优化器的学习率
            input_dims,  # 输入状态的维度
            fc1_dims,  # 第一个全连接层的神经元数量
            fc2_dims,  # 第二个全连接层的神经元数量
            n_actions,  # 可能的动作数量
            name,  # 网络的名称（用于保存检查点）
            chkpt_dir="tmp/ddpg",  # 保存检查点的目录
    ):
        super(CriticNetwork, self).__init__()

        # 初始化参数
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, name + "_ddpg" + ".pth"
        )

        # 定义网络架构
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)  # 第一个全连接层
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # 第二个全连接层

        self.bn1 = nn.LayerNorm(self.fc1_dims)  # 第一个层的层归一化
        self.bn2 = nn.LayerNorm(self.fc2_dims)  # 第二个层的层归一化

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)  # 动作值层

        self.q = nn.Linear(self.fc2_dims, 1)  # Q值输出层

        # 权重初始化使用He初始化
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003  # 输出层初始化的小常数
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1.0 / np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        # Adam优化器
        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")  # 如果可用，则使用GPU

        self.to(self.device)  # 将模型移动到指定设备

    def forward(self, state, action):
        # 网络的前向传播
        state_value = self.fc1(state)  # 第一个层
        state_value = self.bn1(state_value)  # 归一化
        state_value = F.relu(state_value)  # 激活函数
        state_value = self.fc2(state_value)  # 第二个层
        state_value = self.bn2(state_value)  # 归一化

        action_value = self.action_value(action)  # 获取动作值
        state_action_value = F.relu(T.add(state_value, action_value))  # 结合状态值和动作值
        state_action_value = self.q(state_action_value)  # 最终的Q值输出

        return state_action_value  # 返回Q值

    def save_checkpoint(self, print_on_console=False):
        # 将模型状态保存到文件
        if print_on_console:
            print("... 保存检查点 ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, print_on_console=False):
        # 从文件加载模型状态
        if print_on_console:
            print("... 加载检查点 ...")
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self, print_on_console=False):
        # 将最佳模型状态保存到单独的文件
        if print_on_console:
            print("... 保存最佳检查点 ...")
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + "_best")
        T.save(self.state_dict(), checkpoint_file)


class ActorNetwork(nn.Module):
    def __init__(
            self,
            alpha,  # 优化器的学习率
            input_dims,  # 输入状态的维度
            fc1_dims,  # 第一个全连接层的神经元数量
            fc2_dims,  # 第二个全连接层的神经元数量
            n_actions,  # 可能的动作数量
            name,  # 网络的名称（用于保存检查点）
            chkpt_dir="tmp/ddpg",  # 保存检查点的目录
    ):
        super(ActorNetwork, self).__init__()

        # 初始化参数
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, name + "_ddpg" + ".pth"
        )

        # 定义网络架构
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)  # 第一个全连接层
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # 第二个全连接层

        self.bn1 = nn.LayerNorm(self.fc1_dims)  # 第一个层的层归一化
        self.bn2 = nn.LayerNorm(self.fc2_dims)  # 第二个层的层归一化

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)  # 动作输出层

        # 权重初始化使用He初始化
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003  # 动作层初始化的小常数
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        # Adam优化器
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")  # 如果可用，则使用GPU

        self.to(self.device)  # 将模型移动到指定设备

    def forward(self, state):
        # 网络的前向传播
        x = self.fc1(state)  # 第一个层
        x = self.bn1(x)  # 归一化
        x = F.relu(x)  # 激活函数
        x = self.fc2(x)  # 第二个层
        x = self.bn2(x)  # 归一化
        x = F.relu(x)  # 激活函数
        x = T.tanh(self.mu(x))  # 将动作输出限制在[-1, 1]范围内

        return x  # 返回动作概率

    def save_checkpoint(self, print_on_console=False):
        # 将模型状态保存到文件
        if print_on_console:
            print("... 保存检查点 ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, print_on_console=False):
        # 从文件加载模型状态
        if print_on_console:
            print("... 加载检查点 ...")
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self, print_on_console=False):
        # 将最佳模型状态保存到单独的文件
        if print_on_console:
            print("... 保存最佳检查点 ...")
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + "_best")
        T.save(self.state_dict(), checkpoint_file)
