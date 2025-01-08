import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
warnings.filterwarnings(action="ignore")

from Model import Net
from Memory import ReplayMemory, Transition

env_a_shape = 0
class DQN(object):
    def __init__(self, n_states, n_actions, capacity, batch_size, epsilon_start, epsilon_end, epsilon_decay, gamma, replace_iter):
        self.actions = n_actions
        self.states = n_states
        self.replace_iter = replace_iter
        self.eval_net, self.target_net = Net(n_actions=self.actions, n_states=self.states), Net(n_actions=self.actions, n_states=self.states)                
        self.eval_net.train()                                           
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(
        self.eval_net.parameters(), lr=.001, betas=(0.9, 0.999)
        ) # 优化器仅优化eval_net
        self.loss_func = nn.SmoothL1Loss()
        self.loss = 0
        # memory
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.capacity = capacity
        self.memory_counter = 0 
        self.memory = ReplayMemory(self.capacity)
        # gready algthrim
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.epsilon = 0

    def choose_action(self, x):
        # 获取输入
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 选择 max - value
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.learn_step_counter/self.epsilon_decay)
        if np.random.uniform() < self.epsilon:   # greedy # 随机结果是否大于epsilon（0.9）
            self.eval_net.eval()            # 调为eval模式，仅用于评估单状态
            actions_value = self.eval_net.forward(x) # if 取max方法选择执行动作
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if env_a_shape == 0 else action.reshape(env_a_shape)  # return the argmax index
        else:  # 变异情况
            action = np.random.randint(0, self.actions)
            action = action if env_a_shape == 0 else action.reshape(env_a_shape)
        return action

    def memorize(self, state, aciton, reward, next_state):
        self.memory.push(state, aciton, reward, next_state)
    

    def learn(self):
        # 把Eval_Net调到训练模式
        self.eval_net.train()
        # 经验回放不够一个batch的话则继续添加数据
        if self.memory.__len__() < self.batch_size:
            return
        # 神经网络参数更新
        if self.learn_step_counter % self.replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # Sample一个batch的数据
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        # Data
        state_batch = torch.FloatTensor(batch.state)
        action_batch = torch.LongTensor(batch.action)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(batch.next_state)

        # q_eval的学习过程
        # self.eval_net(b_s).gather(1, b_a)  输入我们包（32条）中的所有状态 并得到（32条）所有状态的所有动作价值， .gather(1,b_a) 只取这32个状态中 的 每一个状态的最大值
        q_eval = self.eval_net(state_batch).gather(1, action_batch.unsqueeze(0))
        # q_eval = self.eval_net(b_s).gather(1, b_a)  # eval_net 根据state预测动作价值, .gather(1, b_a) 选择动作价值最大的动作
        # 输入下一个状态 进入critic 输出下一个动作的价值  
        # detach() 阻止网络反向传递，我们的target需要自己定义该如何更新，它的更新在learn那一步
        q_next = self.target_net(next_state_batch).detach()
        # q_target 实际价值的计算  ==  当前价值 + GAMMA（未来价值递减参数）* 未来的价值
        q_target = reward_batch + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        # TD算法, 引入了真实奖励b_r信息
        # q_eval预测值， q_target真实值
        self.loss = self.loss_func(q_eval, q_target)
        
        self.loss.backward()
        self.optimizer.step() 
        self.optimizer.zero_grad()