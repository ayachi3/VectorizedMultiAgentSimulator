import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import argparse
import gc
import sys
import psutil

from scipy.stats import norm
from vmas import make_env
from vmas.simulator.utils import save_video


class Memory:
    def __init__(self, batch_size, state_dim, action_dim):
        self.maxsize = 1_000
        # 暂且只记录 s,a,r,s,done
        self.states = []  # np.zeros((self.maxsize,state_dim))
        self.actions = []  # np.zeros((self.maxsize,action_dim))
        # self.probability = []  # 选取该动作的概率
        # self.vals = []  # critic 产生的状态价值
        self.rewards = []  # np.zeros(self.maxsize)  # 此步骤的奖励
        self.dones = []  # np.zeros(self.maxsize,dtype=np.bool)  # 结束标志
        # self.next_vals = []
        self.next_states = []  # np.zeros((self.maxsize,state_dim))
        self.batch_size = batch_size  # 这个指的是每次取多少条记录，而一条记录有多个环境

        self.size = 0
        self.front = 0  # 逻辑队列头部，当满了之后从头部开始覆盖

    # 返回的是Tensor,顺序是 s,a,r,next_s,done
    def sample(self):
        if len(self.states) < self.batch_size:
            raise RuntimeError("回放池容量不足以进行采样")
        index = np.random.randint(0, self.size, self.batch_size)
        # return (self.states[index], self.actions[index], self.rewards[index], self.dones[index], self.next_states[index])
        return (
            torch.cat([self.states[i] for i in index], dim=0).to('cuda'),
            torch.cat([self.actions[i] for i in index], dim=0).to('cuda'),
            torch.cat([self.rewards[i] for i in index], dim=0).to('cuda'),
            torch.cat([self.next_states[i] for i in index], dim=0).to('cuda'),
            torch.cat([self.dones[i] for i in index], dim=0).to('cuda')
        )

    def push(self, state, action, reward, next_state, done):
        if self.size > self.maxsize:
            self.states[self.front] = state
            self.actions[self.front] = action
            self.rewards[self.front] = reward
            self.dones[self.front] = done
            self.next_states[self.front] = next_state
            # update pointer
            self.front = (self.front + 1) % self.maxsize
        else:
            '''
            self.states[self.size] = state
            self.actions[self.size] = action
            self.rewards[self.size] = reward
            self.dones[self.size] = done
            self.next_states[self.size] = next_state
            '''
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.next_states.append(next_state)
            self.size += 1


# work for agent as policy network
class Actor(nn.Module):
    # 前两个参数是状态空间与行动空间的维度
    # 对于连续动作空间，我们分别计算每个动作的均值和方差
    def __init__(self, state_dim, action_dim, config):
        super(Actor, self).__init__()
        self.action_dim = action_dim

        # 暂且使用两层
        self.actor = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            # input: batchsize * action_dim
            # nn.Softmax(dim=1)
            nn.ReLU()
        ).to("cuda")

        # 分别计算每一个智能体的每一个动作维度的均值与方差
        # batch_size * 8
        # 均值绝对值小于1
        self.mean = nn.Sequential(nn.Linear(config.hidden_dim, action_dim), nn.Tanh()).to("cuda")
        self.variance = nn.Sequential(nn.Linear(config.hidden_dim, action_dim), nn.Softplus()).to("cuda")  # 方差必须为正数

        # 这里传入的是本类的参数，不仅仅有self.actor，还有均值和方差的计算网络
        self.optimizer = optim.Adam(self.parameters(), config.actor_lr)

    # 返回要返回每个行动的概率分布
    # 之前在这里有问题，要返回4个agent的行动，但是只有一个概率
    # 返回概率（均值）与方差，后面用随机值变成实际连续行动
    def forward(self, state):
        # 根据概率返回一个行动的编号
        temp = self.actor(state)  # 中间特征
        variance = self.variance(temp)
        mean = self.mean(temp)
        # chunk的参数是切分之后的块的数目而不是每个块的大小
        '''因为是单独的，所以不进行堆叠'''

        return mean, variance


class Critic(nn.Module):
    def __init__(self, state_dim, config):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        ).to("cuda")
        # 这里暂时不进行归一化处理
        self.optimizer = optim.Adam(self.parameters(), config.critic_lr)

    def forward(self, state):
        # 暂时不进行归一化处理
        return self.critic(state)


class Agent:
    def __init__(self, state_dim, action_dim, config,env):
        # 先从输入中提取数据，将来可以将其放在这里
        self.gamma = config.gamma  # 折扣因子
        self.epoch_num = config.n_epochs  # 每次更新重复次数
        # self.gae_lambda = config.gae_lambda  # GAE参数
        # self.policy_clip = config.policy_clip  # clip参数
        self.device = config.device  # 运行设备
        self.agent_num = config.agent_num
        self.config = config
        # 所处的环境
        self.env = env
        self.env_num = config.env_num
        # 这里是单个智能体的维度，而传入演员与评论家的是拼接4倍的
        self.state_dim = state_dim
        self.action_dim = action_dim
        # MAPPPO 使用分散的演员与集中的评论家
        self.actor = Actor(state_dim, action_dim, config=config)
        self.critic = Critic(state_dim * self.agent_num, config)
        # NOTE: 未来可以添加学习率的动态更新

        self.memory = Memory(config.mini_batch_size, state_dim * self.agent_num, action_dim * self.agent_num)
        # 开始时不容易找到好的结果，就随机探索
        self.epsilon = 1

    # 输入动作，决定采取的动作
    def get_action(self, state):
        distribution = [self.actor(x) for x in state]

        # 输入的action的规模应该是 4 * env_num * action_dim
        action = torch.rand(self.agent_num, self.config.env_num, self.action_dim).to(self.device)
        action = action * 2 - 1

        for i in range(len(distribution)):
            distribution[i] = distribution[i][0].detach(), distribution[i][1].detach()
            action[i] = action[i] * torch.sqrt(distribution[i][1].detach()) + distribution[i][0].detach()

        # 标准化
        action = torch.clamp(action, -1, 1).detach()

        return action

    # 与环境交互
    def act_with_env(self, step_limit: int):
        # state 不是stack,而是cat，因为每一个环境中的四个智能体是等价的，detach是因为本矩阵不求偏导
        # cat的输入是矩阵列表，因此dim指的是成员矩阵的连接维度

        # state = torch.cat(self.env.reset(), dim=1).to(self.device).detach()
        # 由于actor是分散的，这里state不连接。这是一个4个tenser的列表
        state = self.env.reset()

        done = torch.tensor([False] * state[0].shape[0]).to(self.device)
        if self.epsilon > 0.001:
            self.epsilon *= 0.999
            print('epsilon =', self.epsilon)
        while step_limit > 0 and not done.all():
            step_limit -= 1

            # 前期随机探索
            if np.random.rand() > self.epsilon:
                '''由于本来也是按照智能体数目与环境交互，因此这里的行动矩阵的形状不需要修改'''
                action = self.get_action(state)  # 4*env_num*action_dim
            else:
                # 随机探索
                action = torch.rand((self.agent_num, self.config.env_num, self.action_dim)).to(self.device) * 2 - 1

            # 输入的action的规模应该是 4 * env_num * action_dim
            next_state, reward, done, info = self.env.step(action)

            #self.env.render(
            #    mode="rgb_array",
            #    agent_index_focus=None,
            #    visualize_when_rgb=True,
            #)

            # 将 state 转化为张量
            # next_state = torch.cat(next_state, dim=1).to("cuda").detach()
            # 将4个智能体的奖励加起来
            reward = sum(reward)  # list{4} tensor(13,)
            done = done.detach()
            done = ~done
            # 使用原本的API
            # state仍旧将每一个代理的状态连接起来，因为critic需要
            self.memory.push(state=torch.cat(state, dim=1).to(self.device).detach(),
                             action=action,
                             reward=reward,
                             next_state=torch.cat(next_state, dim=1).to(self.device).detach(),
                             done=done)

            state = next_state

    def test(self, step_limit, iter: int):
        state = self.env.reset()
        done = torch.tensor([False] * state[0].shape[0]).to(self.device)
        total_reward = 0
        while step_limit > 0 and not done.all():
            step_limit -= 1
            action = self.get_action(state)  # 4*env_num*action_dim

            # 输入的action的规模应该是 4 * env_num * action_dim
            next_state, reward, done, info = self.env.step(action)
            '''
            self.env.render(
                mode="rgb_array",
                agent_index_focus=None,
                visualize_when_rgb=True,
            )'''

            # 将4个智能体的奖励加起来
            reward = sum(reward)  # list{4} tensor(13,)
            total_reward += torch.mean(reward) / self.env_num
            state = next_state
        print('test reward', total_reward)
        with open('log.txt', 'a') as file:
            file.write(str(iter) + ' test reward : ' + str(total_reward.item()) + '\n')

    def learn(self):
        print("start training")
        while self.memory.size < self.memory.batch_size:
            # 当数量不足时，增加训练数据量
            self.act_with_env(300)
        for i in range(self.epoch_num):
            self.act_with_env(100)
            s, a, r, next_s, done = self.memory.sample()
            # 将a形状进行转变
            # 目标是利用评论家计算下一时刻的值加上当前回报
            r = r.squeeze()
            value = self.critic(next_s).squeeze()
            td_target = (r + self.gamma * value * done)

            # 差值用评论家计算当前的状态价值
            delta = td_target.squeeze() - self.critic(s).squeeze()
            '''注意此时传递个两个网络的state形状并不一样,'''
            mean, variance = self.actor.forward(torch.stack(torch.chunk(s, chunks=self.agent_num, dim=1)))  # 用均值表示概率

            mean = torch.stack(torch.chunk(mean, dim=1, chunks=self.config.mini_batch_size),
                               dim=0)  # 20*4*13*2  13 is env_num
            variance = torch.stack(torch.chunk(variance, dim=1, chunks=self.config.mini_batch_size), dim=0)
            a = torch.stack(torch.chunk(a, dim=0, chunks=self.config.mini_batch_size), dim=0)

            mean = torch.flatten(torch.transpose(mean, dim0=1, dim1=2), start_dim=0, end_dim=1)
            variance = torch.flatten(torch.transpose(variance, dim0=1, dim1=2), start_dim=0, end_dim=1)
            a = torch.flatten(torch.transpose(a, dim0=1, dim1=2), start_dim=0, end_dim=1)

            # scale 是标准差, 补上一个小数值以免过小
            prob = 1e-5 + torch.exp(-(a - mean) ** 2 / 2 / variance) / torch.sqrt(2 * torch.pi * variance)
            # torch.gather() 是从矩阵中取特定元素的函数，例如本代码取出第一维的a号元素
            # 因为a是一个向量，因此取出的元素的下标并不一样

            loss = -torch.log(prob) * (delta.reshape((delta.shape[0], 1, 1))) + \
                   F.smooth_l1_loss(self.critic(s).squeeze(), td_target.squeeze().detach())

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.mean().backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()

            if i % 10 == 0:
                print(i)
                self.test(200, i)
            '''
            print('state',s[1:10])
            print('reward',r[1:10])
            print('action',a[1:10])
            '''


# 暂且如此设置参数，在未来要进行修改
def get_args():
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='MAPPO', type=str, help="name of algorithm")
    parser.add_argument('--env_num', default=128, type=int, help='number of environments')
    parser.add_argument('--train_eps', default=200, type=int, help="episodes of training")
    parser.add_argument('--agent_num',default=-2,type=int,help='number of agents')
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--mini_batch_size', default=20, type=int, help='mini batch size')
    parser.add_argument('--n_epochs', default=5_000, type=int, help='update number')
    parser.add_argument('--actor_lr', default=5e-4, type=float, help="learning rate of actor net")
    parser.add_argument('--critic_lr', default=5e-4, type=float, help="learning rate of critic net")
    parser.add_argument('--gae_lambda', default=0.9, type=float, help='GAE lambda')
    parser.add_argument('--policy_clip', default=0.2, type=float, help='policy clip')
    parser.add_argument('-batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--hidden_dim', default=512, type=int, help='hidden dim')
    parser.add_argument('--device', default='cuda', type=str, help="cpu or cuda")

    args = parser.parse_args()
    return args


# now, we don't need heuristic any more
def run_MAPPO(
        scenario_name: str = "transport",
        n_steps: int = 200,
        env_kwargs=None,
        render: bool = False,
        save_render: bool = False,
        device: str = "cuda",
):
    if env_kwargs is None:
        env_kwargs = {}
    assert not (save_render and not render), "To save the video you have to render it"
    cfg = get_args()
    """
    make_env() is a interface provided by VMAS, it can create an environment
    scenario name will define which environment will be created
    num envs will define how many paralled environment will be created
    device will define whether the environment data is on CPU or GPU
    continuous actions will define whether we choose continuous action space
    """
    env = make_env(
        scenario_name=scenario_name,
        num_envs=cfg.env_num,
        device=device,
        continuous_actions=True,
        wrapper=None,
        # Environment specific variables
        **env_kwargs,
    )

    n_states = env.observation_space[0].shape[0]  # 11
    n_actions = env.action_space[0].shape[0]  # 2
    # make agent
    cfg.agent_num = len(env.observation_space)
    agent = Agent(n_states, n_actions, cfg, env)
    agent.learn()


# when run the py file, it will start training
if __name__ == "__main__":
    run_MAPPO(
        scenario_name="transport",
        n_steps=200,
        env_kwargs={},
        render=True,
        save_render=False,
        device="cuda"
    )

"""Actor 是单独的，Critic是集中的"""
