"""
无人机模拟
"""
import math
import numpy as np
import matplotlib.pyplot as plt   
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist, cdist  #计算点之间的距离
from numpy.linalg import norm
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import cv2 
import random
import os
import numpy as np
from collections      import deque

width, height = 300, 300  #设置屏幕上模拟窗口的宽度和高度
 
class Environment():
    def __init__(self, N):
        """ 
        初始化模拟
        """
        #初始化位置和速度
        self.pos = [width/2.0, height/2.0] + np.random.random([N, 2]) * 50 - 25
        angles = 2*np.pi*np.random.random(N)
        self.vel = np.array(list(zip(np.sin(angles), np.cos(angles)))) * 2
        self.N = N      
        self.minDist = 400
        self.maxRuleVel = 0.03
        # max maginitude of final velocity
        self.maxVel = 2.0
        self.beta = 0.2
        self.count = 0
    def reset(self):
        """
        重置环境
        """
        self.pos = [width/2.0, height/2.0] + np.random.random([self.N, 2]) * 50 - 25
        angles = 2*math.pi*np.random.random(self.N)
        self.vel = np.array(list(zip(np.sin(angles), np.cos(angles))))
        states = []
        for itr in range(self.N):
            velo = np.delete(self.vel, itr, 0).reshape([-1])
            posi = np.delete(self.pos, itr, 0).reshape([-1])
            state = np.concatenate([self.vel[itr], self.pos[itr]]+[self.vel[a] for a in range(self.N) if a!=itr]+[self.pos[itr]-self.pos[a] for a in range(self.N) if a!=itr])
            bound = np.array([np.min([height-self.pos[itr, 0], 50]), 
                np.min([self.pos[itr, 0], 50]), 
                np.min([width-self.pos[itr, 1], 50]), 
                np.min([self.pos[itr, 1], 50])
                ])
            state = np.concatenate([state, bound
            ])
            state = state.reshape([1, -1])
            states.append(state)
        states = np.concatenate(states, axis=0)
        self.count = 0
        self.cpos = []
        return states
    def render(self):
        """
        用于渲染图形
        """
        imgs = np.ones((height,width,3),np.uint8) * 0
        imgs[:, :, 0] = 255
        self.distMatrix = squareform(pdist(self.pos))
        D = (self.distMatrix < self.minDist)
        self.Pairs = D        
        cv2.rectangle(imgs, (50, 50), (height-50, width-50), (255, 255, 255), -1)
        for itr in self.pos:
            x, y = int(itr[0]), int(itr[1])
            cv2.circle(imgs, (x, y), 6, (0, 100, 155), -1)
        #self.cpos.append([x, y])
        #if len(self.cpos) > 3:
        #    for i in range(len(self.pos)-1):
        #        ax, ay = int(self.pos[i][0]), int(self.pos[i][1]) 
        #        bx, by = int(self.pos[i+1][0]), int(self.pos[i+1][1]) 
        #        cv2.line(imgs, (ax, ay), (bx, by), (0, 0, 0), 1)
        #print(len(self.cpos))
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(imgs, 'F%d'%self.count, (10, 10), font, 0.5, (0, 0, 255), 2)
        cv2.imshow("uavs", imgs)
    def tick(self, frameNum):
        """
        进行一次更新，之前测试用到
        """
        self.distMatrix = squareform(pdist(self.pos))     #用 squareform()和 pdist()方法来计算一组点之间两两的距离
        # apply rules:
        self.vel =  (1-self.beta)*self.vel + self.beta*self.applyRules()
        self.pos += self.vel
        self.applyBC()
        self.render()
        """
        # update data
        pts.set_data(self.pos.reshape(2*self.N)[::2], 
                     self.pos.reshape(2*self.N)[1::2])
        vec = self.pos + 10*self.vel/self.maxVel
        beak.set_data(vec.reshape(2*self.N)[::2], 
                      vec.reshape(2*self.N)[1::2])
        """
    def normVelo(self, v):
        """
        由于无加速指令，因此速度为常数
        """
        norm = np.sqrt(np.sum(v**2, 1, keepdims=True))+1e-6
        v = v*self.maxVel/norm
        return v 
    def action(self, a):
        """
        action包含简单的8类，代表在二维空间中的八个方向。
        """
        v = np.zeros_like(self.vel)
        for idx, itr in enumerate(a):
            if itr==0:
                v[idx] = np.array([0, self.maxVel])
            elif itr==1:
                v[idx] = np.array([0, -self.maxVel])
            elif itr==2:
                v[idx] = np.array([self.maxVel, 0])
            elif itr==3:
                v[idx] = np.array([-self.maxVel, 0])
            elif itr==4:
                v[idx] = np.array([1/np.sqrt(2), -1/np.sqrt(2)])*self.maxVel
            elif itr==5:
                v[idx] = np.array([-1/np.sqrt(2), -1/np.sqrt(2)])*self.maxVel
            elif itr==6:
                v[idx] = np.array([-1/np.sqrt(2), 1/np.sqrt(2)])*self.maxVel
            elif itr==7:
                v[idx] = np.array([1/np.sqrt(2), 1/np.sqrt(2)])*self.maxVel
            else:
                print("None")
        return v
    def step(self, actions):
        """
        执行动作action后产生新的状态
        """
        self.vel = (1-self.beta)*self.vel + self.beta*self.action(actions)
        #print(actions)
        self.vel = self.normVelo(self.vel)
        self.pos += self.vel 
        self.count += 1
        states = []
        for itr in range(self.N):
            velo = np.delete(self.vel, itr, 0).reshape([-1])
            posi = np.delete(self.pos, itr, 0).reshape([-1])
            state = np.concatenate([self.vel[itr], self.pos[itr]]+[self.vel[a] for a in range(self.N) if a!=itr]+[self.pos[itr]-self.pos[a] for a in range(self.N) if a!=itr])
            bound = np.array([np.min([height-self.pos[itr, 0], 50]), 
                np.min([self.pos[itr, 0], 50]), 
                np.min([width-self.pos[itr, 1], 50]), 
                np.min([self.pos[itr, 1], 50])
                ])
            state = np.concatenate([state, bound
            ])
            state = state.reshape([1, -1])
            states.append(state)
        states = np.concatenate(states, axis=0)
        reward = 0
        self.applyBC() 
        done = False
        if self.N == len(self.vel):
            reward = 1
        else:
            done = True
        deltaR = 50
        for index,coord in enumerate(self.pos):
            if coord[0] > width - deltaR or coord[0] < deltaR or coord[1] > height - deltaR or coord[1] < deltaR:
                reward -= 0.1
        return states, reward, done, self.vel
    def limitVec(self, vec, maxVal):
        """
        限制二维空间的速度
        """
        mag = norm(vec)
        if mag > maxVal or mag < maxVal * 0.5:
            vec[0], vec[1] = vec[0]*maxVal/mag, vec[1]*maxVal/mag
    def limit(self, X, maxVal):
        """
        限制最大速度
        """
        for vec in X:
            self.limitVec(vec, maxVal)
    def applyBC(self):
        """
        边界条件，不添加其他约束
        在无人机接近边界、距离过近时坠毁，此时程序终止
        在此不设定其他规则，仅形成淘汰过程。
        """
        deltaR = 2.0    #该行中的deltaR提供了一个微小的缓冲
        self.pos_new = []
        self.vel_new = []
        sel_idx = []
        for index,coord in enumerate(self.pos):
            if coord[0] > width + deltaR or coord[0] < -deltaR or coord[1] > height + deltaR or coord[1] < - deltaR:
                pass
            else:
                sel_idx.append(index)
        self.distMatrix = squareform(pdist(self.pos))
        dist = np.zeros(self.N)
        for idx, itr in enumerate(self.distMatrix):
            step = (itr<3).astype(np.int32)
            step[idx] = 0
            dist+=step
        sel_idx2 = []
        for idx, itr in enumerate(dist):
            if itr==0 and (idx in sel_idx):
                sel_idx2.append(idx)
                
        self.pos = self.pos[sel_idx2]
        self.vel = self.vel[sel_idx2]
        self.distMatrix = squareform(pdist(self.pos))
        D = (self.distMatrix < self.minDist)
        self.Pairs = D
    
    def applyRules(self):
        """
        三种既定规则，之前测试时使用到
        """
        D = (self.distMatrix < self.minDist)
        self.Pairs = D
        r_vel = np.zeros_like(self.vel)
        for idx, w in enumerate(D):
            #避免碰撞
            pos = self.pos[w]
            vel = self.vel[w]
            dist = self.pos[idx] - pos
            dist = dist/np.max(np.abs(dist)+1e-6)
            vel1 = -np.sum(1-dist, axis=0)
            #vel1 = 1/(vel1)
            vel1 = vel1/(norm(vel1)+1e-6)
            #趋同
            vel2 = np.mean(vel, axis=0)
            vel2 = vel2/(norm(vel2)+1e-6)
            #聚集
            vel3 = np.mean(self.pos, axis=0) - self.pos[idx]
            #vel3 = np.mean(self.pos[itr], axis=0)-self.pos[idx]
            vel3 = vel3/(norm(vel3)+1e-6)
            r_vel[idx] = 1*vel1 + vel2 + vel3 
        self.limitVec(r_vel, self.maxVel)
        return r_vel 


import torch 
import torch.nn as nn
import torch.nn.functional as F 
from collections import deque
import random 
class Agent():
    def __init__(self, state_size, action_size):
        """
        智能体初始化函数
        """
        # 状态个数
        self.state_size         = state_size
        # 动作个数
        self.action_size        = action_size
        # 设置记忆
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 1e-3
        self.gamma              = 0.95 # DQN中参数
        self.exploration_rate   = 1.0  # 开始时需要随机探索
        self.exploration_min    = 0.05 # 最小探索率
        self.exploration_decay  = 0.995# 设置探索值衰减
        self.Q              = self._build_model()
        self.optim = torch.optim.Adam(self.Q.parameters(), 0.0001)

    def _build_model(self):
        """
        建立多层神经网络作为Q函数
        """
        model = nn.Sequential(
            nn.Linear(self.state_size, 128), 
            nn.ReLU(), 
            nn.Linear(128, 128), 
            nn.ReLU(),    
            nn.Linear(128, self.action_size)         
        )
        return model
    def save_model(self):
        """保存模型"""
        torch.save(self.Q.state_dict(), "ckpt/brain.pkl")
    def act(self, state):
        """
        执行动作，state 当前状态
        """
        if np.random.rand() <= self.exploration_rate:
            # 如果小于某一值，则随机探索
            return np.random.randint(0, self.action_size, state.shape[0])
        else:
            # 否则计算 Q 值最大的动作
            state = torch.from_numpy(state).float()
            act_values = self.Q(state)
            act = act_values.detach().numpy()
            return np.argmax(act, axis=1)
    def remember(self, state, action, reward, next_state, done):
        """将四要素存储在记忆中"""
        self.memory.append((state, action, reward, next_state, done))
    def replay(self, sample_batch_size):
        """从记忆中选择样本进行训练"""
        if len(self.memory) < sample_batch_size:
            # 如果不足长度则退出
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            # 转换为 PyTorch 类型
            state, next_state = torch.from_numpy(state.astype(np.float32)), \
                                torch.from_numpy(next_state.astype(np.float32))
            if not done:
                val = self.Q(next_state) 
                val = val.detach().numpy()
                target = reward + self.gamma * np.amax(val, axis=1)
            # 目标值函数
            targetQ = self.Q(state)
            targetQ = targetQ.detach().numpy()
            targetQ[0, action] = target 
            targetQ = torch.from_numpy(targetQ).float()
            predQ = self.Q(state) # 预测值函数
            loss = F.mse_loss(predQ, targetQ)
            # 反向传播过程
            loss.backward() 
            self.optim.step() 
            self.optim.zero_grad()
            if self.exploration_rate > self.exploration_min:
                self.exploration_rate *= self.exploration_decay
    

class UAVs:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 10000
        self.env               = Environment(1)
        self.state_size        = 8 #包括邻近无人机速度、相对位置，本个无人机速度、边界位置
        self.action_size       = 8 #八个速度方向
        # 定义智能体
        self.agent             = Agent(self.state_size, self.action_size)
    def run(self):
        try:
            # 循环迭代
            for index_episode in range(self.episodes):
                state = self.env.reset()
                # 状态就是八个属性
                state = np.reshape(state, [1, self.state_size])
                done = False
                index = 0
                while not done:
                    # 判断是否结束
                    self.env.render()# 展示飞行过程
                    action = self.agent.act(state) # 预测下一步动作
                    # 计算四要素
                    next_state, reward, done, _ = self.env.step(action)
                    # 下一个时刻状态
                    next_state = np.reshape(next_state, [1, self.state_size])
                    # 存储在记忆中
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                    cv2.waitKey(10)
                # Replay就是从记忆中选择训练的过程
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()
        
def tick(frameNum, env):
    """
    更新函数
    """
    env.tick(frameNum)
def main():
    """
    print('Starting environment...')
    N = 5 
    env = Environment(N)
    count = 0
    while True:
        tick(count, env)
        cv2.waitKey(100)
    """
    uavs = UAVs()
    uavs.run()


if __name__ == '__main__':
    main()
