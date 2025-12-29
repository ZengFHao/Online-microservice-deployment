import numpy as np
import paddle
import parl


ContainerNumber = 9
oldContainerNumber = 6
changeContainerNumber = 3
NodeNumber = 5
from policy import getQnetwork
from  policy import  merge_q

flag = []
flag_temp = []
for o in range(changeContainerNumber * NodeNumber):
    flag.append(0)
    flag_temp.append(0)


class Agent(parl.Agent):
    def __init__(self, algorithm_1, algorithm_2, algorithm_3, act_dim, e_greed=0.3, e_greed_decrement=0):
        """
            algorithm:传给agent的强化学习策略, Q-learning or 其他
            act_dim: 是一个整数, 表示在给定的状态下可执行的动作数量
            e_greed: 学习贪婪度, 介于0到1之间, 决定agent有多大概率随机跳转而不是model预测跳转
            e_greed_decrement: 贪婪度递减, 随着训练的进行逐渐减少e_greed的值, 如果不手动设置贪婪度在过程中不变
            global_step: 记录在训练过程中的总步数
        """
        super(Agent, self).__init__(algorithm_1)
        assert isinstance(act_dim, int)
        self.act_dim = act_dim
        self.alg_2 = algorithm_2
        self.alg_3 = algorithm_3

        self.global_step = 0
        self.update_target_steps = 200  
        self.e_greed = e_greed  
        self.e_greed_decrement = e_greed_decrement 
    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        sample = np.random.random()  
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  
            while flag[act] == 1 or flag_temp[act % changeContainerNumber] == 1: 
                act = np.random.randint(0, self.act_dim)
            flag[act] = 1
            flag_temp[act] = 1
            flag_temp[act % changeContainerNumber] = 1
        else:
            act = self.predict(obs)  
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement) 
        return act



    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        global flag
        global flag_temp
        obs = paddle.to_tensor(obs, dtype='float32')
        pred_q1 = self.alg.predict(obs)     
        pred_q2 = self.alg_2.predict(obs)
        pred_q3 = self.alg_3.predict(obs)

        act,flag,flag_temp=getQnetwork(1,pred_q1,pred_q2,pred_q3,act_dim=self.act_dim,flag=flag,flag_temp=flag_temp)


        return act

    def learn(self, obs, act, reward1, reward2, reward3, next_obs, terminal):
        """ 根据训练数据更新一次模型参数
        """
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
            self.alg_2.sync_target()
            self.alg_3.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward1 = np.expand_dims(reward1, axis=-1)
        reward2 = np.expand_dims(reward2, axis=-1)
        reward3 = np.expand_dims(reward3, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward1 = paddle.to_tensor(reward1, dtype='float32')
        reward2 = paddle.to_tensor(reward2, dtype='float32')
        reward3 = paddle.to_tensor(reward3, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        loss_1 = self.alg.learn(obs, act, reward1, next_obs, terminal)  
        loss_2 = self.alg_2.learn(obs, act, reward2, next_obs, terminal)
        loss_3 = self.alg_3.learn(obs, act, reward3, next_obs, terminal)
        return float(loss_1), float(loss_2), float(loss_3)
