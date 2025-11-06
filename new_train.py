#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import logging
import sys
from agent import Agent
from agent import flag
from agent import flag_temp
from algorithm import DQN  # from parl.algorithms import DQN  # parl >= 1.3.1
# 添加部分全局变量
# from env import ContainerNumber
from env import Env
from env import NodeNumber
from env import ResourceType
from env import e_greed
from env import e_greed_decrement
from model import Model
from pareto import getPareto
from replay_memory import ReplayMemory
from reward import Culreward

# from show import showPareto
# 检查版本

LEARN_FREQ = 0  # learning frequency
# MEMORY_SIZE = 10000  # size of replay memory
MEMORY_SIZE = 10000  # size of replay memory
MEMORY_WARMUP_SIZE = 4000
BATCH_SIZE = 128
LEARNING_RATE = 0.01
GAMMA = 0.9

pareto_set = []
feature_set = []
act_pareto_set = []
sc_comm = 0
sc_var = 0
flag1 = 1
ep = 0
round = 0
# allCost = [[], [], [], [], [], []]
allReward = [[], [], [], [], [], []]
rewardavg = 0


def run_train_episode(agent, env, rpm, start_obs, new_container):
    """ allCost:储每一步的成本
        ep: 当前训练周期的步数
        test_reward: 测试奖励
    """
    global flag1
    # allCost =     [[], [], [], [], [], []]
    global allReward  # allReward = [[], [], [], [], [], []]
    global ep
    global rewardavg
    # ------------

    global round

    obs_list = []
    next_obslist = []
    action_list = []
    done_list = []

    ep += 1
    obs, action = env.reset(start_obs, new_container)

    step = 0
    newContainerNumber = env.get_newContainer_number()
    ContainerNumber = env.get_container_number()
    LEARN_FREQ = newContainerNumber

    print(newContainerNumber)
    for o in range(newContainerNumber * NodeNumber):
        flag_temp[o] = 0
        flag[o] = 0
    flag1 -= 1
    while True:

        step += 1

        # 选择一种动作（随机或最优）
        # act[0]为节点号
        # act[1]为容器编号
        action = agent.sample(obs) #返回一个act
        # 与环境交互
        # container_state_queue中的-1变为该容器部署的节点号（nextobs中）
        # node_state_value中每8号代表一个节点，前六位为容器是否部署在该node（部署为1），后两位为节点的资源占用情况
        next_obs, feature1, feature2, feature3, done, feature1_1, feature1_2 = env.step(action)

        # 记录当前episode的数据
        obs_list.append(obs)
        action_list.append(action)
        next_obslist.append(next_obs)
        done_list.append(done)

        # ------------
        reward1 = 0
        reward2 = 0
        reward3 = 0

        if step == newContainerNumber:

            # feature1：35左右
            # feature2：35左右
            # feature3：35左右
            reward1, reward2, reward3 = Culreward(feature1, feature2, feature3)

            for i in range(newContainerNumber):
                rpm.append(
                     (obs_list[i], action_list[i], reward1, reward2, reward3, next_obslist[i], done_list[i]))
            print(action_list)

            # 输出到日志
            rewardsum = reward1 + reward3 + reward2
            rewardavg = (rewardsum + rewardavg * (ep - 1)) / ep

            root_logger = logging.getLogger()
            for h in root_logger.handlers[:]:
                root_logger.removeHandler(h)
            logging.basicConfig(level=logging.INFO, filename='feature-reward.log')
            logging.info(
                'episode:{} round:{} Ravg:{:.2f} Reward1:{:.2f} Reward2:{:.2f} Reward3:{:.2f} Feature1:{:.2f} Feature2:{:.2f} Feature3:{:.2f}'.format(
                    ep, round, rewardavg, reward1, reward2, reward3, feature1, feature2, feature3))

        # 如果rpm池已满，开始训练
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward1, batch_reward2, batch_reward3, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)

            loss1, loss2, loss3 = agent.learn(batch_obs, batch_action, batch_reward1, batch_reward2,
                                              batch_reward3,
                                              batch_next_obs,
                                              batch_done)  # s,a,r,s',done
            with open("trainloss.txt", "a") as f:
                f.write("ep:%d,loss1:%.3f,loss2:%.3f,loss3:%.3f \n" % (ep, loss1, loss2, loss3))

        obs = next_obs
        if done:
            break
    # 将action list转化为二元组形式
    templist = []
    for i in action_list:
        act = [-1, -1]
        act[0] = int(i / newContainerNumber)
        act[1] = i % newContainerNumber + 6
        templist.append(act)
    action_list = templist
    return feature1, feature2, feature3, reward1, reward2, reward3, action_list


def main(param, env, start_obs, new_container):
    global sc_comm, sc_var
    global rewardavg
    global act_pareto_set, pareto_set, feature_set
    # env = Env()
    ContainerNumber = env.get_container_number()
    newContainerNumber = env.get_newContainer_number()
    obs_shape = ContainerNumber * (ResourceType + 1) + NodeNumber * (
            ContainerNumber + 3) + ContainerNumber * 2  # *3对应containerstate数组，每个container三个值；后半对应nodestate数组
    action_dim = newContainerNumber * NodeNumber

    rpm = ReplayMemory(MEMORY_SIZE)  # Target1的经验回放池

    # 根据parl框架构建agent
    model_1 = Model(obs_shape, 128, 128, action_dim)
    model_2 = Model(obs_shape, 128, 128, action_dim)
    model_3 = Model(obs_shape, 128, 128, action_dim)
    algorithm_1 = DQN(model_1, gamma=GAMMA, lr=LEARNING_RATE)
    algorithm_2 = DQN(model_2, gamma=GAMMA, lr=LEARNING_RATE)
    algorithm_3 = DQN(model_3, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm_1, algorithm_2, algorithm_3,
        act_dim=action_dim,
        e_greed=e_greed,  # 有一定概率随机选取动作，探索
        e_greed_decrement=e_greed_decrement)  # type: ignore # 随着训练逐步收敛，探索的程度慢慢降低

    # 加载模型
    # save_path = './dqn_model.ckpt'
    # agent.restore(save_path)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        # MEMORY_WARMUP_SIZE=2000
        run_train_episode(agent, env, rpm, start_obs, new_container)

    max_episode = 3000

    # start train
    global round
    if param==0:
        text=",默认运行"
    else:
        text=",多次运行,第"+str(param)+"次"

    while round < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        #1.train part
        round += 1
        print("ep,round:" + str(ep) + " " + str(round)+text)
        #2.训练
        f1, f2, f3, r1, r2, r3, action_list = run_train_episode(agent, env, rpm, start_obs, new_container)

        #3.判断是否属于最优解集
        solution = [[f1, f2, f3], round, action_list]
        # set元素格式：[f1,f2,f3],round,[[],[],[],[],[],[]]]
        feature_set.append(solution)
        ispareto, pareto_set, removed_set = getPareto(feature_set, pareto_set, action_list)
        if ispareto:
            with open("pareto_details.txt", "a") as f:
                f.write("在round:" + str(round) + "增加了一个最优解：" + str(solution[0]) + "\n")
        if removed_set:
            for r in removed_set:
                with open("pareto_details.txt", "a") as f:
                    f.write("在round:" + str(round) + "移除了一个最优解：" + str(r[0]) + "round=" + str(r[1]) + "\n")



        #5.保存最后五次结果
        if round > (max_episode - 5):
            with open("all_trains.txt", "a") as f:
                f.write("final ravg" + str(round) + " :" + str(rewardavg) + str(action_list) + "\n")
    # 4.绘图
    # showPareto(pareto_set)
    # 根据索引值从小到大排序pareto集合
    pareto_set = sorted(pareto_set, key=lambda x: x[1])
    # 训练结束，保存模型，保存pareto_set
    with open("pareto_set.txt", "a", encoding='utf-8') as f:
        for a in pareto_set:
            f.write("round:" + str(a[1]) + "; act:" + str(a[2]) + "; feature:" + str(a[0]) + "\n")
    save_path = './mdqn_model.ckpt'
    agent.save(save_path)

print(sys.argv)
try:
    time=int(sys.argv[1])
except:
    time=0

if __name__ == '__main__':
    
    # start_obs = [3, 0.5, 128, 0.5, 
    #              0, 1, 128, 1.2, 
    #              1, 0.5, 256, 0.2, 
    #              0, 0.5, 256, 0.4, 
    #              1, 0.5, 256, 0.4, 
    #              0, 1, 128, 0.8, 
    #              
    #              0, 1, 0, 1, 0, 1, 2.5, 512, 2.4, 
    #              0, 0, 1, 0, 1, 0, 1.0, 512, 0.6, 
    #              0, 0, 0, 0, 0, 0, 0, 0, 0, 
    #              1, 0, 0, 0, 0, 0, 0.5, 128, 0.5, 
    #              0, 0, 0, 0, 0, 0, 0, 0 , 0, 
    #              
    #              8, 5, 8, 5, 8, 5,
    #              100, 200, 100, 200, 200, 100]

    #3个ms在线部署
    start_obs = [4, 0.5, 128, 0.5, 
                 0, 1, 128, 1.2, 
                 1, 0.5, 256, 0.2, 
                 2, 0.5, 256, 0.4, 
                 3, 0.5, 256, 0.4, 
                 2, 1, 128, 0.8, 

                 0, 1, 0, 0, 0, 0, 1.0, 128, 1.2, 
                 0, 0, 1, 0, 0, 0, 0.5, 256, 0.4,
                 0, 0, 0, 1, 0, 1, 1.5, 384, 1.2, 
                 0, 0, 0, 0, 1, 0, 0.5, 256, 0.4, 
                 1, 0, 0, 0, 0, 0, 0.5, 128 ,0.5, 
                 10, 5, 8, 15, 8, 15, 
                 150, 200, 100, 200, 100, 200]
    new_container = [[-1, 0.5, 128, 0.5],
                     [-1, 1, 128, 0.6,
                     -1, 0.5, 256, 0.2]]
    
    # env = Env(start_obs, new_container)
    # env.set_env(start_obs, new_container)

    #4个ms在线部署
    # start_obs = [4, 0.5, 128, 0.5, 
    #              0, 1, 128, 1.2, 
    #              1, 0.5, 256, 0.2, 
    #              2, 0.5, 256, 0.4, 
    #              3, 0.5, 256, 0.4, 
    #              2, 1, 128, 0.8, 
    #              0, 1, 0, 0, 0, 0, 1.0, 128, 1.2, 
    #              0, 0, 1, 0, 0, 0, 0.5, 256, 0.4,
    #              0, 0, 0, 1, 0, 1, 1.5, 384, 1.2, 
    #              0, 0, 0, 0, 1, 0, 0.5, 256, 0.4, 
    #              1, 0, 0, 0, 0, 0, 0.5, 128 ,0.5, 
    #              10, 5, 8, 15, 8, 15, 
    #              150, 200, 100, 200, 100, 200]
    # new_container = [[-1, 0.5, 128, 0.5],
    #                  [-1, 1, 128, 0.6,
    #                  -1, 0.5, 256, 0.2],
    #                  [-1, 0.5, 64, 0.2]]
    
    env = Env(start_obs, new_container)
    env.set_env(start_obs, new_container)
                                
    main(time, env, start_obs, new_container)

