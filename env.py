ContainerNumber = 6  # 容器数
NodeNumber = 5  # 节点数
# ServiceNumber = 6  # 服务种类
# ResourceType = 3  # 资源类型数,CPU,Mem,BandWidth
# service_containernum = [1, 1, 3, 1, 1, 2]  # 每种服务需要启动的容器数量（按编号索引）
# service_container = [[0], [1], [2, 3, 4], [5], [6], [7, 8]]  # 每种服务对应的容器编号（按编号索引）
# service_container_relationship = [0, 1, 2, 2, 2, 3, 4, 5, 5]  # 上面两个数组的对应
ServiceNumber = 4  # 服务种类
ResourceType = 3  # 资源类型数,CPU,Mem,BandWidth
service_containernum = [1, 1, 3, 1]  # 每种服务需要启动的容器数量（按编号索引）
service_container = [[0], [1], [2, 3, 4], [5]]  # 每种服务对应的容器编号（按编号索引）
service_container_relationship = [0, 1, 2, 2, 2, 3]  # 上面两个数组的对应
node_delay = [200, 100, 200, 100, 150]  # 每个节点的延迟
node_loss = [5, 8, 15, 8, 10]  # 每个节点的丢包%
alpha = 0.5  # reward weighting factor
beta = [0.5, 0.5]
count = 0
CPUnum = [4,2,2,1,1]
Mem = [1024,1024,512,512,512]
# BandWidth = [2,1,2,2,2]
BandWidth = [2,1,2,2,2]
e_greed = 0.3  # 模型学习率
e_greed_decrement = 1e-6

import numpy as np


class Env():
    def __init__(self, obs, new_container):
        # State
        global ContainerNumber
        self.State = []
        self.node_state_queue = []
        self.container_state_queue = []
        self.action_queue = []
        self.loss_state_query = []
        self.delay_state_query = []
        new_container_number = 0
        self.prepare(obs, new_container)

    def prepare(self,start_obs, new_container):
        #第一个元素表示是否已经部署
        # self.container_state_queue = [-1, 0.5, 128, 0.5,
        #                               -1, 1, 128, 1.2,
        #                               -1, 0.5, 256, 0.2,
        #                               -1, 0.5, 256, 0.4,
        #                               -1, 0.5, 256, 0.4,
        #                               -1, 1, 128, 0.8]  # 设置硬件条件
        print(len(start_obs) )
        #需要修改-3*新的微服务数目
        old_container_number = int((len(start_obs) - 15) / 11)
        self.new_container_number = self.calContainerNumber(new_container)
        node_state_index = 4 * old_container_number
        self.container_state_queue = start_obs[0 : node_state_index]
        old_loss_state = start_obs[len(start_obs) - old_container_number *2 : len(start_obs) - old_container_number]
        old_delay_state = start_obs[len(start_obs) - old_container_number : len(start_obs)]
        old_loss_state.extend(0 for i in range(0, self.new_container_number))
        old_delay_state.extend(0 for i in range(0, self.new_container_number))
        for sublist in new_container:
            for element in sublist:
                self.container_state_queue.append(element)
        for i in range(NodeNumber):
            # self.node_state_queue.extend([0, 0, 0, 0, 0, 0, 0, 0])
            """ 0-5表示容器部署情况
                6-9表示该节点的资源占用情况
            """
            # self.node_state_queue.extend([0,0,0,0,0,0,0,0,0])  # 微服务启动容器数+3
            old_node_state = start_obs[node_state_index : node_state_index + old_container_number + 3]
            for i in range(self.new_container_number):
                old_node_state.insert(len(old_node_state) - 3, 0)
            self.node_state_queue.extend(old_node_state)
        self.loss_state_query = old_loss_state   #每个容器的丢包率
        self.delay_state_query = old_delay_state  #每个容器的延迟
        self.State = self.container_state_queue + self.node_state_queue + self.loss_state_query + self.delay_state_query    #一个state的所有环境情况
        self.action = [-1, -1]  #[a,b]表示在a节点上部署b容器
        self.action_queue = [-1, -1]    #记录部署动作

        # Communication weight between microservices
        # 四种微服务间的通信成本
        self.service_weight = [[0, 1, 0, 0, 0.5, 0.8], [1, 0, 1, 0, 0.6, 0.4], [0, 1, 0, 0.8, 0.3, 0], [0, 0, 0.8, 0, 0.2, 0], [0.5, 0.6, 0.3, 0.2, 0, 1], [0.8, 0.4, 0, 0, 1, 0]]  
        # Communication distance between nodes
        # 节点间的（物理）通信距离
        self.Dist = [[0, 200, 100, 300, 400], [200, 0, 200, 100, 150], [100, 200, 0, 250, 300], [300, 100, 250, 0, 350], [400, 150, 300, 350, 0]]
    
    def calContainerNumber(self, new_container):
        new_container_number = 0 
        for i in range(len(new_container)):
            container = int(len(new_container[i]) / 4)
            new_container_number += container
        return new_container_number
    
    '''
        重设全局变量
    '''
    def set_env(self, obs, new_container):
        global ServiceNumber
        global ContainerNumber
        old_container_number = int((len(obs) - 15) / 11)
        new_service_number = len(new_container)
        new_container_number = 0
        for i in range(new_service_number):
            container = int(len(new_container[i]) / 4)
            new_container_number += container
            service_containernum.append(container)
            last_element = service_container[-1][-1]
            service_container.append([last_element + i for i in range(1, container + 1)])
            last_element = service_container_relationship[-1]
            service_container_relationship.extend(last_element + 1 for i in range(1, container + 1))
        ContainerNumber = old_container_number + new_container_number
        ServiceNumber += new_service_number 
        
   
    def ContainerCost(self, i, j):
        # to calculate the distance between container i and j
        m = -1
        n = -1
        m = self.container_state_queue[i * (ResourceType + 1)]
        n = self.container_state_queue[j * (ResourceType + 1)]

        p = service_container_relationship[i]
        q = service_container_relationship[j]

        if self.Dist[m][n] != 0 and (p != q):
            container_dist = self.Dist[m][n]
        else:
            container_dist = 0
        return container_dist

    def CalcuCost(self, i, j):
        # to calculate the communication cost between container i and j(我觉得是microservice之间的通讯开销)
        cost = 0
        interaction = self.service_weight[i][j] / (service_containernum[i] * service_containernum[j])
        for k in range(len(service_container[i])):
            for l in range(len(service_container[j])):
                cost += self.ContainerCost(service_container[i][k], service_container[j][l]) * interaction
        return cost

    def sumCost(self):
        Cost = 0
        for i in range(ServiceNumber):
            for j in range(ServiceNumber):
                Cost += self.CalcuCost(i, j)
        return 0.5 * Cost

    def CalcuVar(self):
        NodeCPU = []
        NodeMemory = []
        NodeBandWith = []
        Var = 0
        for i in range(NodeNumber):
            U = self.node_state_queue[i * (ContainerNumber + 3) + ContainerNumber]
            M = self.node_state_queue[i * (ContainerNumber + 3) + (ContainerNumber + 1)]
            B = self.node_state_queue[i * (ContainerNumber + 3) + (ContainerNumber + 2)]
            NodeCPU.append(U)
            NodeMemory.append(M)
            NodeBandWith.append(B)
        
            # if NodeCPU[i] < CPUnum[i] or NodeMemory[i] < Mem[i] or NodeBandWith[i] < BandWidth[i]:
            #      Var -= 100
            if NodeCPU[i] > CPUnum[i] or NodeMemory[i] > Mem[i]:
                Var -= 1
                # Variance of node loads
            if NodeBandWith[i] < BandWidth[i]:
                Var += 10
        Var += beta[0] * np.var(NodeCPU) * 1000 + beta[1] * np.var(NodeMemory) / 100
        return Var, NodeCPU, NodeMemory, NodeBandWith

    def CalcuCostFin(self):
        """ g1: 微服务之间总的通讯开销
            g2: 负载均衡的方差
            re: 奖励函数
        """
        re = 0
        g1 = self.sumCost() #
        g1 /= 10
        # g1 = (g1 - 1300) / (1800 - 1300)
        g2, NodeCPU, NodeMemory, NodeBandWith = self.CalcuVar()
        # g2 = 0
        # g2 /= 1000
        # g2 = - (g2 - (-500)) / (-300 - (-500))
        re += alpha * g1 + (1 - alpha) * g2
        return re, g1, g2

    def state_update(self, container_state_queue, node_state_queue, loss_state_query, delay_state_query):
        self.State = container_state_queue + node_state_queue + loss_state_query + delay_state_query

    def update(self):
        # update state
        if self.action[0] >= 0 and self.action[1] >= 0:
            # update container state
            self.container_state_queue[self.action[1] * (ResourceType + 1)] = self.action[0]
            # update node state
            """
                依次修改容器的部署情况
                更新该节点的CPU占有情况
                更新该节点的内存占有情况
                更新该节点的带宽占有情况
            """
            self.node_state_queue[self.action[0] * (ContainerNumber + 3) + self.action[1]] = 1
            self.node_state_queue[self.action[0] * (ContainerNumber + 3) + ContainerNumber] += self.container_state_queue[self.action[1] * (ResourceType + 1) + 1]
            self.node_state_queue[self.action[0] * (ContainerNumber + 3) + (ContainerNumber + 1)] += self.container_state_queue[self.action[1] * (ResourceType + 1) + 2]
            self.node_state_queue[self.action[0] * (ContainerNumber + 3) + (ContainerNumber + 2)] += self.container_state_queue[self.action[1] * (ResourceType + 1) + 3]
            self.loss_state_query[self.action[1]] = node_loss[self.action[0]]
            self.delay_state_query[self.action[1]] = node_delay[self.action[0]]
            self.action_queue.append(self.action)
        else:
            print("invalid action")
            self.node_state_queue = []
            self.container_state_queue = []
            self.delay_state_query = []
            self.loss_state_query = []
            self.action_queue = []

            self.prepare()
        self.state_update(self.container_state_queue, self.node_state_queue, self.loss_state_query, self.delay_state_query)
        return self.State

    # def CalcuLoss(self):
    #     loss = node_loss[self.action[0]]
    #     feature = (40 - loss) * 12
    #     return feature

    # def CalcuDelay(self):
    #     # act[0]为部署在几号节点上，act[1]为部署第几个微服务容器实例
    #     delay = node_delay[self.action[0]]
    #     feature = (500 - delay) / 10 / ContainerNumber
    #     return 
    
    """要修改！！！
    """
    def cal_loss(self):
        return self.loss_state_query[6]*2 + self.loss_state_query[7]*1.5 + self.loss_state_query[8]*1.8
        # return self.loss_state_query[6] * 0 + self.loss_state_query[7] * 0 + self.loss_state_query[8] * 0
        
    def cal_delay(self):
        return self.delay_state_query[6] * 2 + self.delay_state_query[7] * 1.5 + self.delay_state_query[8] * 1.8
        # return self.delay_state_query[6] * 0 + self.delay_state_query[7] * 0 + self.delay_state_query[8] * 0
    
    # def cal_loss(self):
    #     return self.loss_state_query[0]*1 + self.loss_state_query[1]*2 + self.loss_state_query[2]*0.2 + self.loss_state_query[3]*1.8 +  self.loss_state_query[4]*1.8 + self.loss_state_query[5]*1.6 
        
    # def cal_delay(self):
    #     return self.delay_state_query[0]*1 + self.delay_state_query[1]*2 + self.delay_state_query[2]*0.2 + self.delay_state_query[3]*1.8 +  self.delay_state_query[4]*1.8 + self.delay_state_query[5]*1.6
      
    def step(self, action):
        # input: action(Targetnode，ContainerIndex)
        # output: next state, cost, done
        global count
        self.action = self.index_to_act(action)
        self.update()

        feature1, feature1_1, feature1_2 = self.CalcuCostFin()  # 35左右
        feature2 = self.cal_loss()   # 35/6左右
        feature3 = self.cal_delay()     # 35/6左右

        done = False
        count = 0

        # 判断当前为第几步,用来判断是否完成迭代
        for i in range(ContainerNumber):
            if self.container_state_queue[(ResourceType + 1) * i] != -1:
                count += 1
        if count == ContainerNumber:
            done = True

        # 返回当前的状态、通信开销以及资源方差、loss和、延迟、通讯开销g1、负载均衡方差g2
        return self.State, feature1, feature2, feature3, done, feature1_1, feature1_2

    def reset(self, start_obs, new_container):
        self.node_state_queue = []
        self.container_state_queue = []
        self.prepare(start_obs, new_container)
        return self.State, self.action

    def index_to_act(self, index):
        act = [-1, -1]
        act[0] = int(index / self.new_container_number)
        act[1] = index % self.new_container_number + 6
        return act

    def get_container_number(self):
        return ContainerNumber
    def get_newContainer_number(self):
        return self.new_container_number