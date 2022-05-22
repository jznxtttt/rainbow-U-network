import math
import os
import random
from tqdm import tqdm
from collections import deque
from typing import Deque, Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from rainbow_stock_no_adjust2.replay_buffer.rainbow_replay_buffer_stock import ReplayBuffer
from rainbow_stock_no_adjust2.noisy_distribution_net.noisy_net import NoisyLinear, StockUNetwork

class DQNStockTradingAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        auxiliary_dqn (Network): model to train and select actions, by which one can calculate Q func
        auxiliary_dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer

    """

    def __init__(
        self, 
        obs_dim: int,
        act_dim: int,
        env,
        test_env, 
        memory_size: int,
        batch_size: int,
        target_update: int,
        gamma: float = 0.99,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
        test_time: int = 50000,
        save_folder: str = "/",
        is_from_load: bool = True,
        is_to_save: bool = True,
        dqn_load_name: str = "dqn_net.pth",
        dqn_target_load_name: str = "dqn_target_net.pth",
        dqn_save_name: str = "dqn_net.pth",
        dqn_target_save_name: str = "dqn_target_net.pth",
        is_need_test_in_train: bool = False,
        is_need_test_in_test: bool = True,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        
        self.env = env
        self.test_env = test_env
        self.fee_rate = self.env.fee_rate

        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)
        
        self.memory_size = memory_size
        
        # memory for N-step Learning
        self.use_n_step = True
        
        self.n_step = n_step
        self.memory_n = ReplayBuffer(obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma)
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # Auxiliary Q functions
        # action dimension is 3, 0: do nothing, 1: long, 2: short
        self.auxiliary_dqn = StockUNetwork(
            obs_dim, act_dim, self.atom_size, self.support
        ).to(self.device)
        self.auxiliary_dqn_target = StockUNetwork(
            obs_dim, act_dim, self.atom_size, self.support
        ).to(self.device)
        self.auxiliary_dqn_target.load_state_dict(self.auxiliary_dqn.state_dict())
        self.auxiliary_dqn_target.eval()
        
        # optimizer
        self.lambda0 = 0.00001
        self.optimizer = optim.Adam(self.auxiliary_dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

        # test time per batch
        self.test_time = test_time

        # folder
        self.save_folder = save_folder

        # e- greedy
        self.episilon = 0.01

        self.frame_idx = 0

        # test
        self.is_need_test_in_train = is_need_test_in_train
        self.is_need_test_in_test = is_need_test_in_test

        # save network
        self.is_from_load = is_from_load
        self.is_to_save = is_to_save
        self.dqn_load_name = dqn_load_name
        self.dqn_target_load_name = dqn_target_load_name
        self.dqn_save_name = dqn_save_name
        self.dqn_target_save_name = dqn_target_save_name
        if self.is_from_load:
            self.load_net()
    
    def save_net(self):
        print("save net ...")
        torch.save(self.auxiliary_dqn.state_dict(),str(self.frame_idx)+"_"+self.dqn_save_name)
        torch.save(self.auxiliary_dqn_target.state_dict(),str(self.frame_idx)+"_"+self.dqn_target_save_name)
        

    def load_net(self):
        print("load net...")
        self.auxiliary_dqn.load_state_dict(torch.load(self.dqn_load_name))
        self.auxiliary_dqn_target.load_state_dict(torch.load(self.dqn_target_load_name))

        
    def select_action(self, state: np.ndarray, mid: int, buy_pos: int, sell_pos: int) -> np.ndarray:
        """Select an action from the input state."""
        if not self.is_test:
            rand = random.uniform(0,1)
            if rand < self.episilon:
                selected_action = random.randint(0,2)
                self.transition = [state, mid, buy_pos, sell_pos, selected_action]
            else:
                state_tensor = torch.tensor(state,dtype = torch.float).to(self.device)
                mid_tensor = torch.tensor(mid, dtype = torch.float).to(self.device).unsqueeze(0)
                buy_pos_tensor = torch.tensor(buy_pos, dtype = torch.long).to(self.device).unsqueeze(0)
                sell_pos_tensor = torch.tensor(sell_pos, dtype = torch.long).to(self.device).unsqueeze(0)
                Q_value = self._get_Q_value(state_tensor, mid_tensor, buy_pos_tensor, sell_pos_tensor, mode = "left")
                selected_action = Q_value.argmax(dim = 1)
                self.transition = [state, mid, buy_pos, sell_pos, selected_action]

        else:
        # NoisyNet: no epsilon greedy action selection
        # Q_value: (batch_size, 3)
            state = torch.tensor(state,dtype = torch.float).to(self.device)
            mid = torch.tensor(mid, dtype = torch.float).to(self.device).unsqueeze(0)
            buy_pos = torch.tensor(buy_pos, dtype = torch.long).to(self.device).unsqueeze(0)
            sell_pos = torch.tensor(sell_pos, dtype = torch.long).to(self.device).unsqueeze(0)
            Q_value = self._get_Q_value(state, mid, buy_pos, sell_pos, mode  = "left")
            selected_action = Q_value.argmax(dim = 1)
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, np.float64, np.float64, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, next_mid, next_pos_buy, next_pos_sell, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, next_mid, next_pos_buy, next_pos_sell, done]
            
            self.memory_n.store(*self.transition)
    
        return next_state, next_mid, next_pos_buy, next_pos_sell, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""

        gamma = self.gamma ** self.n_step
        samples = self.memory_n.sample_batch()
        
        elementwise_loss = self._compute_dqn_loss(samples, gamma)
        if elementwise_loss is None:
            return 

        loss = torch.mean(elementwise_loss)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.auxiliary_dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # NoisyNet: reset noise
        self.auxiliary_dqn.reset_noise()
        self.auxiliary_dqn_target.reset_noise()

        return loss.item()
        
    def train(self, num_frames: int):
        """Train the agent."""
        self.is_test = False
        
        state, mid, bp, sp = self.env.reset()
        update_cnt = 0
        

        for frame_idx in range(1, num_frames + 1):
            self.frame_idx = frame_idx
            if frame_idx % 1000 == 0:
                print("training:",frame_idx,"/",num_frames)
            action = self.select_action(state, np.array(mid), np.array(bp), np.array(sp))
            next_state, next_mid, next_bp, next_sp, reward, done = self.step(action)

            state = next_state
            mid = next_mid
            bp = next_bp
            sp = next_sp
            
            # if episode ends
            if done:
                state, mid, bp, sp = self.env.reset()

            # if training is ready
            if len(self.memory_n) >= 0.8*self.memory_size and frame_idx % 100 == 0:
                loss = self.update_model()
                update_cnt += 1
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()
            
            if frame_idx % self.test_time == 0:
                self.is_test = True
                self.auxiliary_dqn.eval()
                self.fee_rate = self.test_env.fee_rate
                
                if self.is_need_test_in_train:
                    self.test_in_train()
                if self.is_need_test_in_test:
                    self.test_in_test()
                
                self.is_test = False
                self.auxiliary_dqn.train()
                self.fee_rate = self.env.fee_rate
                if self.is_to_save:
                    self.save_net()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        holdings = samples["acts"]+ samples["pos_buy"] - samples["pos_sell"]
        indices = np.abs(holdings)>1e-4
        if not indices.any():
            print("[Warning]: no position!")
            return

        state = torch.FloatTensor(samples["obs"][indices]).to(device)
        mid = torch.tensor(samples["mid"][indices], dtype = torch.float).to(device)
        buy_pos = torch.tensor(samples["pos_buy"][indices], dtype = torch.long).to(device)
        sell_pos  = torch.tensor(samples["pos_sell"][indices], dtype = torch.long).to(device)
        next_state = torch.FloatTensor(samples["next_obs"][indices]).to(device)
        next_mid = torch.tensor(samples["next_mid"][indices],dtype = torch.float).to(device)
        next_buy_pos = torch.tensor(samples["next_pos_buy"][indices], dtype = torch.long).to(device)
        next_sell_pos = torch.tensor(samples["next_pos_sell"][indices], dtype = torch.long).to(device)
        action = torch.LongTensor(samples["acts"][indices]).to(device)
        reward = torch.FloatTensor(samples["rews"][indices]).to(device)
        
        new_batch_size = state.shape[0]
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            # state has none

            # e_t: (batch_size, )
            e_t = buy_pos - sell_pos
            next_e_t = next_buy_pos - next_sell_pos

            # next_action: (batch_size,)
            next_action = self._get_Q_value(next_state, next_mid, next_buy_pos, next_sell_pos, mode = "right").argmax(1)
            
            # next_dist: (batch_size, 1, atom_size)
            next_dist = self._get_Q_distribution(next_state, mode = "right")
            next_dist = next_dist[range(new_batch_size),0]
            
            action = torch.where(action == 2, -1, action)
            next_action = torch.where(next_action == 2, -1, next_action)

            holdings = e_t + action
            next_holdings = next_e_t + next_action

            # get target distribution
            fee = torch.abs(action) * self.fee_rate * mid
            next_fee = torch.abs(next_action) * self.fee_rate * mid
            t_z = (reward + fee - gamma * next_fee).unsqueeze(dim = 1).expand(new_batch_size,self.atom_size)

            next_holdings = next_holdings.unsqueeze(dim = 1).expand(new_batch_size,self.atom_size)
            holdings = holdings.unsqueeze(dim = 1).expand(new_batch_size,self.atom_size)
            t_z = t_z + gamma * next_holdings * self.support

            auxiliary_t_z = t_z.clone()
            # auxiliary_t_z: (batch_size, atom_size)
            auxiliary_t_z = t_z / holdings 
            auxiliary_t_z = auxiliary_t_z.clamp(min=self.v_min, max=self.v_max)
            # b, l, u: (batch_size, atom_size)
            b = (auxiliary_t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            
            # offset: (batch_size, atom_size)
            offset = (
                torch.linspace(
                    0, (new_batch_size - 1) * self.atom_size, new_batch_size
                ).long()
                .unsqueeze(1)
                .expand(new_batch_size, self.atom_size)
                .to(self.device)
            )
            # this need implement
            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )
        dist = self._get_Q_distribution(state, mode = "left")
        dist = dist[range(new_batch_size),0]
        log_p = torch.log(dist)
        elementwise_loss = -(proj_dist * log_p).sum(1)

        regularization_loss = 0
        for param in self.auxiliary_dqn.parameters():
            regularization_loss += torch.sum(torch.abs(param))
        
        loss = elementwise_loss + self.lambda0 * regularization_loss
        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        print("target_hard_update")
        self.auxiliary_dqn_target.load_state_dict(self.auxiliary_dqn.state_dict())
    
    def _get_Q_value(self,
        state: torch.tensor,
        mid: torch.tensor,
        buy_pos: torch.tensor,
        sell_pos: torch.tensor,
        mode: str,
    ):
        e_t = buy_pos - sell_pos
        batch_size = e_t.size()[0]
        if mode=="left":
            U_value = self.auxiliary_dqn(state)
        if mode == "right":
            U_value = self.auxiliary_dqn_target(state)
        # U_value: (batch_size,1)
        none_risk_exposure = e_t
        buy_risk_exposure = e_t + 1
        sell_risk_exposure = e_t - 1
        new_risk_exposure = torch.stack((none_risk_exposure,buy_risk_exposure,sell_risk_exposure), dim = 1)
        # new_risk_exposure: (batch_size,3)

        Q_value = new_risk_exposure * U_value
        
        none_fee = torch.tensor(np.zeros(batch_size), dtype = torch.float).to(self.device)
        buy_fee = none_fee + self.fee_rate * mid
        sell_fee = none_fee + self.fee_rate * mid
        fee = torch.stack((none_fee,buy_fee,sell_fee), dim = 1)
        Q_value = Q_value - fee
        return Q_value
    
    def _get_Q_distribution(self,
        state: torch.tensor,
        mode: str,
    ):
        '''
        state: (batch_size, obs_dim)
        U_distribution: (batch_size, 3, atom_size)
        '''
        if mode == "right":
            Q_distribution = self.auxiliary_dqn_target.dist(state)
        if mode == "left":
            Q_distribution = self.auxiliary_dqn.dist(state)
        
        return Q_distribution
    
    def test_in_train(self):
        pnl = 0
        pnl_list = []
        for day in range(len(self.env.stock_price_arr_list)):
            state, mid, bp, sp = self.env.reset(day)
            
            q_list = []
            mid_list = []
            cash_list = []
            buy_list=[]
            sell_list=[]

            while True:
                action = self.select_action(state, mid, bp, sp)
                next_state, next_mid, next_bp, next_sp, reward, done, info = self.env.step(action)
                state = torch.tensor(state,dtype = torch.float).to(self.device)
                U_value = self.auxiliary_dqn(state).cpu().detach().numpy().reshape(-1)
                
                q_list.append(U_value)
                mid_list.append(mid)
                cash_list.append(info)
                buy_list.append(bp)
                sell_list.append(sp)
                
                state = next_state
                mid = next_mid
                bp = next_bp
                sp = next_sp
                
                if done:
                    break
            
            buy_point=[]
            sell_point=[]
            for i in range(len(buy_list)-1):
                if buy_list[i+1]>buy_list[i]:
                    buy_point.append(i)
            for i in range(len(sell_list)-1):
                if sell_list[i+1]>sell_list[i]:
                    sell_point.append(i)
            
            fee_u = self.fee_rate * np.array(mid_list)
            nega_fee_u = -self.fee_rate * np.array(mid_list)

            print("day",self.env.date_list[day])
            print("total_pnl:",cash_list[-1])
            print("total_pos:",max(len(buy_point),len(sell_point)))
            plt.figure(figsize=(15,9))
            plt.subplot(221)
            plt.plot(q_list, label = "buy")
            plt.plot(fee_u, linestyle= "--", color = "yellow")
            plt.plot(nega_fee_u, linestyle = "--", color = "yellow")
            
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("q_value")
            plt.title("q_value")
            plt.legend()

            plt.subplot(222)
            plt.plot(cash_list, label = "pnl")
            plt.xlabel("time")
            plt.ylabel("cash_money")
            plt.grid()
            plt.title("pnl")
            plt.legend()

            plt.subplot(223)
            plt.plot(mid_list, label = "mid")
            plt.grid()
            plt.legend()
            plt.xlabel("time")
            plt.ylabel("mid")
            plt.title("mid")

            plt.subplot(224)
            plt.plot(mid_list,label="mid")
            for i in buy_point:
                plt.scatter(i,mid_list[i]-0.05,color="red",marker="^")
            for i in sell_point:
                plt.scatter(i,mid_list[i]+0.05,color="blue",marker="v")
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("mid")
            plt.title("buy & sell point")
            plt.legend()

            plt.savefig(self.save_folder+str(self.frame_idx)+"_"+self.env.date_list[day]+".png")
            plt.close()

            pnl += cash_list[-1]
            pnl_list.append(pnl)
        plt.figure(figsize=(12,8))
        plt.plot(pnl_list)
        plt.xlabel("day")
        plt.ylabel("pnl")
        plt.savefig(self.save_folder+str(self.frame_idx)+"_allpnl.png")

    def test_in_test(self):
        pnl = 0
        pnl_list = []
        daily_pos_list = []
        for day in range(len(self.test_env.stock_price_arr_list)):
            state, mid, bp, sp = self.test_env.reset(day)
            
            q_list_0 = []
            mid_list = []
            cash_list = []
            buy_list=[]
            sell_list=[]

            while True:
                action = self.select_action(state, mid, bp, sp)
                next_state, next_mid, next_bp, next_sp, reward, done, info = self.test_env.step(action)
                state = torch.tensor(state,dtype = torch.float).to(self.device)
                U_value = self.auxiliary_dqn(state).cpu().detach().numpy().reshape(-1)
                q_list_0.append(U_value)
                mid_list.append(mid)
                cash_list.append(info)
                buy_list.append(bp)
                sell_list.append(sp)
                
                state = next_state
                mid = next_mid
                bp = next_bp
                sp = next_sp

                if done:
                    break
            
            buy_point=[]
            sell_point=[]
            for i in range(len(buy_list)-1):
                if buy_list[i+1]>buy_list[i]:
                    buy_point.append(i)
            for i in range(len(sell_list)-1):
                if sell_list[i+1]>sell_list[i]:
                    sell_point.append(i)
            
            fee_u = self.fee_rate * np.array(mid_list) 
            nega_fee_u = -self.fee_rate * np.array(mid_list)

            print("day",self.test_env.date_list[day])
            print("total_pnl:",cash_list[-1])
            print("total_pos:",max(len(buy_point),len(sell_point)))
            plt.figure(figsize=(15,9))
            plt.subplot(221)
            plt.plot(q_list_0, label = "none")
            plt.plot(fee_u, linestyle= "--", color = "yellow")
            plt.plot(nega_fee_u, linestyle = "--", color = "yellow")
            
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("q_value")
            plt.title("q_value")
            plt.legend()

            plt.subplot(222)
            plt.plot(cash_list, label = "pnl")
            plt.xlabel("time")
            plt.ylabel("cash_money")
            plt.grid()
            plt.title("pnl")
            plt.legend()

            plt.subplot(223)
            plt.plot(mid_list, label = "mid")
            plt.grid()
            plt.legend()
            plt.xlabel("time")
            plt.ylabel("mid")
            plt.title("mid")

            plt.subplot(224)
            plt.plot(mid_list,label="mid")
            for i in buy_point:
                plt.scatter(i,mid_list[i]-0.05,color="red",marker="^")
            for i in sell_point:
                plt.scatter(i,mid_list[i]+0.05,color="blue",marker="v")
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("mid")
            plt.title("buy & sell point")
            plt.legend()

            plt.savefig(self.save_folder+str(self.frame_idx)+"_test_"+self.test_env.date_list[day]+".png")
            plt.close()

            pnl += cash_list[-1]
            pnl_list.append(pnl)
            daily_pos_list.append(max(len(buy_point),len(sell_point)))

            np.save(self.save_folder+"qlist_rainbow"+self.test_env.date_list[day]+".npy",np.array(q_list_0))

        plt.figure(figsize=(12,8))
        plt.plot(pnl_list)
        plt.xlabel("day")
        plt.ylabel("pnl")
        plt.savefig(self.save_folder+str(self.frame_idx)+"_test_allpnl.png")
        np.save(self.save_folder+str(len(self.test_env.stock_price_arr_list))+"rainbow_pnl_list.npy",np.array(pnl_list))
        np.save(self.save_folder+str(len(self.test_env.stock_price_arr_list))+"rainbow_daily_pos.npy",np.array(daily_pos_list))


        