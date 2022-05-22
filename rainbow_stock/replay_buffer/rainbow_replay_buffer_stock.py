import math
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 1, 
        gamma: float = 0.99
    ):
        # s buffer + buy_pos buffer + sell_pos buffer
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.mid_buf = np.zeros([size],dtype = np.float32)
        self.buy_pos_buf=np.zeros([size],dtype=np.float32)
        self.sell_pos_buf=np.zeros([size], dtype=np.float32)
        
        # s_ buffer + next buy_pos buffer + next sell_pos buffer
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_mid_buf = np.zeros([size], dtype = np.float32)
        self.next_buy_pos_buf=np.zeros([size],dtype=np.float32)
        self.next_sell_pos_buf=np.zeros([size],dtype=np.float32)
        
        # a, r, done buffer
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray, 
        mid: np.ndarray,
        buy_pos: np.ndarray,
        sell_pos: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        next_mid: np.ndarray,
        next_buy_pos: np.ndarray,
        next_sell_pos: np.ndarray,
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, mid, buy_pos, sell_pos, act, rew, next_obs, next_mid, next_buy_pos, next_sell_pos, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        obs, mid, buy_pos, sell_pos, act = self.n_step_buffer[0][:5]
        rew, next_obs, next_mid, next_buy_pos , next_sell_pos, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        
        # s, mid, bp, sp
        self.obs_buf[self.ptr] = obs
        self.mid_buf[self.ptr] = mid
        self.buy_pos_buf[self.ptr] = buy_pos
        self.sell_pos_buf[self.ptr] = sell_pos
        
        # s_, mid_, bp_, sp_
        self.next_obs_buf[self.ptr] = next_obs
        self.next_mid_buf[self.ptr] = next_mid
        self.next_buy_pos_buf[self.ptr] = next_buy_pos
        self.next_sell_pos_buf[self.ptr] = next_sell_pos

        # a, r, done
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            mid=self.mid_buf[idxs],
            pos_buy=self.buy_pos_buf[idxs],
            pos_sell=self.sell_pos_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            next_mid=self.next_mid_buf[idxs],
            next_pos_buy=self.next_buy_pos_buf[idxs],
            next_pos_sell=self.next_sell_pos_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )
    
    def sample_batch_from_idxs(
        self, idxs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            mid=self.mid_buf[idxs],
            pos_buy=self.buy_pos_buf[idxs],
            pos_sell=self.sell_pos_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            next_mid=self.next_mid_buf[idxs],
            next_pos_buy=self.next_buy_pos_buf[idxs],
            next_pos_sell=self.next_sell_pos_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step r, next_obs, next_bp , next_sp, and done."""
        # info of the last transition
        rew, next_obs, next_mid, next_buy_pos, next_sell_pos, done = n_step_buffer[-1][-6:]

        # when done, 
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, n_m, n_b, n_s, d = transition[-6:]

            rew = r + gamma * rew * (1 - d)
            if d:
                next_obs, next_mid, next_buy_pos, next_sell_pos, done = n_o, n_m, n_b, n_s, d

        return rew, next_obs, next_mid, next_buy_pos, next_sell_pos, done

    def __len__(self) -> int:
        return self.size