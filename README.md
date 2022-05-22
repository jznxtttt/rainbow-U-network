# rainbow-U-network
## Introduction
- This package is used to building quantitative trading strategies by reinforcement learning method. 
- Our result shows that rainbow method has several advantages.
-     1. Fuzzy learning. Rainbow method models the distribution of returns instead of the expected value.
-     2. Balancing between short term reward and long term reward. This helps to optimize the strategy especially when holding interval and predicting interval are not the same and when the fee rate is high.
- This code is modified by rainbow network https://github.com/Curt-Park/rainbow-is-all-you-need.

## Thesis
Let $s_t$ be the state space, $e_t$ be your holdings at time $t$, $a_t$ be the action of orders, $m_t$ be the mid price of the instrument of time $t$.
The Q-function can be simplified as follows:
$$
Q(s_t,e_t,m_t,a_t) = (e_t+a_t) U(s_t) - |a_t|m_tf,
$$
where f represents fee rate.

This method trains U-function instead of Q-function.
