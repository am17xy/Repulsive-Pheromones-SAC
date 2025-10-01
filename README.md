# Repulsive-Pheromones-SAC
The code for SAC using repulsive-pheromones based action selection used in the paper with the working title 'Repulsive Pheromones-Based Action Selection for Multi-Objective Reinforcement Learning in Continuous State and Action Spaces'.

The code is implemented based on MORL-Baselines (original repository is found here: https://github.com/LucasAlegre/MORL-baselines) [1].

The file included this code was used in replacement of morl_baselines/single_policy/ser/mosac_continuous_action.py in MORL-Baselines. To determine which action-selection method should be used, modify lines 120-135 accordingly. By setting self.ph_action = False, the original SAC action selection method is used (sampling once from the actors probability distribution). Only set one of lin_inc_evap_rate, lin_dec_evap_rate, or adaptive_evap_rate to true at a time. If these all remain false while ph_action is true, then a static evaporation rate is used.

## References
[1] Felten, F., Alegre, L. N., Nowé, A., Bazzan, A. L. C., Talbi, E. G., Danoy, G., & Silva, B. C. da. (2023). A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning. In: _Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS 2023)_.
