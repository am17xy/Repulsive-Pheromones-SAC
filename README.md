# Repulsive-Pheromones-SAC
The code for SAC using repulsive-pheromones based action selection used in the paper with the working title 'Repulsive Pheromones-Based Action Selection for Multi-Objective Reinforcement Learning in Continuous State and Action Spaces'.

The code is implemented using MORL-Baselines (original repository is found here: [MORL-Baselines](https://github.com/LucasAlegre/MORL-baselines)) [1].

The code included in this reposity was used in replacement of morl_baselines/single_policy/ser/mosac_continuous_action.py in the MORL-Baselines package. To select the action selection method, modify lines 120-135 accordingly, noting the following:
- By setting `self.ph_action = False`, the original SAC action selection method is used (i.e. sampling once from the actor's probability distribution).
- Only set one of `self.lin_inc_evap_rate`, `self.lin_dec_evap_rate`, or `self.adaptive_evap_rate` to `True` at once.
- To use a static evaporation rate, set `self.ph_action = True` and all other evap_rate flags to `False`.

The MORL/D implementation from MORL-Baselines was used, and can be found here: [MORL/D](https://github.com/LucasAlegre/morl-baselines/blob/main/morl_baselines/multi_policy/morld/morld.py).

## License
This project uses code from MORL-Baselines (MIT License) and is modified under GPL v3.0. Please refer to the License file for full details.

## References
[1] Felten, F., Alegre, L. N., Nowé, A., Bazzan, A. L. C., Talbi, E. G., Danoy, G., & Silva, B. C. da. (2023). A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning. In: _Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS 2023)_.
