# Toy Tabular Experiments for Tiered RL Setting
Code for Tiered RL paper https://arxiv.org/abs/2205.12418. If you find it helpful, please cite as follow:

```
@article{huang2022tiered,
  title={Tiered Reinforcement Learning: Pessimism in the Face of Uncertainty and Constant Regret},
  author={Huang, Jiawei and Zhao, Li and Qin, Tao and Chen, Wei and Jiang, Nan and Liu, Tie-Yan},
  journal={arXiv preprint arXiv:2205.12418},
  year={2022}
}
```

## How to run experiments


### Run experiments with different gap:
python Tabular_MDP.py -S 5 -A 5 -H 5 --use-tauP --alpha 0.25 --seed 99 199 299 399 499 599 699 799 899 999 --model-seed 1

python Tabular_MDP.py -S 5 -A 5 -H 5 --use-tauP --alpha 0.25 --seed 99 199 299 399 499 599 699 799 899 999 --model-seed 10

python Tabular_MDP.py -S 5 -A 5 -H 5 --use-tauP --alpha 0.25 --seed 99 199 299 399 499 599 699 799 899 999 --model-seed 100

### Plot results
python plot.py --log-dirs [Your Exp. Log Dirs]