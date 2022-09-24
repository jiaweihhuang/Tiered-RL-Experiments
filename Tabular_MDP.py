import numpy as np
import argparse
import os
from LoggerClass import Logger
import random
 
'''
The design of MDP:
 
at each state, the first action is optimal, whose transition probability is [0.5, ...]
at each layer, S1 is the "optimal state", where each action has reward 1, while the others has 0.5

'''
INT_MAX = np.iinfo(np.int64).max
 
class Tabular_MDP():
    def __init__(self, S=3, A=4, H=5):
        self.P = []
        self.R = []
        self.S = S
        self.A = A
        self.H = H
 
        for h in range(self.H):
            # Ph[s,a,s'] = P(s'|s,a)
            Ph = np.random.randint(1, 10, size=[S, A, S]) * 1.0
            # Rh[s,a] = R(s,a)
            Rh = np.random.randint(10, size=[S, A]) / 10.0
 
            for s in range(S):
                for a in range(A):
                    Ph[s][a] = Ph[s][a] / np.sum(Ph[s][a])         # normalization
 
            self.R.append(Rh)
            self.P.append(Ph)
 
        self.compute_optimal_policy()
 
    def compute_optimal_policy(self):
        self.opt_pi = []
        self.opt_Q = []
        self.opt_V = []
        # gap(s,a)
        self.gap = []
 
        opt_Qh_plus_1 = np.zeros([self.S, self.A])
        opt_Vh_plus_1 = np.zeros([self.S, 1])
 
        min_gap = self.H
 
        for h in reversed(range(self.H)):
            Rh = self.R[h]
            Ph = self.P[h]
 
            # check dimension
            opt_Q_h = Rh + np.matmul(Ph.reshape([-1, self.S]), opt_Vh_plus_1).reshape([self.S, self.A])
 
            opt_pi_h = np.argmax(opt_Q_h, axis=1)
            # opt_V_h = np.max(opt_Q_h, axis=1)
            opt_V_h = opt_Q_h[np.arange(self.S), opt_pi_h]
            assert len(opt_pi_h) == self.S
 
            self.opt_pi.insert(0, opt_pi_h)
            self.opt_Q.insert(0, opt_Q_h)
            self.opt_V.insert(0, opt_V_h)
 
            opt_Qh_plus_1 = opt_Q_h
            opt_Vh_plus_1 = opt_V_h
 
            # Compute Gap at Step h
            gap_h = opt_V_h.reshape([self.S, 1]) - opt_Q_h
 
            for g in gap_h.reshape([-1]):
                if g > 0:
                    min_gap = min(g, min_gap)
 
            self.gap.insert(0, gap_h)
       
        # uniform initial distribution
        self.opt_value = np.mean(self.opt_V[0])
 
        self.min_gap = min_gap
        print('Min Gap is ', self.min_gap)
 

    def step(self, state, act, h):
        Ph_sa = self.P[h][state][act]
        sh_plus_1 = np.random.choice(self.S, p=Ph_sa)
        return sh_plus_1
 
    def sample_trajectory(self, policy):
        # uniform initial distribution
        sh = np.random.randint(self.S)  
 
        tau = []  
 
        for h in range(self.H):
            ah = policy[h][sh]
            sh_plus_1 = self.step(sh, ah, h)
            rh = self.R[h][sh][ah]
 
            tau.append((sh, ah, rh, sh_plus_1))
 
            sh = sh_plus_1
 
        # print('tau ', tau)
        return tau
 
    def policy_evaluation(self, policy):
        Vh_plus_1 = np.zeros([self.S, 1])
 
        for h in reversed(range(self.H)):
            Rh = self.R[h]
            Ph = self.P[h]
 
            pi_h = policy[h]
 
            Vh = Rh[np.arange(self.S), pi_h].reshape([-1, 1]) + np.matmul(Ph[np.arange(self.S), pi_h, :], Vh_plus_1)
 
            Vh_plus_1 = Vh
 
        V = np.mean(Vh_plus_1)
        return V
 
    def compute_policy_sub_opt_gap(self, policy):
        V_pi = self.policy_evaluation(policy)
        assert self.opt_value - V_pi >= 0.0, self.opt_value - V_pi
        return self.opt_value - V_pi
 
class Algorithm():
    def __init__(self, S, A, H, R, alpha, M=1.0):
        self.S = S
        self.A = A
        self.H = H
 
        self.N_sas = []
        self.N_sa = []
        self.hatP = []
        self.R = R
 
        self.M = M
 
        self.alpha = alpha
 
        for h in range(self.H):
            self.N_sas.append(np.zeros([self.S, self.A, self.S]))
            self.N_sa.append(np.zeros([self.S, self.A]))
            self.hatP.append(np.ones([self.S, self.A, self.S]) / self.S)
 

    def compute_policy(self, k):
        self.delta = 1.0 / k
        # self.delta = 0.1
 
        self.policy = []
 
        Vh_plus_1 = np.zeros([self.S, 1])
        Vh_plus_1_sub = np.zeros([self.S, 1])
        Qh_plus_1 = np.zeros([self.S, self.A])
 
        for h in reversed(range(self.H)):
            bonus = self.compute_bonus(Vh_plus_1, Vh_plus_1_sub, h)
 
            Q_h = self.R[h] + np.matmul(self.hatP[h].reshape([-1, self.S]), Vh_plus_1).reshape([self.S, self.A]) + self.alpha * bonus
 
            # h starts with 0, so we use H - h
            Q_h = np.clip(Q_h, a_min=0.0, a_max=self.H)
 
            pi_h = np.argmax(Q_h, axis=1)
            V_h = np.max(Q_h, axis=1).reshape([-1, 1])
 
            if self.alpha > 0.0 and h == int(self.H / 1.5) and k % 100 == 0:
                print('N_sa_h', self.N_sa[h])
                print('Q_h', Q_h)
                print('bonus ', bonus)
 
            Q_h_sub = self.R[h] + np.matmul(self.hatP[h].reshape([-1, self.S]), Vh_plus_1).reshape([self.S, self.A]) + self.alpha * bonus
            Q_h_sub = np.clip(Q_h_sub, a_min=0.0, a_max=self.H-h)
            V_h_sub = Q_h_sub[np.arange(self.S), pi_h].reshape([-1, 1])
 
            Vh_plus_1_sub = V_h_sub
            Vh_plus_1 = V_h
 
            self.policy.insert(0, pi_h)
 
        return self.policy
 
    def update(self, tau):
        for h in range(self.H):
            sh, ah, rh, sh_plus_1 = tau[h]
            self.N_sas[h][sh][ah][sh_plus_1] += 1.0
            self.N_sa[h][sh][ah] += 1.0
 
            # update hat P
            self.hatP[h][sh][ah] = self.N_sas[h][sh][ah][:] / self.N_sa[h][sh][ah]
 
    # to avoid divide by 0, we divide by max(1e-9, n) or max(1e-9, n-1)
    def compute_bonus(self, Vh_plus_1, Vh_plus_1_sub, h):
        L = np.sqrt(2 * np.log(10 * self.M ** 2 * np.clip(self.N_sa[h], a_min=1.0, a_max=INT_MAX) / self.delta))
 
        N_sa_m1_clip = np.clip(self.N_sa[h] - 1.0, a_min=1e-4, a_max=INT_MAX)
        N_sa_clip = np.clip(self.N_sa[h], a_min=1e-4, a_max=INT_MAX)
 
        br = 8.0 * L / 3.0 / N_sa_m1_clip
        br = np.clip(br, a_min=0.0, a_max=1.0)
 
        P = self.hatP[h].reshape([-1, self.S])
        assert len(Vh_plus_1.shape) == 2
        Var_Vh_plus_1 = np.matmul(P, Vh_plus_1 ** 2) - np.matmul(P, Vh_plus_1) ** 2
        Var_Vh_plus_1 = Var_Vh_plus_1.reshape([self.S, self.A])
 
        Diff_Square = np.matmul(P, np.square(Vh_plus_1 - Vh_plus_1_sub)).reshape([self.S, self.A])
 
        bprob = np.sqrt(2. * Var_Vh_plus_1 * L / N_sa_clip) + 8.0 * self.H * L / N_sa_m1_clip / 3.0 + np.sqrt(2. * L * Diff_Square / N_sa_clip)
        bprob = np.clip(bprob, a_min=0.0, a_max=self.H)
 
        bstr = np.sqrt(Diff_Square) * np.sqrt(self.S * L / N_sa_clip) + 8.0 / 3.0 * self.S * self.H * L / N_sa_clip
        bstr = np.clip(bstr, a_min=0.0, a_max=self.H)
 
        return np.clip(br + bprob + bstr, a_min=0.0, a_max=self.H)

def main():
    args = get_parser()
 
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
   
    alpha = args.alpha
 
    S, A, H = args.S, args.A, args.H
 
    env = Tabular_MDP(S=S, A=A, H=H)
 
    for seed in args.seed:
        random.seed(seed)
        np.random.seed(seed)
 
        random_policy = []
        for h in range(H):
            random_policy.append(np.zeros([S], dtype=np.int32))
 
        tau = env.sample_trajectory(policy=random_policy)
        AlgO = Algorithm(
                    S=S,
                    A=A,
                    H=H,
                    R=env.R,
                    alpha=alpha,
                )
        AlgP = Algorithm(
                    S=S,
                    A=A,
                    H=H,
                    R=env.R,
                    alpha=-alpha,
                )
 
        R_algO = 0.0
        R_algP = 0.0
 
        log_path = 'log/S{}_A{}_H{}_minGap{}_model_seed{}_algO{}_algP{}_useTauP{}'.format(args.S, args.A, args.H, env.min_gap, args.model_seed, args.alpha, -args.alpha, args.use_tauP)
       
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = '{}/seed{}'.format(log_path, seed)
 
        logger = Logger(args.S, args.A, args.H,
                    env.P, env.R, env.min_gap,
                    algO_alpha=args.alpha,
                    algP_alpha=-args.alpha,
                    log_path=log_file)
 
        for k in range(2, args.K):
            pi_O = AlgO.compute_policy(k)
            pi_P = AlgP.compute_policy(k)
 
            O_gap = env.compute_policy_sub_opt_gap(pi_O)
            P_gap = env.compute_policy_sub_opt_gap(pi_P)
 
            tau = env.sample_trajectory(policy=pi_O)
 
            AlgO.update(tau)
            AlgP.update(tau)
 
            if args.use_tauP:
                tauP = env.sample_trajectory(policy=pi_P)
 
                AlgO.update(tauP)
                AlgP.update(tauP)
 
            R_algO += O_gap
            R_algP += P_gap
           
            if k % 100 == 0:
                print('Iter = ', k)
                print('R_algO: ', R_algO, R_algO / np.log(k), R_algO / np.sqrt(k))
                print('R_algP: ', R_algP, R_algP / np.log(k))
                print('Min Gap ', env.min_gap)
                # print('P_h', env.P[h])
                print(log_path)
                print('\n\n')
 
                logger.update_info(k, R_algO, R_algP)
                logger.dump()
 
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-K', type = int, default = 50000, help='training iteration')
    parser.add_argument('--alpha', type = float, default = 1.0, help='coefficient of bonus term')
    parser.add_argument('-S', type = int, default = 5, help='number of states')
    parser.add_argument('-A', type = int, default = 5, help='number of actions')
    parser.add_argument('-H', type = int, default = 5, help='H')
    parser.add_argument('--seed', type = int, default = [0], nargs='+', help='seed')
    parser.add_argument('--model-seed', type = int, default = 10, help='seed')
    parser.add_argument('--use-tauP', default = False, action='store_true', help='whether use tauP')
 
    args = parser.parse_args()
 
    return args
 
if __name__ == '__main__':
    main()