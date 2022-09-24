import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import argparse

def main():
    args = get_parser()
    log_dirs = args.log_dirs

    parent_path = os.getcwd()
    fig, axs = plt.subplots(1, 3, figsize=(30,8), sharey=True)
    index = 0
    gap_mins = [0.0015, 0.003, 0.009]
    for log_dir in log_dirs:
        os.chdir(log_dir)
        all_R_algO = []
        all_R_algP = []
        ax = axs[index]

        for d in os.listdir():
            print(d)
            with open(d, 'rb') as f:
                data = pickle.load(f)
        
            info = 'S{}-A{}-H{}-GapMin{}'.format(data['S'], data['A'], data['H'], np.around(data['gap_min'], 3))
        
            iters = []
            R_algO = []
            R_algP = []
        
            for item in data['results']:
                k, rO, rP = item
                iters.append(k)
                R_algO.append(rO)
                R_algP.append(rP)
        
            all_R_algO.append(R_algO)
            all_R_algP.append(R_algP)
            print(len(R_algO))
        
        all_R_algO = np.array(all_R_algO)
        all_R_algP = np.array(all_R_algP)
        
        R_algO_mean = np.mean(all_R_algO, axis=0)
        R_algP_mean = np.mean(all_R_algP, axis=0)
        R_algO_std = np.std(all_R_algO, axis=0)
        R_algP_std = np.std(all_R_algP, axis=0)
        
        ax.plot(iters, R_algO_mean, color='r', label='Regret of AlgO')
        ax.plot(iters, R_algP_mean, color='b', label='Regret of AlgP')
        
        ax.fill_between(iters, y1=R_algO_mean-2 * R_algO_std, y2=R_algO_mean+2 * R_algO_std, color='r', alpha=0.25)
        ax.fill_between(iters, y1=R_algP_mean-2 * R_algP_std, y2=R_algP_mean+2 * R_algP_std, color='b', alpha=0.25)
        
        

        ax.set_ylim([0, 15000])
        ax.set_xlim([0, 55000])
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        
        new_x_labels = ['0', '2e4', '4e4']
        new_y_labels = ['', '2.5e3', '5e3', '7.5e3', '1e4', '1.25e4', '1.5e4']
        ax.set_xticklabels(new_x_labels)
        ax.set_yticklabels(new_y_labels)
        ax.set_title(r'$\Delta_{\min}$' + '={}'.format(gap_mins[index]), fontsize=30)

        os.chdir(parent_path)

        if index == len(log_dirs) - 1:
            ax.legend(fontsize=30)
        if index == 0:
            ax.set_ylabel('Regret', fontsize=30)
        if index == 1:
            ax.set_xlabel('Iteration', fontsize=30)

        index += 1

    plt.savefig('Results.png', bbox_inches='tight')



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dirs', type = str, nargs='+', help='dirs of logs to plot')
    args = parser.parse_args()
 
    return args
 

if __name__ == '__main__':
    main()