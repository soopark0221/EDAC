import csv
import json
import argparse
import matplotlib.pyplot as plt

def plot(path):
    path='./results/'+path
    with open(path+'/offline_progress.csv') as f, open(path+'/variant.json') as v:
        log=csv.DictReader(f)
        var=json.load(v)
        num_agents=var['num_agents']
        avg_returns=[]
        epoch=0
        for i in range(num_agents):
            avg_returns.append([])
        for row in log:
            agent=int(row['Agent'])
            if agent==num_agents-1:
                epoch=int(row['Epoch'])
            avg_returns[agent].append(float(row['evaluation/Average Returns']))
    for i in range(len(avg_returns)):
        if len(avg_returns[i])>epoch+1:
            avg_returns[i].pop()
    epochs=range(epoch+1)
    for i, agent_returns in enumerate(avg_returns):
        plt.plot(epochs,agent_returns,label='agent_'+str(i))
    alg, env=var['algorithm'], var['env_name']
    plt.title(alg+' at '+env)
    plt.xlabel("epoch")
    plt.ylabel("avg return")
    plt.legend()    
    plt.savefig(path+'/plot_'+alg+'_'+env+'.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--folder',
                        type=str,
                        required=True,
                        help='algorithm folder name')
    parser.add_argument('--env_seed',
                        default='hopper-medium-v2_0',
                        type=str,
                        help='foramt: env_seed, ex) hopper-medium-v2_0')

    args=parser.parse_args()
    path=args.folder+'/'+args.env_seed
    
    plot(path)
    