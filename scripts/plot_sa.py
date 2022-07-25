import csv
import json
import argparse
import matplotlib.pyplot as plt

def plot(path):
    path='./results/'+path
    with open(path+'/offline_progress.csv') as f, open(path+'/variant.json') as v:
        log=csv.DictReader(f)
        var=json.load(v)
        avg_returns=[]
        epoch=0 
        for row in log:
            avg_returns.append(float(row['evaluation/Average Returns']))
            #epoch=int(float(row["Epoch"]))
            epoch += 1
    epochs=range(epoch)
    plt.plot(epochs,avg_returns)
    alg, env=var['algorithm'], var['env_name']
    plt.title(alg+' at '+env)
    plt.xlabel("epoch")
    plt.ylabel("avg return")    
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
