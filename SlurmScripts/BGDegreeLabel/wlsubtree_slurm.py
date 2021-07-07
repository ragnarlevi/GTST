import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-tt', '--testtype',metavar='', type=str, help='Type of graph generation')
parser.add_argument('-c', '--CpuPerTask',metavar='', type=int, help='cpu per task', const=4, nargs = "?")
parser.add_argument('-i', '--wlitr',metavar='', type=int, help='Nr WL iterations')
parser.add_argument('-norm', '--normalize',metavar='', type=int, help='Normalize kernel?')

args = parser.parse_args()

tt = args.testtype
cpu_per_task = args.CpuPerTask
wl_it = args.wlitr
norm = args.normalize


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    

if tt.lower() == "bgdegreelabel":
    path = "/home/rgudmundarson/projects/MMDGraph/SlurmBatch/BGDegreeLabel"
    mkdir_p(path)
elif tt.lower() == "bgsamelabel":
    path = "/home/rgudmundarson/projects/MMDGraph/SlurmBatch/BGSameLabel"
    mkdir_p(path)
else:
    assert False, f'{tt} not implemented'



nr_nodes = [40, 60, 80]
nr_samples = [20, 60, 100]
k = 4
degree_offsets = [0.25, 0.5, 0.75, 1]


for nr_node in nr_nodes:
    for nr_sample in nr_samples:
        for k_off in degree_offsets:
                 
            # Note that in the slurm batch file we set another working directory which is the reason for this data_name path
            if tt.lower() == "bgdegreelabel":
                data_name = f'data/BGDegreeLabel/WLsubtree/wl_v_{nr_node}_n_{nr_sample}_k_{k_off}_wl_{wl_it}_norm_{norm}.pkl'
            elif tt.lower() == "bgsamelabel":
                data_name = f'data/BGSameLabel/WLsubtree/wl_v_{nr_node}_n_{nr_sample}_k_{k_off}_wl_{wl_it}_norm_{norm}.pkl'
            
            job_file = path + f"/WLsubtree/v_{nr_node}_n_{nr_sample}_k_{k_off}_wl_{wl_it}_norm_{norm}.slurm"

            items = ["#!/bin/bash", 
            f"#SBATCH --time=3:00:00",
            f"#SBATCH --job-name=wl_mmd_{tt}",
            f"#SBATCH --partition=amd-longq",
            f"#SBATCH --nodes=1",
            f"#SBATCH --ntasks-per-node=1",
            f"#SBATCH --cpus-per-task={cpu_per_task}",
            f"#SBATCH --output=/home/rgudmundarson/projects/MMDGraph/outputs/name=mmd_experiment_v_{nr_node}_n_{nr_sample}_k_{k_off}_wl_{wl_it}_norm_{norm}.out",
            f"#SBATCH --error=/home/rgudmundarson/projects/MMDGraph/errors/name=mmd_experiment_v_{nr_node}_n_{nr_sample}_k_{k_off}_wl_{wl_it}_norm_{norm}.err",
            f"#SBATCH --mail-user=rlg2000@hw.ac.uk",
            f"#SBATCH --mail-type=ALL",
            "module purge",
            "RUNPATH=/home/rgudmundarson/projects/MMDGraph",
            "cd $RUNPATH",
            "source .venv/bin/activate"
            ]
            
            if tt.lower() == "bgdegreelabel":
                items.append(f"python3 Experiments/BGDegreeLabel/wl_subtree.py -B 1000 -N 1000 -p {data_name} -s 1 -norm {norm} -nitr {wl_it} -n1 {nr_sample} -n2 {nr_sample} -nnode1 {nr_node} -nnode2 {nr_node} -k1 {k} -k2 {k + k_off} -d {cpu_per_task}")
            elif tt.lower() == "bgsamelabel":
                items.append(f"python3 Experiments/BGSameLabel/wl_subtree.py -B 1000 -N 1000 -p {data_name} -s 1 -norm {norm} -nitr {wl_it} -n1 {nr_sample} -n2 {nr_sample} -nnode1 {nr_node} -nnode2 {nr_node} -k1 {k} -k2 {k + k_off} -d {cpu_per_task}")



            with open(job_file, 'w') as fh:
                fh.writelines(s + '\n' for s in items)


            os.system("sbatch %s" %job_file)