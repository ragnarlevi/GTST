import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--username',metavar='', type=str, help='username')
parser.add_argument('-e', '--email',metavar='', type=str, help='email')
parser.add_argument('-B', '--NrBootstraps',metavar='', type=int, help='Give number of bootstraps')
parser.add_argument('-N', '--NrSampleIterations',metavar='', type=int, help='Give number of sample iterations')
parser.add_argument('-c', '--CpuPerTask',metavar='', type=int, help='cpu per task', const=4, nargs = "?")
parser.add_argument('-i', '--wlitr',metavar='', type=int, help='Nr WL iterations')
parser.add_argument('-norm', '--normalize',metavar='', type=int, help='Normalize kernel?')

args = parser.parse_args()

usr = args.username
email = args.email
cpu_per_task = args.CpuPerTask
wl_it = args.wlitr
norm = args.normalize
B = args.NrBootstraps
N = args.NrSampleIterations

path = f"/home/{usr}/projects/MMDGraph/SlurmBatch/SBMOnlyRandomLabel"


nr_samples = [10, 20, 40, 60, 100, 150]
noises = [0.01, 0.02, 0.04, 0.08, 0.12, 0.2]


for nr_sample in nr_samples:
    for noise in noises:

                    
        # Note that in the slurm batch file we set another working directory which is the reason for this data_name path
        data_name = f'data/SBMOnlyRandomLabel/WLsubtree/wl_n_{nr_sample}_noise_{noise}_wl_{wl_it}_norm_{norm}.pkl'
        
        job_file = path + f"/WLsubtree/wl_n_{nr_sample}_noise_{noise}_wl_{wl_it}_norm_{norm}.slurm"

        items = ["#!/bin/bash", 
        f"#SBATCH --time=3:00:00",
        f"#SBATCH --job-name=wl_sbmORL_n_{nr_sample}_noise_{noise}_wl_{wl_it}_norm_{norm}",
        f"#SBATCH --partition=amd-longq",
        f"#SBATCH --nodes=1",
        f"#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={cpu_per_task}",
        f"#SBATCH --output=/home/{usr}/projects/MMDGraph/outputs/sbmorl_n_{nr_sample}_noise_{noise}_k_wl_{wl_it}_norm_{norm}.out",
        f"#SBATCH --error=/home/{usr}/projects/MMDGraph/errors/sbmorl_n_{nr_sample}_noise_{noise}_k_wl_{wl_it}_norm_{norm}.err",
        f"#SBATCH --mail-user={email}",
        f"#SBATCH --mail-type=FAIL",
        "module purge",
        f"RUNPATH=/home/{usr}/projects/MMDGraph",
        "cd $RUNPATH",
        "source .venv/bin/activate"
        ]
        

        items.append(f"python3 Experiments/SBMOnlyRandomLabel/wl_subtree.py -B {B} -N {B} -n1 {nr_sample} -n2 {nr_sample} -p {data_name} -norm {norm} -nitr {wl_it} -d {cpu_per_task} -noise {noise}")



        with open(job_file, 'w') as fh:
            fh.writelines(s + '\n' for s in items)


        os.system("sbatch %s" %job_file)