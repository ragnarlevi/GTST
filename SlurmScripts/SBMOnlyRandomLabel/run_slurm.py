import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--username',metavar='', type=str, help='username')
parser.add_argument('-e', '--email',metavar='', type=str, help='email')
parser.add_argument('-B', '--NrBootstraps',metavar='', type=int, help='Give number of bootstraps')
parser.add_argument('-N', '--NrSampleIterations',metavar='', type=int, help='Give number of sample iterations')
parser.add_argument('-c', '--CpuPerTask',metavar='', type=int, help='cpu per task', const=4, nargs = "?")

# Kernel specifics
parser.add_argument('-kernel', '--kernel', type=str,metavar='', help='Kernel')
parser.add_argument('-norm', '--normalize', type=int,metavar='', help='Should kernel be normalized')

# Shared parameters
parser.add_argument('-nitr', '--NumberIterations', type=int,metavar='', help='WL nr iterations, wl, wloa, wwl, dk')
parser.add_argument('-wlab', '--wlab', type=int,metavar='', help='With labels?, sp, rw, pyramid')
parser.add_argument('-type', '--type', type=str,metavar='', help='Type of... rw (geometric or exponential) , deepkernel (sp or wl)')
parser.add_argument('-l', '--discount', type=float,metavar='', help='RW, wwl lambda/discount')
parser.add_argument('-tmax', '--tmax', type=int,metavar='', help='Maximum number of walks, used in propagation and RW.')

# pyramid only
parser.add_argument('-L', '--histogramlevel', type=int,metavar='', help='Pyramid histogram level.')
parser.add_argument('-dim', '--dim', type=int,metavar='', help='The dimension of the hypercube.')

# Propagation only
parser.add_argument('-w', '--binwidth', type=float,metavar='', help='Bin width.')
parser.add_argument('-M', '--Distance', type=str,metavar='', help='The preserved distance metric (on local sensitive hashing):')

# ODD only
parser.add_argument('-dagh', '--DAGHeight', type=int,metavar='', help='Maximum (single) dag height. If None there is no restriction.')

# WWL only
parser.add_argument('-sk', '--sinkhorn', type=int,metavar='', help='sinkhorn?')

args = parser.parse_args()

usr = args.username
email = args.email
cpu_per_task = args.CpuPerTask

norm = args.normalize
B = args.NrBootstraps
N = args.NrSampleIterations


kernel_name = args.kernel
# add parameters parsed, may be none
ksp = dict()

# WL iterations
ksp['nitr'] = args.NumberIterations   

# with labels?
if args.wlab is None:
    ksp['wlab'] = bool(1)
else:
    ksp['wlab'] = bool(args.wlab)

ksp['L'] = args.histogramlevel
ksp['dim'] = args.dim

ksp['w'] = args.binwidth
ksp['tmax'] = args.tmax
ksp['M'] = args.Distance

ksp['type'] = args.type  
ksp['discount'] = args.discount   

ksp['dagh'] = args.DAGHeight
ksp['sinkhorn'] = args.sinkhorn


if kernel_name == 'wl':
    k_val = 'WLsubtree'
    unique_identifier = f'nitr_{ksp["nitr"]}'
    script_args = f'-kernel {kernel_name} -nitr {ksp["nitr"]}'

elif kernel_name == 'sp':
    k_val = 'SP'
    unique_identifier = f'wlab_{int(ksp["wlab"])}'
    script_args = f'-kernel {kernel_name} -wlab {int(ksp["wlab"])}'

elif kernel_name == 'pyramid':
    k_val = 'PYRAMID'
    unique_identifier = f'wlab{int(ksp["wlab"])}_L_{ksp["L"]}_dim_{ksp["dim"]}'
    script_args = f'-kernel {kernel_name} -wlab {int(ksp["wlab"])} -L {ksp["L"]} -dim {ksp["dim"]}'

elif kernel_name == 'prop':
    k_val = 'PROP'
    unique_identifier = f'w{ksp["w"]}_tmax{ksp["tmax"]}_M{ksp["M"]}'
    script_args = f'-kernel {kernel_name} -w {ksp["w"]} -tmax {ksp["tmax"]} -M {ksp["M"]}'

elif kernel_name == 'wloa':
    k_val = 'WLOA'
    unique_identifier = f'nitr_{ksp["nitr"]}_'
    script_args = f'-kernel {kernel_name} -nitr {ksp["nitr"]}'

elif kernel_name == 'vh':
    # vertex histogram
    k_val = 'VH'
    unique_identifier = f'_'
    script_args = f'-kernel {kernel_name}'

elif kernel_name == 'rw':
    k_val = 'RW'
    if ksp['tmax'] is None:
        unique_identifier = f'rw_{ksp["type"]}_l_{ksp["discount"]}_wlab_{int(ksp["wlab"])}'
        script_args = f'-kernel {kernel_name}  -type {ksp["type"]} -l {ksp["discount"]} -wlab {int(ksp["wlab"])}'
    else:
        unique_identifier = f'rw_{ksp["type"]}_l_{ksp["discount"]}_wlab_{int(ksp["wlab"])}_tmax_{ksp["tmax"]}'
        script_args = f'-kernel {kernel_name}  -type {ksp["type"]} -l {ksp["discount"]} -wlab {int(ksp["wlab"])} -tmax {ksp["tmax"]}'

elif kernel_name == 'odd':
    k_val = 'ODD'
    if ksp['dagh'] is None:
        unique_identifier = f'dagh_{ksp["dagh"]}'
        script_args = f'-kernel {kernel_name}'
    else:
        unique_identifier = f'dagh_{ksp["dagh"]}'
        script_args = f'-kernel {kernel_name} -dagh {ksp["dagh"]}'

elif kernel_name == 'dk':
    k_val = 'DK'
    unique_identifier = f'wl_{ksp["nitr"]}_t_{ksp["type"]}'
    script_args = f'-kernel {kernel_name} -nitr {ksp["nitr"]} -type {ksp["type"]}'

elif kernel_name == 'wwl':
    k_val = 'WWL'
    unique_identifier = f'wl_{ksp["nitr"]}_l_{ksp["discount"]}_sink_{ksp["sinkhorn"]}'
    script_args = f'-kernel {kernel_name} -nitr {ksp["nitr"]} -l {ksp["discount"]} -sk {ksp["sinkhorn"]}'
    
else:
    raise ValueError(f'No kernel named {kernel_name}')


from pathlib import Path
# Create directories if they do not exist
Path(f"data/SBMOnlyRandomLabel/{k_val}").mkdir(parents=True, exist_ok=True)
Path(f"/home/{usr}/projects/MMDGraph/SlurmBatch/SBMOnlyRandomLabel/{k_val}").mkdir(parents=True, exist_ok=True)


path = f"/home/{usr}/projects/MMDGraph/SlurmBatch/SBMOnlyRandomLabel"


nr_samples = [10, 20, 40, 60, 100, 150]
noises = [0.01, 0.02, 0.04, 0.08, 0.12, 0.2]


for nr_sample in nr_samples:
    for noise in noises:

                    
        # Note that in the slurm batch file we set another working directory which is the reason for this data_name path
        data_name = f'data/SBMOnlyRandomLabel/{k_val}/n_{nr_sample}_noise_{noise}_norm_{norm}_{unique_identifier}.pkl'
        
        job_file = path + f"/{k_val}/n_{nr_sample}_noise_{noise}_norm_{norm}_{unique_identifier}.slurm"

        items = ["#!/bin/bash", 
        f"#SBATCH --time=12:00:00",
        f"#SBATCH --job-name={k_val}_sbmORL_n_{nr_sample}_noise_{noise}_norm_{norm}_{unique_identifier}",
        f"#SBATCH --partition=amd-longq",
        f"#SBATCH --nodes=1",
        f"#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={cpu_per_task}",
        f"#SBATCH --output=/home/{usr}/projects/MMDGraph/outputs/{k_val}_sbmorl_n_{nr_sample}_noise_{noise}_norm_{norm}_{unique_identifier}.out",
        f"#SBATCH --error=/home/{usr}/projects/MMDGraph/errors/{k_val}_sbmorl_n_{nr_sample}_noise_{noise}_norm_{norm}_{unique_identifier}.err",
        f"#SBATCH --mail-user={email}",
        f"#SBATCH --mail-type=FAIL",
        "module purge",
        f"RUNPATH=/home/{usr}/projects/MMDGraph",
        "cd $RUNPATH",
        "source .venv/bin/activate"
        ]
        

        items.append(f"python3 Experiments/SBMOnlyRandomLabel/run.py -B {B} -N {N} -n1 {nr_sample} -n2 {nr_sample} -p {data_name} -norm {norm} -d {cpu_per_task} -noise {noise} {script_args}")



        with open(job_file, 'w') as fh:
            fh.writelines(s + '\n' for s in items)


        os.system("sbatch %s" %job_file)