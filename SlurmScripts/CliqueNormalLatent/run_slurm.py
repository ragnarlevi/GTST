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

# Graph hopper
parser.add_argument('-mu', '--mu', type=float,metavar='', help='parameter of gaussian')

# Shared parameters
parser.add_argument('-nitr', '--NumberIterations', type=int,metavar='', help='WL nr iterations, wl, wloa, wwl, dk, hashkernel ')
parser.add_argument('-wlab', '--wlab', type=int,metavar='', help='With labels?, sp, rw, pyramid')
parser.add_argument('-type', '--type', type=str,metavar='', help='Type of... rw (geometric or exponential) , deepkernel (sp or wl), hashkenrel(sp or wl), graph hopper (gh) ( ‘linear’, ‘gaussian’, ‘bridge’)')
parser.add_argument('-l', '--discount', type=float,metavar='', help='RW, wwl lambda/discount')
parser.add_argument('-tmax', '--tmax', type=int,metavar='', help='Maximum number of walks, used in propagation and RW.')
parser.add_argument('-w', '--binwidth', type=float,metavar='', help='Bin width.')

# Hash graph
parser.add_argument('-iterations', '--iterations', type=int,metavar='', help='hash kernel iteration')
parser.add_argument('-basekernel', '--basekernel', type=str,metavar='', help='Base kernel WL_kernel or shortest_path_kernel')
parser.add_argument('-scale', '--scale', type=int,metavar='', help='Scale attrubutes?')


# Propagation only

parser.add_argument('-M', '--Distance', type=str,metavar='', help='The preserved distance metric (on local sensitive hashing):')

# Gik
parser.add_argument('-distances', '--distances', type=int,metavar='', help='node neigbourhood depth')

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


ksp['w'] = args.binwidth

# graph hopper
ksp['mu'] = args.mu

# gik
ksp['distances'] = args.distances

# Hash
ksp['iterations'] = args.iterations
ksp['scale'] = args.scale
ksp['basekernel'] = args.basekernel

# Propagation
ksp['tmax'] = args.tmax
ksp['M'] = args.Distance

ksp['type'] = args.type  
ksp['discount'] = args.discount   



experiment_name = 'CliqueNormalLatent'


if kernel_name == 'gh':
    k_val = 'GH'
    if ksp["type"] == 'gaussian':
        unique_identifier = f'ktype_{ksp["type"]}_mu_{ksp["mu"]}'
        script_args = f'-kernel {kernel_name} -type {ksp["type"]} -mu {ksp["mu"]}'
    else:
        unique_identifier = f'ktype_{ksp["type"]}_mu_NA'
        script_args = f'-kernel {kernel_name} -type {ksp["type"]}'
elif kernel_name == 'hash':
    k_val = 'HASH'
    unique_identifier = f'its_{ksp["iterations"]}_scale_{ksp["scale"]}_w_{ksp["w"]}_nitr_{ksp["nitr"]}_bkernel_{ksp["basekernel"]}'
    script_args = f'-kernel {kernel_name} -iterations {ksp["iterations"]} -w {ksp["w"]} -scale {ksp["scale"]} -nitr {ksp["nitr"]} -basekernel {ksp["basekernel"]}'
elif kernel_name == 'gik':
    k_val = 'GIK'
    unique_identifier = f'dist_{ksp["distances"]}__nitr_{ksp["nitr"]}_l_{ksp["discount"]}'
    script_args = f'-kernel {kernel_name} -distances {ksp["distances"]} -discount {ksp["discount"]} -nitr {ksp["nitr"]}'

# elif kernel_name == 'wl':
#     k_val = 'WLsubtree'
#     unique_identifier = f'nitr_{ksp["nitr"]}'
#     script_args = f'-kernel {kernel_name} -nitr {ksp["nitr"]}'

# elif kernel_name == 'sp':
#     k_val = 'SP'
#     unique_identifier = f'wlab_{int(ksp["wlab"])}'
#     script_args = f'-kernel {kernel_name} -wlab {int(ksp["wlab"])}'
# elif kernel_name == 'pyramid':
#     k_val = 'PYRAMID'
#     unique_identifier = f'wlab{int(ksp["wlab"])}_L_{ksp["L"]}_dim_{ksp["dim"]}'
#     script_args = f'-kernel {kernel_name} -wlab {int(ksp["wlab"])} -L {ksp["L"]} -dim {ksp["dim"]}'
elif kernel_name == 'prop':
    k_val = 'PROP'
    unique_identifier = f'w{ksp["w"]}_tmax{ksp["tmax"]}_M_{ksp["M"]}'
    script_args = f'-kernel {kernel_name} -w {ksp["w"]} -tmax {ksp["tmax"]} -M {ksp["M"]}'
# elif kernel_name == 'wloa':
#     k_val = 'WLOA'
#     unique_identifier = f'nitr_{ksp["nitr"]}_'
#     script_args = f'-kernel {kernel_name} -nitr {ksp["nitr"]}'
# elif kernel_name == 'vh':
#     # vertex histogram
#     k_val = 'VH'
#     unique_identifier = f'_'
#     script_args = f'-kernel {kernel_name}'
# elif kernel_name == 'rw':
#     k_val = 'RW'
#     if ksp['tmax'] is None:
#         unique_identifier = f'rw_{ksp["type"]}_l_{ksp["discount"]}_wlab_{int(ksp["wlab"])}'
#         script_args = f'-kernel {kernel_name}  -type {ksp["type"]} -l {ksp["discount"]} -wlab {int(ksp["wlab"])}'
#     else:
#         unique_identifier = f'rw_{ksp["type"]}_l_{ksp["discount"]}_wlab_{int(ksp["wlab"])}_tmax_{ksp["tmax"]}'
#         script_args = f'-kernel {kernel_name}  -type {ksp["type"]} -l {ksp["discount"]} -wlab {int(ksp["wlab"])} -tmax {ksp["tmax"]}'
# elif kernel_name == 'odd':
#     k_val = 'ODD'
#     if ksp['dagh'] is None:
#         unique_identifier = f'dagh_{ksp["dagh"]}'
#         script_args = f'-kernel {kernel_name}'
#     else:
#         unique_identifier = f'dagh_{ksp["dagh"]}'
#         script_args = f'-kernel {kernel_name} -dagh {ksp["dagh"]} '
# elif kernel_name == 'dk':
#     k_val = 'DK'
#     unique_identifier = f'wl_{ksp["nitr"]}_t_{ksp["type"]}'
#     script_args = f'-kernel {kernel_name} -nitr {ksp["nitr"]} -type {ksp["type"]}'
# elif kernel_name == 'wwl':
#     k_val = 'WWL'
#     unique_identifier = f'wl_{ksp["nitr"]}_l_{ksp["discount"]}_sink_{ksp["sinkhorn"]}'
#     script_args = f'-kernel {kernel_name} -nitr {ksp["nitr"]} -l {ksp["discount"]} -sk {ksp["sinkhorn"]}'
else:
    raise ValueError(f'No kernel named {kernel_name}')


from pathlib import Path
# Create directories if they do not exist
Path(f"data/CliqueNormalLatent/{k_val}").mkdir(parents=True, exist_ok=True)
Path(f"/home/{usr}/projects/MMDGraph/SlurmBatch/{experiment_name}/{k_val}").mkdir(parents=True, exist_ok=True)


path = f"/home/{usr}/projects/MMDGraph/SlurmBatch/{experiment_name}"


# the parameters and for loops should be changed for a custom experiment
# nr_nodes = [40, 60, 80]
nr_node_1 = 15
nr_node_2_offsets = [5, 10, 15, 20]
nr_samples = [20, 60]
lat_1 = 0
lat_offsets = [0.1, 0.2, 0.3]


for nr_node_2_offset in nr_node_2_offsets:
    for nr_sample in nr_samples:
        for lat_off in lat_offsets:
            lat_2 = lat_1 + lat_off

            nr_node_2 = nr_node_1 + nr_node_2_offset

                    
            # Note that in the slurm batch file we set another working directory which is the reason for this data_name path
            data_name = f'data/CliqueNormalLatent/{k_val}/v1_{nr_node_1}_v2_{nr_node_2}_n_{nr_sample}_lat1_{lat_1}_lat2_{lat_2}_norm_{norm}_{unique_identifier}.pkl'
            
            job_file = path + f"/{k_val}/v1_{nr_node_1}_v2_{nr_node_2}_n_{nr_sample}_lat1_{lat_1}_lat2_{lat_2}_norm_{norm}_{unique_identifier}.slurm"

            items = ["#!/bin/bash", 
            f"#SBATCH --time=15:00:00",
            f"#SBATCH --job-name={k_val}_normlat_v1_{nr_node_1}_v2_{nr_node_2}_n_{nr_sample}_lat1_{lat_1}_lat2_{lat_2}_norm_{norm}_{unique_identifier}",
            f"#SBATCH --partition=amd-longq",
            f"#SBATCH --nodes=1",
            f"#SBATCH --ntasks-per-node=1",
            f"#SBATCH --cpus-per-task={cpu_per_task}",
            f"#SBATCH --output=/home/{usr}/projects/MMDGraph/outputs/{k_val}_normlat_v1_{nr_node_1}_v2_{nr_node_2}_n_{nr_sample}_lat1_{lat_1}_lat2_{lat_2}_norm_{norm}_{unique_identifier}.out",
            f"#SBATCH --error=/home/{usr}/projects/MMDGraph/errors/{k_val}_normlat_v1_{nr_node_1}_v2_{nr_node_2}_n_{nr_sample}_lat1_{lat_1}_lat2_{lat_2}_norm_{norm}_{unique_identifier}.err",
            f"#SBATCH --mail-user={email}",
            f"#SBATCH --mail-type=FAIL",
            "module purge",
            f"RUNPATH=/home/{usr}/projects/MMDGraph",
            "cd $RUNPATH",
            "source .venv/bin/activate"
            ]
            

            items.append(f"python3 Experiments/{experiment_name}/run.py -B {B} -N {N} -p {data_name} -norm {norm} -d {cpu_per_task} -n1 {nr_sample} -n2 {nr_sample} -nnode1 {nr_node_1} -nnode2 {nr_node_2} -latent1 {lat_1} -latent2 {lat_2} {script_args}")



            with open(job_file, 'w') as fh:
                fh.writelines(s + '\n' for s in items)


            os.system("sbatch %s" %job_file)