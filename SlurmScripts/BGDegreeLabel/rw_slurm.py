import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-tt', '--testtype',metavar='', type=str, help='Type of graph generation')
parser.add_argument('-c', '--CpuPerTask',metavar='', type=int, help='cpu per task', const=4, nargs = "?")
parser.add_argument('-norm', '--normalize',metavar='', type=int, help='Normalize kernel?')
parser.add_argument('-rw', '--rwtype', type=str,metavar='', help='Defines how rw inner summation will be applied.')
parser.add_argument('-l', '--discount', type=float,metavar='', help='RW lambda')

args = parser.parse_args()

tt = args.testtype
cpu_per_task = args.CpuPerTask
norm = args.normalize
rw_type = args.rwtype  
discount = args.discount 


# def mkdir_p(dir):
#     '''make a directory (dir) if it doesn't exist'''
#     if not os.path.exists(dir):
#         os.mkdir(dir)
    


path = "/home/rgudmundarson/projects/MMDGraph/SlurmBatch/BGDegreeLabel"
#mkdir_p(path)




nr_nodes = [40, 60, 80]
nr_samples = [20, 60, 100]
k = 4
degree_offsets = [0.25, 0.5, 0.75, 1]


for nr_node in nr_nodes:
    for nr_sample in nr_samples:
        for k_off in degree_offsets:
                 
            # Note that in the slurm batch file we set another working directory which is the reason for this data_name path
            if tt.lower() == "bgdegreelabel":
                data_name = f'data/BGDegreeLabel/RW/rw_v_{nr_node}_n_{nr_sample}_k_{k_off}_norm_{norm}.pkl'
            
            job_file = path + f"/RW/v_{nr_node}_n_{nr_sample}_k_{k_off}_norm_{norm}.slurm"

            items = ["#!/bin/bash", 
            f"#SBATCH --time=3:00:00",
            f"#SBATCH --job-name=rw_mmd_{tt}_{nr_node}_n_{nr_sample}_k_{k_off}_norm_{norm}",
            f"#SBATCH --partition=amd-longq",
            f"#SBATCH --nodes=1",
            f"#SBATCH --ntasks-per-node=1",
            f"#SBATCH --cpus-per-task={cpu_per_task}",
            f"#SBATCH --output=/home/rgudmundarson/projects/MMDGraph/outputs/rw_v_{nr_node}_n_{nr_sample}_k_{k_off}_norm_{norm}.out",
            f"#SBATCH --error=/home/rgudmundarson/projects/MMDGraph/errors/rw_v_{nr_node}_n_{nr_sample}_k_{k_off}_norm_{norm}.err",
            f"#SBATCH --mail-user=rlg2000@hw.ac.uk",
            f"#SBATCH --mail-type=ALL",
            "module purge",
            "RUNPATH=/home/rgudmundarson/projects/MMDGraph",
            "cd $RUNPATH",
            "source .venv/bin/activate"
            ]
            
            if tt.lower() == "bgdegreelabel":
                items.append(f"python3 Experiments/BGDegreeLabel/rw.py -B 1000 -N 1000 -p {data_name} -s 1 -norm {norm} -rw {rw_type} -l {discount} -n1 {nr_sample} -n2 {nr_sample} -nnode1 {nr_node} -nnode2 {nr_node} -k1 {k} -k2 {k + k_off} -d {cpu_per_task}")



            with open(job_file, 'w') as fh:
                fh.writelines(s + '\n' for s in items)


            os.system("sbatch %s" %job_file)