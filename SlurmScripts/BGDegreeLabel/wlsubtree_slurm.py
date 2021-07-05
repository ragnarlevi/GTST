#!~/projects/MMDGraph/.venv/bin/python3

import os

parser = argparse.ArgumentParser()
parser.add_argument('-tt', '--testtype',metavar='', type=str, help='Type of graph generation')
parser.add_argument('-c', '--CpuPerTask',metavar='', type=int, help='cpu per task', const=4)

args = parser.parse_args()

tt = args.testtype
cpu_per_task = args.CpuPerTask


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    

if tt.tolower() == "bgdegreelabel":
    path = "~/projects/MMDGraph/SlurmBatch/BGDegreeLabel"
    mkdir_p(path)
else:
    assert False, f'{tt} not implemented'


# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)

lizards=["LizardA","LizardB"]

nr_nodes = [40, 60, 80]
nr_samples = [20, 30, 60, 100]
k = 4
degree_offsets = [0.25, 0.5, 0.75, 1, 2]
wl_iterations = [2,3,4,5,6,7,8]


for nr_node in nr_nodes:
    for nr_sample in nr_samples:
        for k_off in degree_offsets:
            for wl_it in wl_iterations:
                
                # Note that in the slurm batch file we set another working directory which is the reason for this data_name path
                if tt.tolower() == "bgdegreelabel":
                    data_name = f'data/BGDegreeLabel/wl_v_{nr_node}_n_{nr_sample}_k_{k_off}_{wl_it}.pkl'
                
                job_file = path + "/wl_subtree.slurm"

                with open(job_file) as fh:
                    fh.writelines("#!/bin/bash")
                    fh.writelines(f"#SBATCH --time = 1:00:00")
                    fh.writelines(f"#SBATCH --job-name=mmd_{tt}")
                    fh.writelines(f"#SBATCH --partition=amd-shortq")
                    fh.writelines(f"#SBATCH --nodes=1")
                    fh.writelines(f"#SBATCH --ntasks-per-node=1")
                    fh.writelines(f"#SBATCH --cpus-per-task={cpu_per_task}")
                    fh.writelines(f"#SBATCH --output=/home/rgudmundarson/projects/MMDGraph/outputs/name=mmd_experiment-%j.out")
                    fh.writelines(f"#SBATCH --error=/home/rgudmundarson/projects/MMDGraph/errors/name=mmd_experiment-%j.err")
                    fh.writelines(f"#SBATCH --mail-user=rlg2000@hw.ac.uk")
                    fh.writelines(f"#SBATCH --mail-type=ALL")

                    fh.writelines("module purge")
                    fh.writelines("RUNPATH=/home/rgudmundarson/projects/MMDGraph")
                    fh.writelines("cd $RUNPATH")
                    fh.writelines("source .venv/bin/activate)")

                    if tt.tolower() == "bgdegreelabel":
                        fh.writelines(f"python 3 MMDGraph/Experiments/BGSameLabel/wl_subtree.py -B 2000 -N 2000 -p {data_name} -s 1 -norm 1 -niter {wl_it} -n1 {nr_sample} -n2 {nr_sample} -nnode1 {nr_node} -nnode2 {nr_node} -k1 {k} -k2 {k + k_off} -d {cpu_per_task}")



                os.system("sbatch %s" %job_file)