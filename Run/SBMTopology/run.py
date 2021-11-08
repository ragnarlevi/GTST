

import os

# Prop
# for tmax in [2,3,4,5]:
#     for w in [0.1, 0.01, 0.001, 0.0001]:
#         os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/PROP/n1_20_n2_20_tmax_{tmax}_M_H_w_{w}_norm_0_diff_0.1' -B 1000 -N 1000 -kernel prop -n1 20 -n2 20 -diff 0.1 -d 4 -norm 0 -tmax {tmax} -w {w} -M H")


# # Pyramid
# for L in [2,4,6]:
#     for d in [2, 4, 6]:
#         print(f'{L} {d}')
#         os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/PYRAMID/n1_20_n2_20_L_{L}_dim_{d}_norm_0_diff_0.1' -B 1000 -N 1000 -kernel pyramid -n1 20 -n2 20 -diff 0.1 -d 4 -norm 0 -L {L} -dim {d}")

# wloa
# for L in [2,4,6]:
#     print(f'{L}')
#     os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/WLOA/n1_20_n2_20_nitr_{L}_norm_0_diff_0.1' -B 1000 -N 1000 -kernel wloa -n1 20 -n2 20 -diff 0.1 -d 4 -norm 0 -nitr {L}")


# WWL
for l in [0.001, 0.01, 0.1, 1]:
    for nitr in [2, 4, 6]:
        print(f'{l} {nitr}')
        os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/WWL/n1_20_n2_20_nitr_{nitr}_l_{l}_norm_0_diff_0.1' -B 1000 -N 1000 -kernel wwl -n1 20 -n2 20 -diff 0.1 -d 4 -norm 0 -nitr {nitr} -l {l}")

