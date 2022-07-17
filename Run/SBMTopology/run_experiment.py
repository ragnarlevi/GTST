

import os
from datetime import datetime

# os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/SP/n1_20_n2_20_v1_60_v2_60_norm_0_wlab_{0}_k_{0.25}' -B 3000 -N 3000 -kernel sp -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 20 -n2 20 -d 4 -norm 0 -wlab 0")

diff = 0.08
# Prop
# for tmax in [2,3,4,5]:
#     for w in [0.1, 0.01, 0.001, 0.0001]:
#         os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/PROP/n1_20_n2_20_tmax_{tmax}_M_H_w_{w}_norm_0_diff_{diff}' -B 3000 -N 3000 -kernel prop -n1 20 -n2 20 -diff {diff} -d 4 -norm 0 -tmax {tmax} -w {w} -M H")


# # Pyramid
# for L in [2,4,6, 8, 10, 12]:
#     for d in [2,3, 4, 6]:
#         for wlab in [0, 1]:
#             print(f'{L} {d} {wlab}')
#             os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/PYRAMID/n1_20_n2_20_L_{L}_dim_{d}_wlab_{wlab}_norm_0_diff_{diff}' -B 3000 -N 3000 -kernel pyramid -n1 20 -n2 20 -diff {diff} -d 4 -norm 0 -L {L} -dim {d} -wlab {wlab}")

# for wlab in [0, 1]:
#     os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/SP/n1_20_n2_20_norm_wlab_{wlab}_0_diff_{diff}' -B 3000 -N 3000 -kernel sp -n1 20 -n2 20 -diff {diff} -d 4 -norm 0 -wlab {wlab}")


# # wloa
# for L in [2,4,6]:
#     print(f'{L}')
#     os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/WLOA/n1_20_n2_20_nitr_{L}_norm_0_diff_{diff}' -B 3000 -N 3000 -kernel wloa -n1 20 -n2 20 -diff {diff} -d 4 -norm 0 -nitr {L}")


# # # WWL
# for l in [0.001, 0.01, 0.1, 1]:
#     for nitr in [2, 4, 6]:
#         print(f'{l} {nitr}')
#         os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/WWL/n1_20_n2_20_nitr_{nitr}_l_{l}_norm_0_diff_{diff}' -B 3000 -N 3000 -kernel wwl -n1 20 -n2 20 -diff {diff} -d 4 -norm 0 -nitr {nitr} -l {l}")


# DK
# os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/DK/n1_20_n2_20_type_sp_norm_0_diff_{diff}' -B 3000 -N 3000 -kernel dk -n1 20 -n2 20 -diff {diff} -d 4 -norm 0 -nitr 4 -type sp ")
# for nitr in [2, 4, 6]:
#     print(f'{nitr}')
#     os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/DK/n1_20_n2_20_nitr_{nitr}_type_wl_norm_0_diff_{diff}' -B 3000 -N 3000 -kernel dk -n1 20 -n2 20 -diff {diff} -d 4 -norm 0 -nitr {nitr} -type wl")



# # Random Walk
# for r in [2, 4, 6, 8]:
#     for c in [0.001]:
#         for type in ['p-rw']:#["ARKU_plus"]:#, "ARKL"]:# ['exponential', 'p-rw']:#
#             for tmax in [2, 4, 6, 8, 10, 20]:
#                 now = datetime.now()
#                 print(f'{r} {c} {type} {tmax} ')
#                 string = f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/RW/n1_20_n2_20_c_{c}_tmax_{tmax}_r_{r}_type_{type}_norm_0_k_adjnorm_0_rownorm_0_diff_{diff}' -B 3000 -N 3000 -kernel rw -n1 20 -n2 20 -diff {diff} -d 4 -norm 0 -rwApprox {r} -l {c} -type {type} -adj_norm {0} -row_norm {0} -tmax {tmax}"
#                 print(string)
#                 os.system(string )
#                 print(datetime.now() - now )

# # # Random Walk
# for r in [3, 4, 6]:
#     for c in [0.001]:
#         for type in ["ARKU_plus", 'exponential']:#, "ARKL"]:# ['exponential', 'p-rw']:#
#             #for tmax in [6, 8, 20]:
#             now = datetime.now()
#             print(f'{r} {c} {type} {0} ')
#             string = f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/RW/n1_20_n2_20_c_{c}_tmax_{0}_r_{r}_type_{type}_norm_0_k_adjnorm_0_rownorm_0_diff_{diff}' -B 3000 -N 3000 -kernel rw -n1 20 -n2 20 -diff {diff} -d 4 -norm 0 -rwApprox {r} -l {c} -type {type} -adj_norm {0} -row_norm {0}"
#             print(string)
#             os.system(string )
#             print(datetime.now() - now )

# # wl
# for n in [80, 100, 140, 180, 200]:
#     for L in [4]:
#         now = datetime.now()
#         print(f'{n} {L}')
#         os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/WL/n1_{n}_n2_{n}_nitr_{L}_norm_0_diff_{diff}' -B 3000 -N 1000 -kernel wl -n1 {n} -n2 {n} -diff {diff} -d 4 -norm 0 -nitr {L}")
#         print(datetime.now() - now )

# for r in [10]:
#     for c in[0.01]:
#         for type in ['ARKL']:# , 'p-rw']:#["ARKU_plus", "ARKL"]:
#             # for tmax in [6, 8, 20]:
#             for noise in [0.04]:
#                 now = datetime.now()
#                 print(f'{r} {c} {type} {noise}')
#                 string = f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/RW/n1_60_n2_60_c_{c}_tmax_{0}_r_{r}_type_{type}_norm_0_noise_{noise}_adjnorm_0_rownorm_0.pkl' -B 1000 -N 1000 -kernel rw -n1 60 -n2 60 -noise {noise} -d 1 -norm 0 -rwApprox {r} -l {c} -type {type} -adj_norm {0} -row_norm {0}"
#                 print(string)
#                 os.system(string )
#                 print(datetime.now() - now )
        

# Random Walk degree label
# for r in [2, 4, 6]: #[2,4,6]:
#     for c in [0.01]: # [0.01, 0.001]:
#         for type in ['ARKU_plus', 'exponential']:# ['p-rw']:#['exponential']:# , 'p-rw']:#["ARKU_plus", "ARKL"]:
#             for k_diff in [0.25]:#[0.1, 0.25, 0.5]:
#                 now = datetime.now()
#                 k1 = 4
#                 k2 = k1 + k_diff
#                 print(f'{r} {c} {type} {0} {k_diff}')
#                 string = f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/RW/n1_20_n2_20_v1_60_v2_60_c_{c}_tmax_{0}_r_{r}_type_{type}_norm_0_k_{k_diff}_adjnorm_0_rownorm_0' -B 3000 -N 10000 -kernel rw -nnode1 60 -nnode2 60 -k1 {k1} -k2 {k2} -n1 20 -n2 20 -d 4 -norm 0 -rwApprox {r} -l {c} -type {type} -adj_norm {0} -row_norm {0}"
#                 print(string)
#                 os.system(string )
#                 print(datetime.now() - now )

# for r in [2, 4, 6]: #[2,4,6]:
#     for c in [0.01]: # [0.01, 0.001]:
#         for type in ['p-rw']:# ['p-rw']:#['exponential']:# , 'p-rw']:#["ARKU_plus", "ARKL"]:
#             for tmax in [2, 4, 6, 8, 10, 20]:
#                 for k_diff in [0.25]:#[0.1, 0.25, 0.5]:
#                     now = datetime.now()
#                     k1 = 4
#                     k2 = k1 + k_diff
#                     print(f'{r} {c} {type} {tmax} {k_diff}')
#                     string = f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/RW/n1_20_n2_20_v1_60_v2_60_c_{c}_tmax_{tmax}_r_{r}_type_{type}_norm_0_k_{k_diff}_adjnorm_0_rownorm_0' -B 3000 -N 10000 -kernel rw -nnode1 60 -nnode2 60 -k1 {k1} -k2 {k2} -n1 20 -n2 20 -d 4 -norm 0 -rwApprox {r} -l {c} -type {type} -adj_norm {0} -row_norm {0} -tmax {tmax}"
#                     print(string)
#                     os.system(string )
#                     print(datetime.now() - now )


# odd
for d in [1,2,3,None]:
    if d is None:
        os.system(f"python Experiments/SBMTopology/run.py -p data/SBMTopology/odd/n1_20_n2_20_norm_0_diff_{diff}_dagh_{d}.pkl -B 10000 -N 10000 -kernel odd -n1 20 -n2 20 -diff {diff} -d 8 -norm 0 ")
    else:
        os.system(f"python Experiments/SBMTopology/run.py -p data/SBMTopology/odd/n1_20_n2_20_norm_0_diff_{diff}_dagh_{d}.pkl -B 10000 -N 10000 -kernel odd -n1 20 -n2 20 -diff {diff} -d 8 -norm 0 -dagh {d}")


# for n in [100, 150]:
#     print(f'wl {n}')
#     os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/WL/n1_{n}_n2_{n}_diff_{diff}_nitr_{3}' -B 3000 -N 3000 -kernel wl -n1 {n} -n2 {n} -diff {diff} -d 4 -norm 0 -nitr {3}")
#     print(f'wloa {n}')
#     os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/WLOA/n1_{n}_n2_{n}_diff_{diff}_nitr_{3}' -B 3000 -N 3000 -kernel wloa -n1 {n} -n2 {n} -diff {diff} -d 4 -norm 0 -nitr {3}")
#     print(f'sp {n}')
#     os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/SP/n1_{n}_n2_{n}_norm_0_diff_{diff}' -B 3000 -N 3000 -kernel sp -diff {diff} -n1 {n} -n2 {n} -d 4 -norm 0 -wlab 0")
#     print(f'wwl {n}')
#     os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/WWL/n1_{n}_n2_{n}_nitr_{3}_l_{0.1}_norm_0_diff_{diff}' -B 3000 -N 3000 -kernel wwl -diff {diff} -n1 {n} -n2 {n}  -d 4 -norm 0 -nitr {3} -l {0.1}")
#     print(f'prop {n}')
#     os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/PROP/n1_{n}_n2_{n}_tmax_{3}_M_TV_w_{0.01}_norm_0_diff_{diff}.pkl' -B 3000 -N 3000 -kernel prop -diff {diff} -n1 {n} -n2 {n} -d 4 -norm 0 -tmax {3} -w {0.01} -M H")
#     print(f'pyramid {n}')
#     os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/PYRAMID/n1_{n}_n2_{n}_L_{6}_dim_{2}_norm_0_diff_{diff}_wlab_{1}' -B 3000 -N 3000 -kernel pyramid -diff {diff} -n1 {n} -n2 {n} -d 4 -norm 0 -L {6} -dim {2} -wlab 0")
#     print(f'dk {n}')
#     os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/DK/n1_{n}_n2_{n}_type_sp_norm_0_diff_{diff}' -B 3000 -N 3000 -kernel dk -diff {diff} -n1 {n} -n2 {n} -d 4 -norm 0 -type sp")
#     print(f'rw {n}')
#     os.system(f"python Experiments/SBMTopology/run.py -p '../../data/SBMTopology/RW/n1_{n}_n2_{n}_c_{0.001}_tmax_{0}_r_{4}_type_{'ARKU_plus'}_norm_0_adjnorm_0_rownorm_0_diff_{diff}' -B 3000 -N 3000 -kernel rw -diff {diff} -n1 {n} -n2 {n} -d 4 -norm 0 -rwApprox {4} -l {0.01} -type ARKU_plus -adj_norm {0} -row_norm {0}")
