import os
from datetime import datetime


now = datetime.now()
print(now)

noise1 = 0.1
noise2 = 0.14

# odd

for d in [1,2,3,None]:
    if d is None:
        os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p data/SBMOnlyRandomLabel/ODD/n1_60_n2_60_noise1_{noise1}_noise2_{noise2}_dagh_{d}.pkl -B 3000 -N 3000 -kernel odd -n1 60 -n2 60 -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 ")
    else:
        os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p data/SBMOnlyRandomLabel/ODD/n1_60_n2_60_noise1_{noise1}_noise2_{noise2}_dagh_{d}.pkl -B 3000 -N 3000 -kernel odd -n1 60 -n2 60 -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -dagh {d}")



# # wl
# # for nitr in [1,2,4,6, 8]:
# #     print(f'{nitr}')
# #     os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/WLsubtree/n1_60_n2_60_nirtr_{nitr}_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel wl -n1 60 -n2 60 -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -nitr {nitr}")

# # wloa
# # for nitr in [1, 2,4,6,8]:
# #     print(f'{nitr}')
# #     os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/WLOA/n1_60_n2_60_nirtr_{nitr}_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel wloa -n1 60 -n2 60 -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -nitr {nitr}")


# # SP
# # os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/SP/n1_60_n2_60_norm_0_k_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel sp -n1 60 -n2 60 -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -wlab 1")

# # VH
# # os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/VH/n1_60_n2_60_norm_0_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel vh -n1 60 -n2 60 -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 ")


# # WWL
# # for l in [0.1]:
# #     for nitr in [2, 4, 6, 8]:
# #         print(f'{l} {nitr}')
# #         os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/WWL/n1_60_n2_60_nitr_{nitr}_l_{l}_norm_0_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel wwl -n1 60 -n2 60 -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -nitr {nitr} -l {l}")

# # # PROP
# # for tmax in [2,3,4,5]:
# #     for w in [0.01, 0.001, 0.0001]:
# #         print(f'{tmax} {w}')
# #         os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/PROP/n1_60_n2_60_tmax_{tmax}_M_TV_w_{w}_norm_0_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel prop -n1 60 -n2 60 -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -tmax {tmax} -w {w} -M H")


# # Pyramid
# for L in [10, 12]:
#     for d in [2, 3, 4, 6]:
#         if L == 10 and d == 2:
#             continue
#         print(f'{L} {d}')
#         os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/PYRAMID/n1_60_n2_60_L_{L}_dim_{d}_norm_0_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel pyramid -n1 60 -n2 60 -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -L {L} -dim {d}")

# # # DK
# print('dk')
# os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/DK/n1_60_n2_60_type_sp_norm_0_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel dk -n1 60 -n2 60 -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -type sp")
# for nitr in [4]:
#     print(f'{nitr}')
#     os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/DK/n1_60_n2_60_type_wl_nitr_{nitr}_norm_0_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel dk -n1 60 -n2 60 -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -type wl -nitr {nitr}")




# # Random Walk
# for r in [4, 6, 8, 10, 15, 20]:
#     for c in[0.01]:
#         for type in ['ARKL']:# , 'p-rw']:#["ARKU_plus", "ARKL"]:
#             #for tmax in [6, 8, 20]:
#             now = datetime.now()
#             print(f'{r} {c} {type}')
#             string = f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/RW/n1_60_n2_60_c_{c}_tmax_{0}_r_{r}_type_{type}_norm_0_noise1_{noise1}_noise2_{noise2}_adjnorm_0_rownorm_0.pkl' -B 3000 -N 1000 -kernel rw -n1 60 -n2 60 -noise1 {noise1} -noise2 {noise2} -d 1 -norm 0 -rwApprox {r} -l {c} -type {type} -adj_norm {0} -row_norm {0}"
#             # print(string)
#             os.system(string )
#             print(datetime.now() - now)



# for n in [20, 80, 100, 150]:
#     print(f'wl {n}')
#     os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/WLsubtree/n1_{n}_n2_{n}_nirtr_{4}_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel wl -n1 {n} -n2 {n} -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -nitr {4}")
#     print(f'wloa {n}')
#     os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/WLOA/n1_{n}_n2_{n}_nirtr_{4}_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel wloa -n1 {n} -n2 {n} -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -nitr {4}")
#     print(f'sp {n}')
#     os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/SP/n1_{n}_n2_{n}_norm_0_k_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel sp -n1 {n} -n2 {n} -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -wlab 1")
#     print(f'wwl {n}')
#     os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/WWL/n1_{n}_n2_{n}_nitr_{4}_l_{0.1}_norm_0_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel wwl -n1 {n} -n2 {n} -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -nitr {4} -l {0.1}")
#     print(f'prop {n}')
#     os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/PROP/n1_{n}_n2_{n}_tmax_{3}_M_TV_w_{0.01}_norm_0_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel prop -n1 {n} -n2 {n} -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -tmax {3} -w {0.01} -M H")
#     print(f'pyramid {n}')
#     os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/PYRAMID/n1_{n}_n2_{n}_L_{4}_dim_{3}_norm_0_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel pyramid -n1 {n} -n2 {n} -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -L {4} -dim {3} -wlab 1")
#     print(f'dk {n}')
#     os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/DK/n1_{n}_n2_{n}_type_sp_norm_0_noise1_{noise1}_noise2_{noise2}.pkl' -B 3000 -N 3000 -kernel dk -n1 {n} -n2 {n} -noise1 {noise1} -noise2 {noise2} -d 4 -norm 0 -type sp")

# for n in [20, 80, 100, 150]:
#     print(f'rw {n}')
#     os.system(f"python Experiments/SBMOnlyRandomLabel/run.py -p '../../data/SBMOnlyRandomLabel/RW/n1_{n}_n2_{n}_c_{0.01}_tmax_{0}_r_{10}_type_{'ARKL'}_norm_0_noise1_{noise1}_noise2_{noise2}_adjnorm_0_rownorm_0.pkl' -B 3000 -N 500 -kernel rw -n1 {n} -n2 {n} -noise1 {noise1} -noise2 {noise2} -d 1 -norm 0 -rwApprox {10} -l {0.01} -type {'ARKL'} -adj_norm {0} -row_norm {0}")

