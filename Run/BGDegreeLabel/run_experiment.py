import os
from datetime import datetime


now = datetime.now()
print(now)



# Random Walk
# for r in [2, 4, 6]: #[2,4,6]:
#     for c in [0.01]: # [0.01, 0.001]:
#         for type in ['p-rw']:# , 'p-rw']:#["ARKU_plus", "ARKL"]:
#             for tmax in [6, 8, 20]:
#                 for k_diff in [0.25]:#[0.1, 0.25, 0.5]:
#                     now = datetime.now()
#                     k1 = 4
#                     k2 = k1 + k_diff
#                     print(f'{r} {c} {type} {tmax} {k_diff}')
#                     string = f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/RW/n1_20_n2_20_v1_60_v2_60_c_{c}_tmax_{tmax}_r_{r}_type_{type}_norm_0_k_{k_diff}_adjnorm_0_rownorm_0' -B 10000 -N 10000 -kernel rw -nnode1 60 -nnode2 60 -k1 {k1} -k2 {k2} -n1 20 -n2 20 -d 4 -norm 0 -rwApprox {r} -l {c} -type {type} -adj_norm {0} -row_norm {0} -tmax {tmax}"
#                    # print(string)
#                     os.system(string )
#                     print(datetime.now() - now )

# for r in [2]: #[2,4,6]:
#     for c in [0.01]: # [0.01, 0.001]:
#         for type in ['ARKU_plus']:# , 'p-rw']:#["ARKU_plus", "ARKL"]:
#             for k_diff in [0.25]:#[0.1, 0.25, 0.5]:
#                 now = datetime.now()
#                 k1 = 4
#                 k2 = k1 + k_diff
#                 print(f'{r} {c} {type} {k_diff}')
#                 string = f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/RW/n1_20_n2_20_v1_60_v2_60_c_{c}_tmax_{tmax}_r_{r}_type_{type}_norm_0_k_{k_diff}_adjnorm_0_rownorm_0' -B 10000 -N 10000 -kernel rw -nnode1 60 -nnode2 60 -k1 {k1} -k2 {k2} -n1 20 -n2 20 -d 4 -norm 0 -rwApprox {r} -l {c} -type {type} -adj_norm {0} -row_norm {0}"
#               #  print(string)
#                 os.system(string )
#                 print(datetime.now() - now )
            
# SP
#print('sp')
#os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/SP/n1_20_n2_20_v1_60_v2_60_norm_0_wlab_{0}_k_{0.25}' -B 10000 -N 10000 -kernel sp -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 20 -n2 20 -d 4 -norm 0 -wlab 0")
#print('sp')
#os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/SP/n1_20_n2_20_v1_60_v2_60_norm_0_wlab_{1}_k_{0.25}' -B 10000 -N 10000 -kernel sp -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 20 -n2 20 -d 4 -norm 0 -wlab 1")

# VH
#print('vh')
#os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/VH/n1_20_n2_20_v1_60_v2_60_norm_0_k_{0.25}' -B 10000 -N 10000 -kernel vh -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 20 -n2 20 -d 4 -norm 0 ")


# WWL
# for l in [0.1]:
#     for nitr in [1, 2,4, 6, 8]:
#         print(f'{l} {nitr}')
#         os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/WWL/n1_20_n2_20_v1_60_v2_60_nitr_{nitr}_l_{l}_norm_0_k_{0.25}' -B 1000 -N 10000 -kernel wwl -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 20 -n2 20 -d 4 -norm 0 -nitr {nitr} -l {l}")

# # PROP
# for tmax in [2,3,4,5]:
#     for w in [0.01, 0.001, 0.0001]:
#         print(f'{tmax} {w}')
#         os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/PROP/n1_20_n2_20_v1_60_v2_60_tmax_{tmax}_M_TV_w_{w}_norm_0_diff_0.1.pkl' -B 1000 -N 10000 -kernel prop -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 20 -n2 20 -d 4 -norm 0 -tmax {tmax} -w {w} -M H")


# # Pyramid
# for L in [8, 10, 12]:
#     for d in [2]:
#         print(f'{L} {d}')
#         os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/PYRAMID/n1_20_n2_20_v1_60_v2_60_L_{L}_dim_{d}_norm_0_diff_0.1' -B 10000 -N 10000 -kernel pyramid -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 20 -n2 20 -d 4 -norm 0 -L {L} -dim {d} -wlab 0")

# # DK
# print('dk')
# #os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/PYRAMID/n1_20_n2_20_v1_60_v2_60_type_sp_norm_0_diff_0.1' -B 10000 -N 10000 -kernel dk -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 20 -n2 20 -d 4 -norm 0 -type sp")
# for nitr in [2, 4, 6]:
#     print(f'{nitr}')
#     os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/PYRAMID/n1_20_n2_20_v1_60_v2_60_type_sp_nitr_{nitr}_norm_0_diff_0.1' -B 10000 -N 10000 -kernel dk -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 20 -n2 20 -d 4 -norm 0 -type wl -nitr {nitr}")




# wloa 
# for nitr in [2,3,4,5,6]:
#     os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/WLOA/n1_20_n2_20_v1_60_v2_60_nirtr_{nitr}_diff_{0.25}' -B 1000 -N 10000 -kernel wloa -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 20 -n2 20 -d 4 -norm 0 -nitr {nitr}")

# wl
# for nitr in [1, 2,3,4,5,6]:
#     print(f'{nitr}')
#     os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/WLsubtree/n1_20_n2_20_v1_60_v2_60_nirtr_{nitr}_diff_{0.25}' -B 1000 -N 10000 -kernel odd -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 20 -n2 20 -d 4 -norm 0 -nitr {nitr}")


# Graph Stats
# os.system(f"python Experiments/BGDegreeLabel/run_graphstat.py -p '../../data/BGDegreeLabel/GRAPHSTATS/n1_20_n2_20_v1_60_v2_60_N_1000.pkl' -B 1000 -N 10000 -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 20 -n2 20 -d 4 ")

# odd
# for n in [150]:
#     for d in [1,2,3,None]:
#         if d is None:
#             os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/odd/n1_{n}_n2_{n}_v1_60_v2_60_diff_{0.25}_dagh_{d}.pkl' -B 3000 -N 3000 -kernel vh -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 4 -norm 0 ")
#         else:
#             os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/odd/n1_{n}_n2_{n}_v1_60_v2_60_diff_{0.25}_dagh_{d}.pkl' -B 3000 -N 3000 -kernel vh -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 4 -norm 0 -dagh {d}")



# gntk
# for n in [20]:
#     for nl in [8]:
#         for mlp in [nl]:
#             for s in ['uniform']:
#                 for jk in [1]:
#                     print(f'{n} {nl} {mlp} {s} {jk}')
#                     os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/GNTK/n1_{n}_n2_{n}_v1_60_v2_60_diff_{0.25}_L_{nl}_mlp_{mlp}_s_{s}_jk_{jk}_norm_0.pkl' -B 10000 -N 10000 -kernel gntk -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 5 -norm 0 -L {nl} -dim {mlp} -sk {jk} -type {s}")


# graphstat
# for n in [20]:
#     for method in ['sp', 'degree']:
#         os.system(f"python Experiments/BGDegreeLabel/run.py -p data/BGDegreeLabel/GRAPHSTAT/n1_{n}_n2_{n}_v1_60_v2_60_diff_{0.25}_meth_{method}_norm_0.pkl -B 10000 -N 10000 -kernel graphstat -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 5 -norm 0 -type {method}")

# graphlet sampling
print("graphlet sampling")
for n in [20]:
    for k in [4,5]:
        for epsilon in [0.1, 0.05]:
            os.system(f"python Experiments/BGDegreeLabel/run.py -p data/BGDegreeLabel/GR/n1_{n}_n2_{n}_v1_60_v2_60_diff_{0.25}_k_{k}_eps_{epsilon}_norm_0.pkl -B 10000 -N 10000 -kernel graphlet_sampling -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 5 -norm 0 -tmax {k} -l {epsilon}")


# Test multiple samples

# for n in [20, 60, ]:
#     print('vh {n}')
#     os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/VH/n1_{n}_n2_{n}_v1_60_v2_60_diff_{0.25}' -B 3000 -N 3000 -kernel vh -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 4 -norm 0 -nitr 4")
#     #print('wl {n}')
#     #os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/WLsubtree/n1_{n}_n2_{n}_v1_60_v2_60_nitr_3_diff_{0.25}' -B 3000 -N 3000 -kernel wl -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 4 -norm 0 -nitr 4")
#     #print('wloa {n}')
#     #os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/WLOA/n1_{n}_n2_{n}_v1_60_v2_60_nitr_3_diff_{0.25}' -B 3000 -N 3000 -kernel wloa -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 4 -norm 0 -nitr 4")
#     #print('sp')
#     #os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/SP/n1_{n}_n2_{n}_v1_60_v2_60_norm_0_k_{0.25}' -B 3000 -N 3000 -kernel sp -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 4 -norm 0 -wlab 0")
#     print('wwl')
#     os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/WWL/n1_{n}_n2_{n}_v1_60_v2_60_nitr_{3}_l_{0.1}_norm_0_k_{0.25}' -B 3000 -N 3000 -kernel wwl -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 4 -norm 0 -nitr {3} -l {0.1}")
#     # print('prop')
#     # os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/PROP/n1_{n}_n2_{n}_v1_60_v2_60_tmax_{2}_M_TV_w_{0.01}_norm_0_diff_0.1.pkl' -B 3000 -N 3000 -kernel prop -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 4 -norm 0 -tmax {2} -w {0.01} -M H")
#     # print('pyramid')
#     # os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/PYRAMID/n1_{n}_n2_{n}_v1_60_v2_60_L_{6}_dim_{2}_norm_0_diff_0.1_wlab_{1}' -B 3000 -N 3000 -kernel pyramid -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 4 -norm 0 -L {6} -dim {2} -wlab 1")
#     # print('dk')
#     # os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/DK/n1_{n}_n2_{n}_v1_60_v2_60_type_sp_norm_0_diff_0.25' -B 3000 -N 3000 -kernel dk -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 4 -norm 0 -type sp")
#     # print('rw')
#     # os.system(f"python Experiments/BGDegreeLabel/run.py -p '../../data/BGDegreeLabel/RW/n1_{n}_n2_{n}_v1_60_v2_60_c_{0.01}_tmax_{0}_r_{2}_type_{'ARKU_plus'}_norm_0_k_{0.25}_adjnorm_0_rownorm_0' -B 3000 -N 3000 -kernel rw -nnode1 60 -nnode2 60 -k1 {4} -k2 {4.25} -n1 {n} -n2 {n} -d 4 -norm 0 -rwApprox {2} -l {0.01} -type ARKU_plus -adj_norm {0} -row_norm {0}")
 





















