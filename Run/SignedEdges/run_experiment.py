import os
from datetime import datetime

b1 = 0.7
b2 = 0.5

# # edge Histogram
# now = datetime.now()
# print("edge")
# print(now)
# os.system(f"python Experiments/SignedEdges/run.py -p '../../data/SignedEdges/EH/n1_60_n2_60_nnode1_200_nnode2_200_norm_0_b1_09_b2_05' -B 1000 -N 1000 -kernel eh -k 5 -n1 60 -n2 60 -nnode1 200 -nnode2 200 -b1 {b1} -b2 {b2} -d 4 -norm 0 ")
# print(datetime.now() - now )

# # # vertex Histogram
# now = datetime.now()
# print("vh")
# print(now)
# os.system(f"python Experiments/SignedEdges/run.py -p '../../data/SignedEdges/VH/n1_60_n2_60_nnode1_100_nnode2_100_norm_0_b1_09_b2_05' -B 1000 -N 1000 -kernel eh -k 5 -n1 60 -n2 60 -nnode1 200 -nnode2 200 -b1 {b1} -b2 {b2} -d 4 -norm 0 ")
# print(datetime.now() - now )

# # wloa
# print("wloa")
# for L in [2,4,6, 8]:
#     now = datetime.now()
#     print(now)
#     print(f'{L}')
#     os.system(f"python Experiments/SignedEdges/run.py -p '../../data/SignedEdges/WLOA/n1_60_n2_60_nnode1_200_nnode2_200_nitr_{L}_norm_0_b1_09_b2_05' -B 1000 -N 1000 -kernel wloa -k 5 -n1 60 -n2 60 -nnode1 200 -nnode2 200 -b1 {b1} -b2 {b2} -d 4 -norm 0 -nitr {L}")
#     print(datetime.now() - now )

# RW
print("RW")
for r in [4]:#[2,4,6,8, 10, 15, 20]:
    for c in[0.01, 0.1]:
        for type in ['ARKU_edge']:# , 'p-rw']:#["ARKU_plus", "ARKL"]:
            #for tmax in [6, 8, 20]:
            now = datetime.now()
            print(f'{r} {c} {type}')
            string = f"python Experiments/SignedEdges/run.py -p '../../data/SignedEdges/RW/n1_60_n2_60_nnode1_200_nnode2_200_c_{c}_tmax_{0}_r_{r}_type_{type}_norm_0_adjnorm_0_rownorm_0.pkl' -B 1000 -N 1000 -kernel rw -k 5 -n1 60 -n2 60 -nnode1 200 -nnode2 200 -b1 {b1} -b2 {b2} -d 4 -norm 0 -rwApprox {r} -l {c} -type {type} -adj_norm {0} -row_norm {0}"
            #print(string)
            os.system(string )
            print(datetime.now() - now )