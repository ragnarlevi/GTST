import os
from datetime import datetime

p1 = 0.6
p2 = 0.55


# wloa
print("wloa")
for L in [1,2,4,6, 8]:
    now = datetime.now()
    print(now)
    print(f'{L}')
    os.system(f"python Experiments/SignedEdges2/run.py -p data/SignedEdges2/WLOA/n1_60_n2_60_nnode1_50_nnode2_50_nitr_{L}_norm_0_p1_{p1}_p2_{p2}.pkl -B 10000 -N 4000 -kernel wloa -n1 60 -n2 60 -nnode1 50 -nnode2 50 -p1 {p1} -p2 {p2} -d 4 -norm 0 -nitr {L}")
    print(datetime.now() - now )

# # vertex Histogram
now = datetime.now()
print("vh")
print(now)
os.system(f"python Experiments/SignedEdges2/run.py -p data/SignedEdges2/VH/n1_60_n2_60_nnode1_50_nnode2_50_norm_0_p1_{p1}_p2_{p2}.pkl -B 10000 -N 4000 -kernel vh  -n1 60 -n2 60 -nnode1 50 -nnode2 50 -p1 {p1} -p2 {p2} -d 4 -norm 0 ")
print(datetime.now() - now )


# edge Histogram
now = datetime.now()
print("edge")
print(now)
os.system(f"python Experiments/SignedEdges2/run.py -p data/SignedEdges2/EH/n1_60_n2_60_nnode1_50_nnode2_50_norm_0_p1_{p1}_p2_{p2}.pkl -B 10000 -N 4000 -kernel eh  -n1 60 -n2 60 -nnode1 50 -nnode2 50 -p1 {p1} -p2 {p2} -d 4 -norm 0 ")
print(datetime.now() - now )


# RW
print("RW")
for r in [2, 4, 6, 8]:#[2,4,6,8, 10, 15, 20]:
    for c in[0.001, 0.01, 0.1]:
        for type in ['ARKU_edge']:# , 'p-rw']:#["ARKU_plus", "ARKL"]:
            #for tmax in [6, 8, 20]:
            now = datetime.now()
            print(f'{r} {c} {type}')
            string = f"python Experiments/SignedEdges2/run.py -p data/SignedEdges2/RW/n1_60_n2_60_nnode1_50_nnode2_50_c_{c}_tmax_{0}_r_{r}_type_{type}_norm_0_adjnorm_0_rownorm_0_p1_{p1}_p2_{p2}.pkl -B 10000 -N 4000 -kernel rw -n1 60 -n2 60 -nnode1 50 -nnode2 50 -p1 {p1} -p2 {p2} -d 4 -norm 0 -rwApprox {r} -l {c} -type {type} -adj_norm {0} -row_norm {0}"
            #print(string)
            os.system(string )
            print(datetime.now() - now )