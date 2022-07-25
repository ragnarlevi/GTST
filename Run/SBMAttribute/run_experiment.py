import os
from datetime import datetime

noise1 = 0.1
noise2 = 0.14
mean11 = 0
mean21 = 0
mean12 = 1
mean22 = 1
mean13 = 2
mean23 = 2
nr_sample = 60
N = 3000
B = 10000

print(f'N {N}')
print(nr_sample)
print(B)





for r in [4, 6]:
    for c in [0.001]:
        for type in ['p-rw']:#["ARKU_plus"]:#, "ARKL"]:# ['exponential', 'p-rw']:#
            for tmax in [2, 4, 6, 8, 10]:
                now = datetime.now()
                print(now)
                print(f'{r} {c} {type} {tmax} ')
                os.system(f"python Experiments/SBMAttributeNormal/run.py -p data/SBMAttributeNormal/RW/n1_{nr_sample}_n2_{nr_sample}_type_{type}_tmax_{tmax}_r_{r}_c_{c}_norm_0_noise1_{noise1}_noise2_{noise2}_m11_{mean11}_m12_{mean12}_m13_{mean13}_m21_{mean21}_m22_{mean22}_m23_{mean23}_new.pkl -B {B} -N {N} -n1 {nr_sample} -n2 {nr_sample}  -mean11 {mean11} -mean12 {mean12} -mean13 {mean13} -mean21 {mean21} -mean22 {mean22} -mean23 {mean23} -noise1 {noise1} -noise2 {noise2} -norm 0 -d 4 -kernel rw -rwApprox {r} -l {c} -type {type} -tmax {tmax} -adj_norm {0} -row_norm {0}")
                print(datetime.now() - now )

for r in [2]:
    for c in [0.001]:
        for type in ['p-rw']:#["ARKU_plus"]:#, "ARKL"]:# ['exponential', 'p-rw']:#
            for tmax in [10]:
                now = datetime.now()
                print(now)
                print(f'{r} {c} {type} {tmax} ')
                os.system(f"python Experiments/SBMAttributeNormal/run.py -p data/SBMAttributeNormal/RW/n1_{nr_sample}_n2_{nr_sample}_type_{type}_tmax_{tmax}_r_{r}_c_{c}_norm_0_noise1_{noise1}_noise2_{noise2}_m11_{mean11}_m12_{mean12}_m13_{mean13}_m21_{mean21}_m22_{mean22}_m23_{mean23}_new.pkl -B {B} -N {N} -n1 {nr_sample} -n2 {nr_sample}  -mean11 {mean11} -mean12 {mean12} -mean13 {mean13} -mean21 {mean21} -mean22 {mean22} -mean23 {mean23} -noise1 {noise1} -noise2 {noise2} -norm 0 -d 4 -kernel rw -rwApprox {r} -l {c} -type {type} -tmax {tmax} -adj_norm {0} -row_norm {0}")
                print(datetime.now() - now )

for r in [6]:
    for c in [0.001]:
        for type in ["ARKU_plus"]:#, "ARKL"]:# ['exponential', 'p-rw']:#
            now = datetime.now()
            print(now)
            print(f'{r} {c} {type} ')
            os.system(f"python Experiments/SBMAttributeNormal/run.py -p data/SBMAttributeNormal/RW/n1_{nr_sample}_n2_{nr_sample}_type_{type}_tmax_{tmax}_r_{r}_c_{c}_norm_0_noise1_{noise1}_noise2_{noise2}_m11_{mean11}_m12_{mean12}_m13_{mean13}_m21_{mean21}_m22_{mean22}_m23_{mean23}_new.pkl -B {B} -N {N} -n1 {nr_sample} -n2 {nr_sample}  -mean11 {mean11} -mean12 {mean12} -mean13 {mean13} -mean21 {mean21} -mean22 {mean22} -mean23 {mean23} -noise1 {noise1} -noise2 {noise2} -norm 0 -d 4 -kernel rw -rwApprox {r} -l {c} -type {type} -adj_norm {0} -row_norm {0}")
            print(datetime.now() - now )


for tmax in [1, 2, 4, 6, 8, 10, 12]:
    for w in [0.1, 0.01, 0.001]:
        for M in ['L1']:
            now = datetime.now()
            print(now)
            print(f'{tmax} {w} {M}')
            os.system(f"python Experiments/SBMAttributeNormal/run.py -p data/SBMAttributeNormal/PROP/n1_{nr_sample}_n2_{nr_sample}_tmax_{tmax}_w_{w}_M_{M}_norm_0_noise1_{noise1}_noise2_{noise2}_m11_{mean11}_m12_{mean12}_m13_{mean13}_m21_{mean21}_m22_{mean22}_m23_{mean23}_new.pkl -B {B} -N {N} -n1 {nr_sample} -n2 {nr_sample}  -mean11 {mean11} -mean12 {mean12} -mean13 {mean13} -mean21 {mean21} -mean22 {mean22} -mean23 {mean23} -noise1 {noise1} -noise2 {noise2} -norm 0 -d 4 -kernel prop -M {M} -tmax {tmax} -w {w}")
            print(datetime.now() - now )

# gntk

# print(os.getcwd())
# for nl in [1, 2, 4, 6]:
#     for mlp in [1, 2, 4, 6]:
#         for s in ['uniform']:
#             for jk in [1]:
#                 print(f'{nl} {mlp} {s} {jk}')
#                 pp = f'data/SBMAttributeNormal/GNTK/n1_{nr_sample}_n2_{nr_sample}_L_{nl}_mlp_{mlp}_s_{s}_jk_{jk}_norm_0_noise1_{noise1}_noise2_{noise2}_m11_{mean11}_m12_{mean12}_m13_{mean13}_m21_{mean21}_m22_{mean22}_m23_{mean23}.pkl'
#                 os.system(f"python Experiments/SBMAttributeNormal/run.py -p {pp} -B {B} -N {N} -n1 {nr_sample} -n2 {nr_sample} -mean11 {mean11} -mean12 {mean12} -mean13 {mean13} -mean21 {mean21} -mean22 {mean22} -mean23 {mean23} -noise1 {noise1} -noise2 {noise2} -norm 0 -d 8 -kernel gntk -norm 0 -L {nl} -dim {mlp} -sk {jk} -type {s}")


# pp = f'data/SBMAttributeNormal/GH/n1_{nr_sample}_n2_{nr_sample}_type_linear_mu_None_norm_0_noise1_{noise1}_noise2_{noise2}_m11_{mean11}_m12_{mean12}_m13_{mean13}_m21_{mean21}_m22_{mean22}_m23_{mean23}.pkl'
# os.system(f"python Experiments/SBMAttributeNormal/run.py -p {pp} -B {B} -N {N} -n1 {nr_sample} -n2 {nr_sample} -mean11 {mean11} -mean12 {mean12} -mean13 {mean13} -mean21 {mean21} -mean22 {mean22} -mean23 {mean23} -noise1 {noise1} -noise2 {noise2} -norm 0 -d 8 -kernel gh -norm 0 -type linear")



# -p {pp} -B {B} -N {N} -n1 {nr_sample} -n2 {nr_sample} -mean11 {mean11} -mean12 {mean12} -mean13 {mean13} -mean21 {mean21} -mean22 {mean22} -mean23 {mean23} -noise1 {noise1} -noise2 {noise2} -norm 0 -d 4 -kernel gntk -norm 0 -L {nl} -dim {mlp} -sk {jk} -type {s}