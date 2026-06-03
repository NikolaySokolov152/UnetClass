import numpy as np

num_experiments = 20

str_num = []

for id_experiment in range(num_experiments):
    random_seed = np.random.randint(np.iinfo(np.int32).max)
    str_num.append(f"{random_seed}")
    
print(str_num)
print(" ".join(str_num))