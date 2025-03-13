from datasets import load_dataset 
from tqdm import tqdm 

# d = 2 
# d = 3 

datafiles = {} 
path = "Igsm/zero_context/plus/" 

# fourthtransition = 90 if d == 2 else 80 
for i in tqdm(range(4, 21)): 
    ip = {
        4: 5, 
        5: 5, 
        6: 5, 
        7: 5, 
        8: 5, 
        9: 10, 
        10: 10, 
        11: 10, 
        12: 10, 
        13: 10, 
        14: 10, 
        15: 10, 
        16: 20, 
        17: 20, 
        18: 20, 
        19: 20, 
        20: 20, 
    } 
    datafiles["ops_{}".format(i)] = path + "{}/".format(i) + "igsm_op{}_ip{}_force_True.jsonl".format(i, ip[i]) 
    # datafiles["ops {}".format(i)] = path + "igsm_op{}_ip20_force_True_0.jsonl".format(i) 

datasets = load_dataset("json", data_files = datafiles) 
print(datasets) 

datasets.push_to_hub("YangZhoumill/operation_times", private = False) 
