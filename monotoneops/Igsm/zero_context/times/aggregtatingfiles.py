import os
import json

# Define file prefix and range
def workerr(filename, num): 
    prefix = filename 
    files_to_aggregate = [f"{prefix}_{i}.jsonl" for i in range(num)]
    output_file = prefix + ".jsonl" 

    # Aggregate data
    aggregated_data = []
    for file in files_to_aggregate:
        if os.path.exists(file):
            with open(file, "r") as f:
                for line in f:
                    aggregated_data.append(json.loads(line))

    # Write aggregated data to a new file
    with open(output_file, "w") as f:
        for entry in aggregated_data:
            f.write(json.dumps(entry) + "\n")

    # Remove original files
    for file in files_to_aggregate:
        if os.path.exists(file):
            os.remove(file)

    print(f"Aggregated {len(aggregated_data)} entries into {output_file} and removed the original files.") 

if __name__ == "__main__": 
    # Define file prefix and range 
    for i in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]: 
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
            # 20: 20, 
        } 
            
        filename = "{}/igsm_op{}_ip{}_force_True".format(i, i, ip[i]) 
        num = 8 

        # Aggregate data
        workerr(filename, num) 
