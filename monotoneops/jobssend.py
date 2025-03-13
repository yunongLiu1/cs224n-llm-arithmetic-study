import subprocess 
import os 

ids = [] 

with open("jobendids.txt", "r") as file:
    jobendids = file.readlines() 
    for line in jobendids: 
        ids.append(line.split(" ")[0]) 

print(" ".join(ids)) 

subprocess.run(["kill", "-9"] + ids)
