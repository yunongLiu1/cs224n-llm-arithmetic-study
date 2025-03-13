import torch 
from torch.utils.data.distributed import DistributedSampler 
import torch.distributed as dist 
import transformers 
from accelerate import Accelerator 
from datasets import load_dataset 

from transformers import GPT2Tokenizer 

from transformers import AutoTokenizer 
from transformers import AutoConfig 
from transformers import AutoModelForCausalLM 
from transformers import LlamaForCausalLM 

import numpy as np 
from datasets import concatenate_datasets 
from datasets import Dataset 
from torch.utils.data import DataLoader 
from typing import List, Literal, Optional, Tuple, Union 
import argparse 
from tqdm import tqdm 
from termcolor import colored 
from tabulate import tabulate 
import copy 
import os 
import random 
import subprocess 
import re 
import json 

from simple_names_three import message, messagetwo, messagethree 

### Generate with custom termination condition ### 

# naming_match = {
#     "zero_context": "YangZhoumill/infini_gsm_zero_context", 
#     "4k_close": "YangZhoumill/infini_igsm_4k_noise_close", 
#     "4k_general": "YangZhoumill/infini_gsm_4k_noise_general", 
#     "8k_close": "YangZhoumill/infini_gsm_8k_noise_close", 
#     "8k_general": "YangZhoumill/infini_gsm_8k_noise_general", 
# } 

def set_seed(seed):
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        
    # CUDA convolution determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

# Usage
set_seed(42)  # You can use any integer value as the seed 

### Parsing the arguments ### 
parser = argparse.ArgumentParser(description = "CommonSense Reasoning with generation and chain-of-thoughts") 
parser.add_argument("--op", type = int, default = 15, help = "Number of operations") 
parser.add_argument("--ip", type = int, default = 20, help = "Number of items per operation") 
parser.add_argument("--force", action = "store_true", help = "Force the generation of the dataset") 
parser.add_argument("--add_fewshot", action = "store_true", help = "Add few-shot learning to the dataset") 
parser.add_argument("--verbose", action = "store_true", help = "Verbose mode") 
parser.add_argument("--limit", type = int, default = None, help = "Limit the number of examples") 
parser.add_argument("--testsuite", type = str, default = "zero_context", help = "Test suite") 
parser.add_argument("--modelname", type = str, default = "meta-llama/Llama-3.1-8B-Instruct", help = "Model name") 
parser.add_argument("--d", type = int, default = None, help = "Depth of the structure") 
parser.add_argument("--batch_size", type = int, default = 1, help = "Batch size") 
parser.add_argument("--template", type = str, default = "crazy_zootopia", help = "Template") 
parser.add_argument("--mode", type = str, default = "normalforward", help = "Mode") 
parser.add_argument("--local_rank", type = int, default = 0, help = "Local rank") 
parser.add_argument("--output_dir", type=str, default="./llm_outputs", help="Directory to save LLM outputs")  # 添加这行

suffaddition = "The following questions might be easier to solve by equations." 

os.environ['NCCL_TIMEOUT'] = '1800'  # For 2 hours 
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
print("NCCL_TIMEOUT {}".format(os.environ['NCCL_TIMEOUT'])) 

from accelerate.utils import InitProcessGroupKwargs 
from datetime import timedelta 

kwargs = InitProcessGroupKwargs(timeout = timedelta(minutes = 60)) 
accelerator = Accelerator(kwargs_handlers=[kwargs]) 

# Check if we are in a distributed setup
is_distributed = accelerator.distributed_type != "NO" 
print("is_distributed {}".format(is_distributed)) 

args = parser.parse_args() 

print(args) 

if is_distributed: 
    device = "cuda:{}".format(accelerator.process_index) if torch.cuda.is_available() else "cpu" 
else: 
    device = "cuda" if torch.cuda.is_available() else "cpu" 

### Loading the tokenizer and the model ### 
tokenizer = AutoTokenizer.from_pretrained(args.modelname) 
tokenizer.padding_side = "left" 
if args.modelname == "meta-llama/Llama-3.1-70B-Instruct": 
    model = LlamaForCausalLM.from_pretrained(args.modelname, device_map = "auto", torch_dtype = torch.bfloat16) 
else: 
    if args.modelname == "meta-llama/Llama-3.1-8B-Instruct": 
        model = LlamaForCausalLM.from_pretrained(args.modelname, device_map = device, torch_dtype = torch.bfloat16) 
    else: 
        model = AutoModelForCausalLM.from_pretrained(
            args.modelname, 
            device_map = device, 
            torch_dtype = torch.bfloat16, 
            # _attn_implementation = "flash_attention_2", 
        ) 

tokenizer.pad_token = tokenizer.eos_token 
model.config.pad_token_id = model.config.eos_token_id 

model.eval() 

### Loading the datasets ### 
def get_dataset(
    opmax = None, 
    batch_size = 1, 
    limit = None, 
    rootpath = None, 
    datafilepath = None, 
    message = None, 
    mode = None, 
    template = None, 
    is_distributed = False, 
): 
    messages = [] 
    # loading the manually written cot prompt 
    # with open("{}_cot_prompts{}.txt".format(datasetname, requirements), "r") as file: 
    # with open(args.testsuite + "igsm_op8_ip{}_force_{}_cot.txt".format(ipmax, force), "r") as file: 
    # loading the actual dataset 
    # filename = "Igsm/medtest/" 
    # setting up question prompt 
    # messageone = "Answer the questions below. Note: any Location's Adult Animal refers to sum of all types of adult animals ever mentioned for the specific location throughout the problem EXCLUDING their number of newborn children. The average number of newborn children of the same type of animal might vary across different locations. The Location's Total Newborn Children refers to the sum of the TOTAL newborn children (not average newborn children) from all adult animals mentioned for that specific location. Each question is independent of the others." 
    messageone = message 
    
    # filename = rootpath + datafilepath 
    
    # dataset = load_dataset("json", data_files = filename, split = "train") 
    dataset = load_dataset(datafilepath, split = "ops_{}".format(opmax)) 
    dataset = dataset.filter(lambda x: x["template"] == template and x["mode"] == mode) 
    
    if limit is not None: 
        limit = min(limit, len(dataset)) 
        dataset = dataset.select(range(limit)) 
    
    print(colored("Number of examples: {}".format(len(dataset)), "green")) 
    
    max_length = 0 
    for i in range(len(dataset)): 
        example = dataset[i] 
        newmessages = [] 
        inputtext = "Problem: " + example["problem"] + " Question: " + example["question"] + " Solution: " 
        # newmessages.append({"role": "user", "content": inputtext}) 
        newmessages.append({"role": "user", "content": messageone + " " + inputtext}) 
        messagestext = tokenizer.apply_chat_template(
            newmessages, 
            tokenize = False, 
            add_generation_prompt = True, 
        ) 
        outputdi = tokenizer(messagestext, return_tensors = "pt")["input_ids"] 
        max_length = max(max_length, outputdi.shape[1]) 
    
    print("max length: {}".format(max_length)) 
    def encodewithtokenizer(examples): 
        batchtext = [] 
        if isinstance(examples["problem"], str): 
            newmessages = [] 
            inputtext = "Problem: " + examples["problem"] + " Question: " + examples["question"] + " Solution: " 
            newmessages.append({"role": "user", "content": messageone + " " + inputtext}) 
            messagestext = tokenizer.apply_chat_template(
                newmessages, 
                tokenize = False, 
                add_generation_prompt = True, 
            ) 
            batchtext.append(messagestext) 
        elif isinstance(examples["problem"], list): 
            for i in range(len(examples["problem"])): 
                newmessages = [] 
                inputtext = "Problem: " + examples["problem"][i] + " Question: " + examples["question"][i] + " Solution: " 
                
                newmessages.append({"role": "user", "content": messageone + " " + inputtext}) 
                messagestext = tokenizer.apply_chat_template(
                    newmessages, 
                    tokenize = False, 
                    add_generation_prompt = True, 
                ) 
                batchtext.append(messagestext) 
        outputdi = tokenizer( 
            batchtext, 
            return_tensors = "pt", 
            padding = "max_length", 
            max_length = max_length, 
            truncation = False, 
            add_special_tokens = True, 
        ) 
        input_ids = [outputdi["input_ids"][i] for i in range(outputdi["input_ids"].shape[0])] 
        attention_mask = [outputdi["attention_mask"][i] for i in range(outputdi["attention_mask"].shape[0])] 
        examples["input_ids"] = input_ids 
        examples["attention_mask"] = attention_mask 
        # here, the start of sequence token isn't added 
        return examples 
    
    dataset = dataset.map(encodewithtokenizer, num_proc = 1, batched = True, batch_size = batch_size) 
    dataset.set_format(type = "torch", columns = ["input_ids", "attention_mask", "solution"]) 
    
    # print("length of dataset: {}".format(len(dataset))) 
    
    if is_distributed: 
        distributedsampler = DistributedSampler(dataset, num_replicas = accelerator.num_processes, rank = accelerator.process_index, drop_last = True, shuffle = True) 
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, sampler = distributedsampler) 
    else: 
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False) 
    return dataloader 

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False 

def criteriaoutput(generatedtext, inputexample): 
    correctedanswers = 0 
    totalanswers = 0 
    # parsing the answer key 
    for i in range(len(generatedtext)): 
        totalanswers += 1 
        # print(colored(inputexample["solution"], "yellow")) 
        idx_answer_start = inputexample["solution"][i].find("Answer: ") 
        idx_answer_end = inputexample["solution"][i].find(".", idx_answer_start) 
        answer_text = inputexample["solution"][i][idx_answer_start + len("Answer: ") : idx_answer_end] 
        answer_text = int(answer_text.lower()) 
        
        generatedtext[i] = re.sub('.\x08', 'b', generatedtext[i])
        generatedtext[i] = generatedtext[i].lower() 
        if args.verbose and args.local_rank == 0: 
            print(colored(inputexample["solution"][i], "yellow"), flush = True) 
            print(colored(generatedtext[i], "cyan"), flush = True) 
        
        idx_generated_begin = -1 
        idx_generated_conclude = -1 
        keywords = ["answer: ", "solution: ", "oxed{", "**answer:** ", "**answer: ", "final answer: answer: ", "\nanswer: ", r"\text{answer: } ", " **", "**answer: answer: ", " "] # updated 
        keywordsend = [".", ".", "}", ".", "**", ".", ".", None, "**", "**", "."] 
        cnt = 0 
        
        while not (idx_generated_begin != -1 and idx_generated_conclude != -1) and cnt < len(keywords): 
            if keywords[cnt] == " **" or keywords[cnt] == " ": 
                idx_generated_begin = generatedtext[i].rfind(keywords[cnt]) 
            else: 
                idx_generated_begin = generatedtext[i].find(keywords[cnt]) 
            if idx_generated_begin != -1: 
                if keywordsend[cnt] is None: 
                    idx_generated_conclude = idx_generated_begin + len(keywords[cnt]) 
                    while generatedtext[0][idx_generated_conclude].isdigit() == True: 
                        idx_generated_conclude += 1 
                else: 
                    idx_generated_conclude = generatedtext[i].find(keywordsend[cnt], idx_generated_begin + len(keywords[cnt])) 
                if idx_generated_conclude == -1: 
                    # idx_generated_conclude = generatedtext[i].find(tokenizer.eos_token, idx_generated_begin) 
                    # idx_generated_conclude = len(generatedtext[i]) 
                    idx_generated_conclude = generatedtext[i].find(tokenizer.eos_token, idx_generated_begin + len(keywords[cnt])) 
            cnt += 1 
            if not is_integer(generatedtext[i][idx_generated_begin + len(keywords[cnt - 1]) : idx_generated_conclude]): 
                idx_generated_begin = -1 
                idx_generated_conclude = -1 
                continue # if not this line, it will exit the loop 
        
        if idx_generated_begin == -1: 
            if args.local_rank == 0: 
                print(colored("Answer not found", "red"), flush = True) 
            correctedanswers += 0 
            continue 
        else: 
            try: 
                answergenerated_text = int(generatedtext[i][idx_generated_begin + len(keywords[cnt - 1]) : idx_generated_conclude]) 
            except: 
                if args.local_rank == 0: 
                    print(colored("Answer not found", "red"), flush = True) 
                # return 0 
                correctedanswers += 0 
                continue 
            if args.local_rank == 0: 
                if answergenerated_text == answer_text: 
                    print(colored("Answer {} expected {}".format(answergenerated_text, answer_text), "green"), flush = True) 
                else: 
                    print(colored("Answer {} expected {}".format(answergenerated_text, answer_text), "red"), flush = True) 
            # return int(answergenerated_text == answer_text) 
            correctedanswers += int(answergenerated_text == answer_text) 
    return correctedanswers, totalanswers 

countaccum = {} 
listtasks = [] 
totalcounttokengenerated = 0 

for mode in ["normalforward", "forwardreverse"]: 
    if mode == "forwardreverse": 
        continue 
    for template in ["crazy_zootopia", "teachers_in_school", "movie_festival_awards"]: 
        if template in ["teachers_in_school", "movie_festival_awards"]: 
            continue 
        task = "op{}_ip{}_force{}_{}_{}".format(args.op, args.ip, args.force, mode, template) 
        listtasks.append(task) 
        countaccum[task] = [0, 0, 0] 

        prompttttt = {
            "crazy_zootopia": message, 
            "teachers_in_school": messagetwo, 
            "movie_festival_awards": messagethree, 
        } 

        if mode == "forwardreverse": 
            for key in prompttttt.keys(): 
                prompttttt[key] += " " + suffaddition 
        
        limit = None 
        
        if template == "crazy_zootopia": 
            limit = int(args.limit * 0.8) 
        else: 
            limit = int(args.limit * 0.1) 
        
        # if mode == "forwardreverse": 
        #     rootpath = "Igsm/{}/{}/{}/".format(args.testsuite, "medium" if args.d == 2 else "hard", args.op - 1) 
        #     # making sure that the reverse mode counts fair 
        #     promptfilepath = "Igsm/zero_context/{}/{}/igsm_op{}_ip20_force_True_{}_{}_cot.txt".format("medium" if args.d == 2 else "hard", args.op - 1, args.op - 1, mode, template) 
        #     datafilepath = "igsm_op{}_ip20_force_True_{}_{}.jsonl".format(args.op - 1, mode, template) 
        # else: 
            
        promptfilepath = "Igsm/zero_context/{}/{}/igsm_op{}_ip20_force_True_{}_{}_cot.txt".format("mediumtemp" if args.d == 2 else "hard", args.op, args.op, args.mode, args.template) 
        # datafilepath = "igsm_op{}_ip20_force_True_{}_{}.jsonl".format(args.op, args.mode, args.template) 
        # datasetpath = "YangZhoumill/factor_{}_{}".format("medium" if args.d == 2 else "hard", args.testsuite if args.testsuite != "zero_context" else "zerocontext") 
        # datasetpath = "YangZhoumill/factorreit_{}_{}".format(args.testsuite if args.testsuite != "zero_context" else "zerocontext", "medium" if args.d == 2 else "hard") 
        datasetpath = 'Yunong/operations_plus'
        # datasetpath = 'Yunong/operation_times'

        dataloader = get_dataset( 
            opmax = args.op, 
            batch_size = args.batch_size, 
            limit = limit, 
            # rootpath = rootpath, 
            datafilepath = datasetpath, 
            message = prompttttt[args.template], 
            mode = mode, 
            template = template, 
            is_distributed = is_distributed, 
        ) 

        totalexamples = 0 
        correctanswers = 0 

        for i, batch in enumerate(tqdm(dataloader)): 
            input_ids = batch["input_ids"] 
            # print("input_ids shape {}".format(input_ids.shape)) 
            input_ids = input_ids.to(device) 
            # print("input_ids[0] {}".format(tokenizer.decode(input_ids[1]))) 
            # print("attention_mask shape {}".format(batch["attention_mask"].shape)) 
            attention_mask = batch["attention_mask"].to(device) 
            # print("attention_mask[1] {}".format(batch["attention_mask"][1])) 
            
            outputs = model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask, 
                max_new_tokens=8192, 
                use_cache = True, 
                # pad_token_id = tokenizer.pad_token_id, 
                do_sample = False, 
            ) 
            # generatedtext = tokenizer.batch_decode(outputs[0][:, input_ids.shape[1] :]) 
            generatedtext = tokenizer.batch_decode(outputs[:, input_ids.shape[1] : ]) 
            if args.verbose: 
                print("full text: {}".format(tokenizer.decode(outputs[0]))) 
            
            if not isinstance(generatedtext, list): 
                generatedtext2 = [generatedtext] 
            else: 
                generatedtext2 = generatedtext 
            for text in generatedtext2: 
                length = text.rfind(tokenizer.eos_token) 
                if length == -1: 
                    totalcounttokengenerated += len(text) 
                else: 
                    totalcounttokengenerated += length 
            
            # check_criteria = criteriaoutput(outputs[:, input_ids.shape[1] : ], batch) 
            corrected, total = criteriaoutput(generatedtext, batch) 
            
            totalexamples += total 
            correctanswers += corrected 

           #  if accelerator.is_main_process:
               # output_data = {
               #     "batch_index": i,
               #     "input_ids": input_ids.tolist(),  # 转换为列表以便 JSON 序列化
               #     "generated_text": generatedtext,
               #     "solution": batch["solution"],
               #     "corrected": corrected,
               #     "total": total
               # }
               # output_file = os.path.join(args.output_dir, f"batch_{i}_output.json")
               # with open(output_file, "w", encoding="utf-8") as f:
               #     json.dump(output_data, f, ensure_ascii=False, indent=4)
            
            # adding synchronization rounds 
            #  if is_distributed and i == len(dataloader)//2: 
            #   dist.barrier()

        print("got here") 

        # statistics 
        headers = ["Task"] 
        # task = "op{}_ip{}_force{}".format(args.op, args.ip, args.force) 
        data = [task] 
        if is_distributed: 
            # print("index {} start communication".format(accelerator.process_index)) 
            dist.barrier() 
            totalexamples = torch.tensor(totalexamples, device = device) 
            correctanswers = torch.tensor(correctanswers, device = device) 
            dist.all_reduce(totalexamples, op = dist.ReduceOp.SUM) 
            dist.all_reduce(correctanswers, op = dist.ReduceOp.SUM) 
            totalexamples = totalexamples.item() 
            correctanswers = correctanswers.item() 
            totalgenerationcount = torch.tensor(totalcounttokengenerated, device = device) 
            dist.all_reduce(totalgenerationcount, op = dist.ReduceOp.SUM) 
            totalgenerationcount = totalgenerationcount.item() 

        # Print table 
        if accelerator.is_main_process: 
            data = [task, totalexamples, correctanswers, correctanswers/totalexamples] 
            print(tabulate([data], headers = headers, tablefmt = "grid")) 
            countaccum[task] = [totalexamples, correctanswers, correctanswers / totalexamples] 
            print("total generation count: {}".format(totalgenerationcount))
        print("got here") 

if accelerator.is_main_process: 
    # formatting the output 
    print(args) 
    os.makedirs(args.output_dir, exist_ok=True)
    
    # stage one 
    for mode in ["normalforward", "forwardreverse"]: 
        taskname = "op{}_aggregate_{}".format(args.op, mode) 
        aggregatetotalexamples = 0 
        aggregatecorrectanswers = 0 
        for keys in countaccum.keys(): 
            if mode in keys: 
                aggregatetotalexamples += countaccum[keys][0] 
                aggregatecorrectanswers += countaccum[keys][1] 
        if aggregatetotalexamples == 0: 
            continue 
        countaccum[taskname] = [aggregatetotalexamples, aggregatecorrectanswers, aggregatecorrectanswers / aggregatetotalexamples] 
    
    # stage another 
    taskname = "op{}_aggregate".format(args.op) 
    aggregatetotalexamples = 0 
    aggregatecorrectanswers = 0 
    for taske in countaccum.keys(): 
        aggregatetotalexamples += countaccum[taske][0] 
        aggregatecorrectanswers += countaccum[taske][1] 
    countaccum[taskname] = [aggregatetotalexamples, aggregatecorrectanswers, aggregatecorrectanswers / aggregatetotalexamples] 
    
    headers = ["Task", "Total", "Correct", "Solve Rate"] 
    data = [] 
    for task in countaccum.keys(): 
        data.append([task, countaccum[task][0], countaccum[task][1], countaccum[task][2]]) 
    print(tabulate(data, headers = headers, tablefmt = "grid")) 
    print("average generation count per example {}".format(totalcounttokengenerated / aggregatetotalexamples)) 
