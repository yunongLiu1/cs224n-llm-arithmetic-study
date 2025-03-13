 model=google/recurrentgemma-9b-it 


output_dir="./results"


mkdir -p "$output_dir"

contextlengths=(zero_context) 
for contextlength in "${contextlengths[@]}" 
do 
    modes=(3) 
    for mode in "${modes[@]}" 
    do 
	op=(16 17 18 19 20) 
        for i in "${op[@]}" 
        do 
            accelerate launch --main_process_port 29501 --num_processes 8 evaluate_two_recurrent.py --op $i --ip 20 --add_fewshot --limit 100 --testsuite $contextlength --force --modelname $model --batch_size 1 --d $mode --verbose | tee -a outputtwo.txt 
        done 
    done 
done 
