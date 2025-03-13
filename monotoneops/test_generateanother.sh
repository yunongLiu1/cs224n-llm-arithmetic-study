d=2 
total=80 

lengths=(zero_context) 

for length in "${lengths[@]}" 
do 
    python data_generate3.py --numprocs 16 --opmax 180 --ipmax 90 --total $total --mod -1 --number_range 5 --target_length $length --d $d --usemax  --force --listoperations 141 142 143 144 145 146 147 148 149 150 
    # python data_generate3.py --numprocs 16 --opmax 160 --ipmax 85 --total $total --mod -1 --number_range 5 --target_length $length --d $d --usemax --force --listoperations 131 132 133 134 135 136 137 138 139 140 
    
    # python data_generate3.py --numprocs 16 --opmax 150 --ipmax 75 --total $total --mod -1 --number_range 5 --target_length $length --d $d --usemax --force --listoperations 100 101 102 103 104 105 106 107 108 109 110 
    # python data_generate3.py --numprocs 16 --opmax 140 --ipmax 70 --total $total --mod -1 --number_range 5 --target_length $length --d $d --usemax --force --listoperations 91 92 93 94 95 96 97 98 99 100 
    # python data_generate3.py --numprocs 8 --opmax 25 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 16 17 18 
    
    # python data_generate3.py --numprocs 8 --opmax 20 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 12 13 14 15 
    # python data_generate3.py --numprocs 8 --opmax 15 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 10 11 
    # python data_generate3.py --numprocs 8 --opmax 10 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 7 8 9 
    # python data_generate3.py --numprocs 8 --opmax 6 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 5 6 
    # python data_generate3.py --numprocs 8 --opmax 4 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 4 

    # python data_generate3.py --numprocs 8 --opmax 4 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 3 
    # python data_generate3.py --numprocs 8 --opmax 3 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 2 
done 
