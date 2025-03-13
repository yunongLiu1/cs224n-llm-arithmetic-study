d=3 
total=100

lengths=(zero_context) 

for length in "${lengths[@]}" 
do 
    # python data_generate3.py --numprocs 8 --opmax 110 --ipmax 50 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 71 72 73 74 75 76 77 78 79 80 
    # python data_generate3.py --numprocs 8 --opmax 96 --ipmax 40 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 61 62 63 64 65 66 67 68 69 70 
    # python data_generate3.py --numprocs 8 --opmax 80 --ipmax 40 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 51 52 53 54 55 56 57 58 59 60 
    # python data_generate3.py --numprocs 8 --opmax 60 --ipmax 30 --total $total --mod -1 --number_range 3 --target_length $length --d $d --force --listoperations 41 42 43 44 45 46 47 48 49 50 
    # python data_generate3.py --numprocs 8 --opmax 50 --ipmax 30 --total $total --mod -1 --number_range 3 --target_length $length --d $d --force --listoperations 31 32 33 34 35 36 37 38 39 40 
    # python data_generate3.py --numprocs 8 --opmax 35 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 27 28 29 30 
    # python data_generate3.py --numprocs 8 --opmax 30 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 21 22 23 24 25 26 
    # python data_generate3.py --numprocs 8 --opmax 25 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 16 17 18 
    
    # python data_generate3.py --numprocs 8 --opmax 20 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 12 13 14 15 
    # python data_generate3.py --numprocs 8 --opmax 15 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 10 11 
    # python data_generate3.py --numprocs 8 --opmax 10 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 7 8 9 
    # python data_generate3.py --numprocs 8 --opmax 6 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 5 6 
    # python data_generate3.py --numprocs 8 --opmax 4 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 4 

    # python data_generate3.py --numprocs 8 --opmax 4 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 3 
    # python data_generate3.py --numprocs 8 --opmax 3 --total $total --mod -1 --number_range 5 --target_length $length --d $d --force --listoperations 2 
    python data_generate3.py --numprocs 8 --opmax 7 --ipmax 5 --total $total --mod -1 --number_range 1000 --target_length $length --d $d --force --plusortimes --listoperations 4 5 
    python data_generate3.py --numprocs 8 --opmax 10 --ipmax 5 --total $total --mod -1 --number_range 1000 --target_length $length --d $d --force --plusortimes --listoperations 6 7 8 
    python data_generate3.py --numprocs 8 --opmax 15 --ipmax 10 --total $total --mod -1 --number_range 1000 --target_length $length --d $d --force --plusortimes --listoperations 9 10 11 
    python data_generate3.py --numprocs 8 --opmax 20 --ipmax 10 --total $total --mod -1 --number_range 1000 --target_length $length --d $d --force --plusortimes --listoperations 12 13 14 15 
    python data_generate3.py --numprocs 8 --opmax 25 --ipmax 20 --total $total --mod -1 --number_range 1000 --target_length $length --d $d --force --plusortimes --listoperations 16 17 18 19 20 
done 
