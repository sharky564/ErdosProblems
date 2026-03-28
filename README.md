**Problem 389:**
Compiling: I use GCC 15.2
```
g++ -O3 -march=native -std=c++23 .\erdos_problem_389.cpp -o erdos_problem_389.exe
```
Current performance:
```
Detected 8 logical cores. Using C++ Thread Pool...
Generating primes up to 200,000,000...
Primes generated in 1.47169 seconds.

n =  1 | min k =            1 | Time:   0.0000 s | Speed:     2.00 M candidates/s
n =  2 | min k =            5 | Time:   5.9942 s | Speed:     0.00 M candidates/s
n =  3 | min k =            4 | Time:   6.2366 s | Speed:     0.00 M candidates/s
n =  4 | min k =          207 | Time:   6.2200 s | Speed:     0.00 M candidates/s
n =  5 | min k =          206 | Time:   6.3315 s | Speed:     0.00 M candidates/s
n =  6 | min k =         2475 | Time:   6.2493 s | Speed:     0.00 M candidates/s
n =  7 | min k =          984 | Time:   6.2682 s | Speed:     0.00 M candidates/s
n =  8 | min k =         8171 | Time:   6.2045 s | Speed:     0.00 M candidates/s
n =  9 | min k =         8170 | Time:   6.2722 s | Speed:     0.00 M candidates/s
n = 10 | min k =        45144 | Time:   6.1092 s | Speed:     0.01 M candidates/s
n = 11 | min k =        45143 | Time:   6.2064 s | Speed:     0.01 M candidates/s
n = 12 | min k =      3648830 | Time:   6.5104 s | Speed:     0.56 M candidates/s
n = 13 | min k =      3648829 | Time:   6.5536 s | Speed:     0.56 M candidates/s
n = 14 | min k =      7979077 | Time:   8.5048 s | Speed:     0.94 M candidates/s
n = 15 | min k =      7979076 | Time:   8.6225 s | Speed:     0.93 M candidates/s
n = 16 | min k =     58068862 | Time:  16.9073 s | Speed:     3.43 M candidates/s
n = 17 | min k =     58068861 | Time:  17.0802 s | Speed:     3.40 M candidates/s
n = 18 | min k =    255278295 | Time:  58.1312 s | Speed:     4.39 M candidates/s
n = 19 | min k =    255278294 | Time:  58.7075 s | Speed:     4.35 M candidates/s
```


**Problem 396:**
Compiling: I use GCC 15.2
```
g++ -O3 -march=native -std=c++23 .\erdos_problem_396.cpp -o erdos_problem_396.exe
```
Current performance:
```
Detected 8 logical cores. Using C++ Thread Pool...
Generating primes up to 200,000,000...
Primes generated in 1.41729 seconds.

k =  1 | min n =               2 | Time:       0.0072 s | Speed:     0.00 M candidates/s
k =  2 | min n =            2480 | Time:       0.0086 s | Speed:     0.29 M candidates/s
k =  3 | min n =            8178 | Time:       0.0078 s | Speed:     0.73 M candidates/s
k =  4 | min n =           45153 | Time:       0.0089 s | Speed:     4.13 M candidates/s
k =  5 | min n =         3648841 | Time:       0.0114 s | Speed:   317.31 M candidates/s
k =  6 | min n =         7979090 | Time:       0.0112 s | Speed:   386.16 M candidates/s
k =  7 | min n =       101130029 | Time:       0.0899 s | Speed:  1036.00 M candidates/s
k =  8 | min n =       339949252 | Time:       0.2474 s | Speed:   965.39 M candidates/s
k =  9 | min n =      1019547844 | Time:       0.8597 s | Speed:   790.54 M candidates/s
k = 10 | min n =     17609764994 | Time:      23.6000 s | Speed:   702.97 M candidates/s
```
