**Problem 389:**
Compiling: I use GCC 15.2
```
g++ -O3 -march=native -std=c++23 .\erdos_problem_389.cpp -o erdos_problem_389.exe
```
Current performance:
```
Detected 8 logical cores. Using C++ Thread Pool...
Generating primes up to 200,000,000...
Primes generated in 1.52039 seconds.

n =  1 | min k =            1 | Time:     0.0000 s | Speed:     0.19 M candidates/s
n =  2 | min k =            5 | Time:     7.8846 s | Speed:     0.00 M candidates/s
n =  3 | min k =            4 | Time:     8.3063 s | Speed:     0.00 M candidates/s
n =  4 | min k =          207 | Time:     8.3735 s | Speed:     0.00 M candidates/s
n =  5 | min k =          206 | Time:     8.9155 s | Speed:     0.00 M candidates/s
n =  6 | min k =         2475 | Time:     8.9218 s | Speed:     0.00 M candidates/s
n =  7 | min k =          984 | Time:     8.6139 s | Speed:     0.00 M candidates/s
n =  8 | min k =         8171 | Time:     8.4136 s | Speed:     0.00 M candidates/s
n =  9 | min k =         8170 | Time:     8.5763 s | Speed:     0.00 M candidates/s
n = 10 | min k =        45144 | Time:     8.7100 s | Speed:     0.01 M candidates/s
n = 11 | min k =        45143 | Time:     8.7577 s | Speed:     0.01 M candidates/s
n = 12 | min k =      3648830 | Time:     9.5455 s | Speed:     0.38 M candidates/s
n = 13 | min k =      3648829 | Time:     9.3620 s | Speed:     0.39 M candidates/s
n = 14 | min k =      7979077 | Time:    13.8112 s | Speed:     0.58 M candidates/s
n = 15 | min k =      7979076 | Time:    13.4043 s | Speed:     0.60 M candidates/s
n = 16 | min k =     58068862 | Time:    32.9392 s | Speed:     1.76 M candidates/s
n = 17 | min k =     58068861 | Time:    30.3987 s | Speed:     1.91 M candidates/s
n = 18 | min k =    255278295 | Time:   152.7693 s | Speed:     1.67 M candidates/s
n = 19 | min k =    255278294 | Time:   127.6790 s | Speed:     2.00 M candidates/s
n = 20 | min k =   1019547825 | Time:   523.2445 s | Speed:     1.95 M candidates/s
n = 21 | min k =   1019547824 | Time:   585.0277 s | Speed:     1.74 M candidates/s
n = 22 | min k =  17609764973 | Time: 11151.8843 s | Speed:     1.58 M candidates/s
n = 23 | min k =  17609764972 | Time:  8798.7524 s | Speed:     2.00 M candidates/s
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
Primes generated in 1.52039 seconds.

k =  1 | min n =               2 | Time:       0.0168 s | Speed:     0.00 M candidates/s
k =  2 | min n =            2480 | Time:       0.0148 s | Speed:     0.17 M candidates/s
k =  3 | min n =            8178 | Time:       0.0131 s | Speed:     0.44 M candidates/s
k =  4 | min n =           45153 | Time:       0.0112 s | Speed:     3.30 M candidates/s
k =  5 | min n =         3648841 | Time:       0.0201 s | Speed:   179.68 M candidates/s
k =  6 | min n =         7979090 | Time:       0.0193 s | Speed:   224.62 M candidates/s
k =  7 | min n =       101130029 | Time:       0.1916 s | Speed:   486.28 M candidates/s
k =  8 | min n =       339949252 | Time:       0.5722 s | Speed:   417.40 M candidates/s
k =  9 | min n =      1019547844 | Time:       1.4225 s | Speed:   477.76 M candidates/s
k = 10 | min n =     17609764994 | Time:      44.7478 s | Speed:   370.75 M candidates/s
```
