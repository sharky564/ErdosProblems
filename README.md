**Problem 396:**
Compiling: I use GCC 15.2
```
g++ -O3 -march=native -std=c++23 .\erdos_problem_396.cpp -o erdos_problem_396.exe
```
Current performance:
```
Detected 8 logical cores. Using C++ Thread Pool...
Generating primes up to 200,000,000...
Primes generated in 1.65097 seconds.

k =  1 | min n =               2 | Time:       0.2347 s | Speed:     0.00 M candidates/s
k =  2 | min n =            2480 | Time:       0.1800 s | Speed:     0.01 M candidates/s
k =  3 | min n =            8178 | Time:       0.2188 s | Speed:     0.03 M candidates/s
k =  4 | min n =           45153 | Time:       0.2427 s | Speed:     0.15 M candidates/s
k =  5 | min n =         3648841 | Time:       0.2374 s | Speed:    15.18 M candidates/s
k =  6 | min n =         7979090 | Time:       0.2548 s | Speed:    17.00 M candidates/s
k =  7 | min n =       101130029 | Time:       0.4642 s | Speed:   200.65 M candidates/s
k =  8 | min n =       339949252 | Time:       0.7855 s | Speed:   304.02 M candidates/s
k =  9 | min n =      1019547844 | Time:       2.4136 s | Speed:   281.57 M candidates/s
k = 10 | min n =     17609764994 | Time:      71.7100 s | Speed:   231.35 M candidates/s
```
