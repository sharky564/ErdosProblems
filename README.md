**Problem 396:**
Compiling: I use GCC 15.2
```
g++ -O3 -march=native -std=c++23 .\erdos_problem_396.cpp -o erdos_problem_396.exe
```
Current performance:
```
Detected 8 logical cores. Using C++ Thread Pool...
Generating primes up to 200,000,000...
Primes generated in 1.83339 seconds.

k =  1 | min n =               2 | Time:       0.2033 s | Speed:     0.00 M candidates/s
k =  2 | min n =            2480 | Time:       0.1665 s | Speed:     0.01 M candidates/s
k =  3 | min n =            8178 | Time:       0.1670 s | Speed:     0.03 M candidates/s
k =  4 | min n =           45153 | Time:       0.1912 s | Speed:     0.19 M candidates/s
k =  5 | min n =         3648841 | Time:       0.1927 s | Speed:    18.70 M candidates/s
k =  6 | min n =         7979090 | Time:       0.2183 s | Speed:    19.84 M candidates/s
k =  7 | min n =       101130029 | Time:       0.3827 s | Speed:   243.39 M candidates/s
k =  8 | min n =       339949252 | Time:       0.5959 s | Speed:   400.77 M candidates/s
k =  9 | min n =      1019547844 | Time:       1.8460 s | Speed:   368.15 M candidates/s
k = 10 | min n =     17609764994 | Time:      59.5331 s | Speed:   278.67 M candidates/s
```
