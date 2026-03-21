**Problem 396:**
Compiling: I use GCC 15.2, and I used TBB for multithreading, though I needed to manually find and link it (the last two flags here).
```
g++ -O3 -march=native -std=c++23 .\erdos_problem_396.cpp -o erdos_problem_396.exe -LC:/msys64/mingw64/lib -ltbb12
```
I ran the program overnight, and achieved the following results:
```
Generating primes up to 5,000,000...
Primes generated in 0.0200225 seconds.

k =  1 | min n =             2 | Time:     0.5462 s | Speed:     0.00 M candidates/s
k =  2 | min n =          2480 | Time:     0.5319 s | Speed:     0.00 M candidates/s
k =  3 | min n =          8178 | Time:     0.4341 s | Speed:     0.02 M candidates/s
k =  4 | min n =         45153 | Time:     0.4562 s | Speed:     0.10 M candidates/s
k =  5 | min n =       3648841 | Time:     0.4166 s | Speed:     8.76 M candidates/s
k =  6 | min n =       7979090 | Time:     0.4530 s | Speed:    17.61 M candidates/s
k =  7 | min n =     101130029 | Time:     1.3541 s | Speed:    74.68 M candidates/s
k =  8 | min n =     339949252 | Time:     4.4740 s | Speed:    75.98 M candidates/s
k =  9 | min n =    1019547844 | Time:    14.2317 s | Speed:    71.64 M candidates/s
k = 10 | min n =   17609764994 | Time:   253.4640 s | Speed:    69.48 M candidates/s
k = 11 | min n = 1070858041585 | Time: 18082.2002 s | Speed:    59.22 M candidates/s
```
