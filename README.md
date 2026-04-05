**Problem 389:**
Compiling: I use GCC 15.2
```
g++ -O3 -march=native -std=c++23 .\erdos_problem_389.cpp -o erdos_problem_389.exe
```
Current performance:
```
Detected 8 logical cores. Using C++ Thread Pool...
Generating primes up to 200,000,000...
Primes generated in 1.5962 seconds.

n =  1 | min k =               1 | Time:       0.0000 s | Speed:     3.33 M candidates/s
n =  2 | min k =               5 | Time:       0.0115 s | Speed:     0.00 M candidates/s
n =  3 | min k =               4 | Time:       0.0150 s | Speed:     0.00 M candidates/s
n =  4 | min k =             207 | Time:       0.0101 s | Speed:     0.02 M candidates/s
n =  5 | min k =             206 | Time:       0.0093 s | Speed:     0.02 M candidates/s
n =  6 | min k =            2475 | Time:       0.0089 s | Speed:     0.28 M candidates/s
n =  7 | min k =             984 | Time:       0.0088 s | Speed:     0.11 M candidates/s
n =  8 | min k =            8171 | Time:       0.0084 s | Speed:     0.97 M candidates/s
n =  9 | min k =            8170 | Time:       0.0065 s | Speed:     1.25 M candidates/s
n = 10 | min k =           45144 | Time:       0.0073 s | Speed:     6.15 M candidates/s
n = 11 | min k =           45143 | Time:       0.0071 s | Speed:     6.35 M candidates/s
n = 12 | min k =         3648830 | Time:       0.0094 s | Speed:   386.75 M candidates/s
n = 13 | min k =         3648829 | Time:       0.0115 s | Speed:   317.71 M candidates/s
n = 14 | min k =         7979077 | Time:       0.0163 s | Speed:   488.12 M candidates/s
n = 15 | min k =         7979076 | Time:       0.0152 s | Speed:   525.36 M candidates/s
n = 16 | min k =        58068862 | Time:       0.0437 s | Speed:  1329.41 M candidates/s
n = 17 | min k =        58068861 | Time:       0.0464 s | Speed:  1250.61 M candidates/s
n = 18 | min k =       255278295 | Time:       0.1824 s | Speed:  1399.71 M candidates/s
n = 19 | min k =       255278294 | Time:       0.1970 s | Speed:  1295.72 M candidates/s
n = 20 | min k =      1019547825 | Time:       0.6719 s | Speed:  1517.45 M candidates/s
n = 21 | min k =      1019547824 | Time:       0.6754 s | Speed:  1509.44 M candidates/s
n = 22 | min k =     17609764973 | Time:      14.8726 s | Speed:  1184.04 M candidates/s
n = 23 | min k =     17609764972 | Time:      14.8410 s | Speed:  1186.56 M candidates/s
n = 24 | min k =   1070858041562 | Time:    1374.6043 s | Speed:   779.03 M candidates/s
n = 25 | min k =   1070858041561 | Time:    1505.2951 s | Speed:   711.39 M candidates/s
```


**Problem 396:**
Compiling: I use Clang 21.1.8, building on a box with 5.1GHz Ryzen 7 9800X3D
```
clang++ -O3 -march=native -std=c++23 .\erdos_problem_396.cpp -o erdos_problem_396.exe -pthread
```
Current performance:
```
Detected 8 logical cores. Using C++ Thread Pool...
Generating primes up to 200,000,000...
Primes generated in 0.617694 seconds.

k =  1 | min n =               2 | Time:       0.0037 s | Speed:     0.00 M candidates/s
k =  2 | min n =            2480 | Time:       0.0026 s | Speed:     0.96 M candidates/s
k =  3 | min n =            8178 | Time:       0.0022 s | Speed:     2.58 M candidates/s
k =  4 | min n =           45153 | Time:       0.0021 s | Speed:    17.26 M candidates/s
k =  5 | min n =         3648841 | Time:       0.0020 s | Speed:  1791.85 M candidates/s
k =  6 | min n =         7979090 | Time:       0.0016 s | Speed:  2641.37 M candidates/s
k =  7 | min n =       101130029 | Time:       0.0190 s | Speed:  4901.29 M candidates/s
k =  8 | min n =       339949252 | Time:       0.0461 s | Speed:  5179.35 M candidates/s
k =  9 | min n =      1019547844 | Time:       0.1362 s | Speed:  4990.57 M candidates/s
k = 10 | min n =     17609764994 | Time:       3.5138 s | Speed:  4721.40 M candidates/s
k = 11 | min n =   1070858041585 | Time:     286.0180 s | Speed:  3682.45 M candidates/s
```
