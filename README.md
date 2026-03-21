Compilation for Problem 396: 
I use GCC 15.2, and I used TBB for multithreading, though I needed to manually find and link it (the last two flags here).
```
g++ -O3 -march=native -std=c++23 .\erdos_problem_396.cpp -o erdos_problem_396.exe -LC:/msys64/mingw64/lib -ltbb12
```
