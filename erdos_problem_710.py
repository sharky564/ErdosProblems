import numpy as np
import numba as nb
import math
import time

@nb.njit
def sift_down(h_val, h_idx, n, i):
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i

        if left < n and h_val[left] < h_val[smallest]:
            smallest = left
        if right < n and h_val[right] < h_val[smallest]:
            smallest = right

        if smallest != i:
            tmp_val = h_val[i]
            h_val[i] = h_val[smallest]
            h_val[smallest] = tmp_val

            tmp_idx = h_idx[i]
            h_idx[i] = h_idx[smallest]
            h_idx[smallest] = tmp_idx

            i = smallest
        else:
            break

@nb.njit
def _f_numba(n, MAX_V, MAX_E):
    match_U = np.zeros(n + 1, dtype=np.int32)
    match_V = np.zeros(MAX_V, dtype=np.int32)

    head = np.full(MAX_V, -1, dtype=np.int32)
    to = np.zeros(MAX_E, dtype=np.int32)
    nxt = np.zeros(MAX_E, dtype=np.int32)
    edge_cnt = 0

    h_val = np.zeros(n, dtype=np.int64)
    h_idx = np.zeros(n, dtype=np.int32)

    for i in range(1, n + 1):
        h_val[i - 1] = ((n // i) + 1) * i
        h_idx[i - 1] = i

    for i in range(n // 2, -1, -1):
        sift_down(h_val, h_idx, n, i)

    queue = np.zeros(MAX_V, dtype=np.int32)
    visited_U = np.zeros(n + 1, dtype=np.int32)
    parent_U = np.zeros(n + 1, dtype=np.int32)
    visit_token = 0

    divs = np.zeros(n + 1, dtype=np.int32)

    match_size = 0
    v = n

    while match_size < n:
        v += 1

        if v >= MAX_V:
            return -1

        div_cnt = 0
        while h_val[0] == v:
            j = h_idx[0]
            divs[div_cnt] = j
            div_cnt += 1

            h_val[0] = v + j
            sift_down(h_val, h_idx, n, 0)

        if div_cnt == 0:
            continue

        divs_slice = divs[:div_cnt]
        divs_slice.sort()

        for i in range(div_cnt - 1, -1, -1):
            j = divs_slice[i]

            if edge_cnt >= MAX_E:
                return -1

            to[edge_cnt] = j
            nxt[edge_cnt] = head[v]
            head[v] = edge_cnt
            edge_cnt += 1

        visit_token += 1
        q_head = 0
        q_tail = 0
        queue[q_tail] = v
        q_tail += 1

        unmatched_j = 0

        while q_head < q_tail:
            curr_v = queue[q_head]
            q_head += 1

            e = head[curr_v]
            while e != -1:
                j = to[e]
                if visited_U[j] != visit_token:
                    visited_U[j] = visit_token
                    parent_U[j] = curr_v

                    if match_U[j] == 0:
                        unmatched_j = j
                        break
                    else:
                        queue[q_tail] = match_U[j]
                        q_tail += 1
                e = nxt[e]

            if unmatched_j != 0:
                break

        if unmatched_j != 0:
            curr_j = unmatched_j
            while True:
                prev_v = parent_U[curr_j]
                if prev_v == v:
                    match_U[curr_j] = prev_v
                    match_V[prev_v] = curr_j
                    break

                j_prev = match_V[prev_v]
                match_U[curr_j] = prev_v
                match_V[prev_v] = curr_j
                curr_j = j_prev

            match_size += 1

    return v - n + 1

def f(n):
    if n == 1: return 2

    k_est = int(n * (math.log(n) + 2) + 1000)
    MAX_V = n + k_est

    MAX_E = int(MAX_V * (math.log(n) + 3))

    while True:
        res = _f_numba(n, MAX_V, MAX_E)
        if res != -1:
            return res

        MAX_V = int(MAX_V * 1.5)
        MAX_E = int(MAX_E * 1.5)

_ = f(10)
values = []
with open('b390246-2.txt', 'w') as file:
    for n in range(1, 10001):
        x = f(n)
        values.append((n, x))
        print(' '*20, end='\r')
        print(n, x, end='\r')
        print(n, x, file=file)