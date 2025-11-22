import heapq
from collections import defaultdict
import numpy as np
import heapq
import numpy as np
from collections import defaultdict
import time
from prettytable import PrettyTable

from tqdm import tqdm

class SlidingMedian:
    def __init__(self):
        self.max_heap = []
        self.min_heap = []
        self.delayed = defaultdict(int)
        self.size_max = 0
        self.size_min = 0

    def _prune(self, heap):
        while heap:
            x = -heap[0] if heap is self.max_heap else heap[0]
            if self.delayed[x] > 0:
                self.delayed[x] -= 1
                heapq.heappop(heap)
            else:
                break

    def _rebalance(self):
        if self.size_max > self.size_min + 1:
            x = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, x)
            self.size_max -= 1
            self.size_min += 1
            self._prune(self.max_heap)
        elif self.size_min > self.size_max:
            x = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -x)
            self.size_min -= 1
            self.size_max += 1
            self._prune(self.min_heap)

    def add(self, x):
        if not self.max_heap or x <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -x)
            self.size_max += 1
        else:
            heapq.heappush(self.min_heap, x)
            self.size_min += 1
        self._rebalance()

    def remove(self, x):
        self.delayed[x] += 1
        if x <= -self.max_heap[0]:
            self.size_max -= 1
            if x == -self.max_heap[0]:
                self._prune(self.max_heap)
        else:
            self.size_min -= 1
            if self.min_heap and x == self.min_heap[0]:
                self._prune(self.min_heap)
        self._rebalance()

    def median(self):
        if self.size_max > self.size_min:
            return float(-self.max_heap[0])
        else:
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0

def median_sliding_window(nums: np.ndarray, k: int) -> np.ndarray:
    n = nums.size
    out = np.empty(n - k + 1, dtype=float)

    sm = SlidingMedian()

    for i in range(k):
        sm.add(nums[i])
    out[0] = sm.median()

    for i in range(k, n):
        sm.add(nums[i])
        sm.remove(nums[i - k])
        out[i - k + 1] = sm.median()

    return out

def median_sliding_window_naive(nums: np.ndarray, k: int) -> np.ndarray:
    n = nums.size
    out = np.empty(n - k + 1, dtype=float)

    for i in range(n - k + 1):
        window = np.sort(nums[i:i+k])
        if k % 2 == 1:
            out[i] = window[k // 2]
        else:
            out[i] = (window[k//2 - 1] + window[k//2]) / 2.0

    return out


def generate_test(n=None, k=None, low=10_000, high=100_000):
    if n is None:
        n = np.random.randint(low, high)
    if k is None:
        k = np.random.randint(2, min(10, n))
    arr = np.random.randint(-100, 100, size=n)
    return arr, n, k


def benchmark_multiple(n_values, k_values, repeats=5):
    """
    n_values: list of n sizes  (e.g. [1000, 10000, 100000])
    k_values: list of k values (e.g. [3, 10, 77, 154])
    repeats:  number of repeats per (n,k) pair
    """

    results_table = PrettyTable()
    results_table.field_names = [
        "n", "k",
        "Fast min (ms)",
        "Slow min (ms)",
        "Speedup (x)"
    ]

    for n in n_values:
        for k in k_values:
            assert k <= n

            fast_min = float("inf")
            slow_min = float("inf")

            # repeated runs for min time
            for _ in tqdm(range(repeats), desc=f"n={n}, k={k}", leave=False):
                nums, _, _ = generate_test(n, k)

                # --- heaps ---
                start = time.perf_counter()
                fast = median_sliding_window(nums, k)
                t_fast = (time.perf_counter() - start) * 1000
                fast_min = min(fast_min, t_fast)

                # --- naive ---
                start = time.perf_counter()
                slow = median_sliding_window_naive(nums, k)
                t_slow = (time.perf_counter() - start) * 1000
                slow_min = min(slow_min, t_slow)

                # correctness check
                if not np.allclose(fast, slow):
                    print("\nMismatch detected!")
                    print(f"{n=}, {k=}")
                    return

            speedup = slow_min / fast_min if fast_min > 0 else float("inf")

            results_table.add_row([
                n, k,
                f"{fast_min:.4f}",
                f"{slow_min:.4f}",
                f"{speedup:.2f}×"
            ])

    print("\nAll sliding-median tests completed!\n")
    print(results_table)
    return results_table


if __name__ == "__main__":
    benchmark_multiple(n_values=[1000, 10_000, 100_000, 1_000_000],
                       k_values=[10, 100, 500],
                       repeats=25)


    
