# Task 1: Timestamps Matching

| Function | Description | Complexity |
|----------|-------------|------------|
| `match_timestamps` | Basic sequential algorithm. Iterates through `timestamps1` and slides a pointer `j` through `timestamps2` to find the closest match. | **O(n + m)**, where `n = len(timestamps1)`, `m = len(timestamps2)` |
| `match_timestamps_hybrid` | Hybrid approach: first finds the initial pointer `j` using **binary search** on `timestamps2`, then slides `j` forward like the O(n+m) algorithm. Also used for parallel implementation. | **O(log m + n)** |
| `match_timestamps_parallel` | Splits `timestamps1` into chunks, then runs `match_timestamps_hybrid` on each chunk in parallel using `multiprocessing.Pool`. Results are concatenated at the end. | **O(n/m + log m)** per process, + barriers overhead for parallelism |
| `match_timestamps_numba_accelerated` | Numba JIT-compiled version of `match_timestamps`. Executes the same O(n+m) algorithm, but compiled to machine code for huge speedups. | **O(n + m)**, runtime drastically reduced due to JIT compilation |

## **Benchmarking Evaluation Method**

- `'avg'` → Average execution time over multiple iterations.  
- `'min'` → Minimum execution time over multiple iterations to reduce noise.  

Personally, I'd prefer to judge by minimum execution time, especially on multiprocess implementation, because some other processes can affect on average time badly, which is not a honest assesment.

## Average Time

## FPS = 30, Hours = 1
| Function                       | Avg Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.113610 |
| match_timestamps_hybrid        | 0.171854 |
| match_timestamps_parallel      | 0.111176 |
| match_timestamps_numba_accelerated | 0.002077 |

## FPS = 30, Hours = 2
| Function                       | Avg Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.229862 |
| match_timestamps_hybrid        | 0.365996 |
| match_timestamps_parallel      | 0.277647 |
| match_timestamps_numba_accelerated | 0.004206 |

## FPS = 30, Hours = 3
| Function                       | Avg Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.349771 |
| match_timestamps_hybrid        | 0.558975 |
| match_timestamps_parallel      | 0.342378 |
| match_timestamps_numba_accelerated | 0.004736 |

## FPS = 60, Hours = 1
| Function                       | Avg Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.238017 |
| match_timestamps_hybrid        | 0.337967 |
| match_timestamps_parallel      | 0.243753 |
| match_timestamps_numba_accelerated | 0.003069 |

## FPS = 60, Hours = 2
| Function                       | Avg Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.466814 |
| match_timestamps_hybrid        | 0.689048 |
| match_timestamps_parallel      | 0.454537 |
| match_timestamps_numba_accelerated | 0.005783 |

## FPS = 60, Hours = 3
| Function                       | Avg Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.698220 |
| match_timestamps_hybrid        | 1.048730 |
| match_timestamps_parallel      | 0.635266 |
| match_timestamps_numba_accelerated | 0.009139 |

## FPS = 120, Hours = 1
| Function                       | Avg Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.471144 |
| match_timestamps_hybrid        | 0.677448 |
| match_timestamps_parallel      | 0.389900 |
| match_timestamps_numba_accelerated | 0.005278 |

## FPS = 120, Hours = 2
| Function                       | Avg Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.950431 |
| match_timestamps_hybrid        | 1.385051 |
| match_timestamps_parallel      | 0.934344 |
| match_timestamps_numba_accelerated | 0.010076 |

## FPS = 120, Hours = 3
| Function                       | Avg Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 1.443193 |
| match_timestamps_hybrid        | 2.101148 |
| match_timestamps_parallel      | 1.296713 |
| match_timestamps_numba_accelerated | 0.017783 |


## Minimum Time

## FPS = 30, Hours = 1
| Function                       | Min Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.104213 |
| match_timestamps_hybrid        | 0.138090 |
| match_timestamps_parallel      | 0.095883 |
| match_timestamps_numba_accelerated | 0.001711 |

## FPS = 30, Hours = 2
| Function                       | Min Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.212010 |
| match_timestamps_hybrid        | 0.313939 |
| match_timestamps_parallel      | 0.211498 |
| match_timestamps_numba_accelerated | 0.003395 |

## FPS = 30, Hours = 3
| Function                       | Min Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.323800 |
| match_timestamps_hybrid        | 0.473006 |
| match_timestamps_parallel      | 0.249343 |
| match_timestamps_numba_accelerated | 0.004536 |

## FPS = 60, Hours = 1
| Function                       | Min Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.214584 |
| match_timestamps_hybrid        | 0.309786 |
| match_timestamps_parallel      | 0.211477 |
| match_timestamps_numba_accelerated | 0.002870 |

## FPS = 60, Hours = 2
| Function                       | Min Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.428078 |
| match_timestamps_hybrid        | 0.633410 |
| match_timestamps_parallel      | 0.397203 |
| match_timestamps_numba_accelerated | 0.005158 |

## FPS = 60, Hours = 3
| Function                       | Min Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.650176 |
| match_timestamps_hybrid        | 0.965001 |
| match_timestamps_parallel      | 0.474472 |
| match_timestamps_numba_accelerated | 0.008779 |

## FPS = 120, Hours = 1
| Function                       | Min Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.423403 |
| match_timestamps_hybrid        | 0.632211 |
| match_timestamps_parallel      | 0.383940 |
| match_timestamps_numba_accelerated | 0.004981 |

## FPS = 120, Hours = 2
| Function                       | Min Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 0.855874 |
| match_timestamps_hybrid        | 1.294776 |
| match_timestamps_parallel      | 0.777571 |
| match_timestamps_numba_accelerated | 0.010148 |

## FPS = 120, Hours = 3
| Function                       | Min Time (s) |
|--------------------------------|--------------|
| match_timestamps               | 1.403641 |
| match_timestamps_hybrid        | 2.042071 |
| match_timestamps_parallel      | 1.165164 |
| match_timestamps_numba_accelerated | 0.014869 |

## **Key Takeaways**

- The basic sequential implementation is simple and memory-efficient but slower for large datasets.
- For real-time applications, **Numba-accelerated** matching is the fastest and most efficient.  
- The hybrid approach can help in specific edge cases but may not always justify its complexity.  
- For multi-core systems without Numba, **parallel processing** offers moderate speedups and superior in performance to a more efficient algorithm than the core (single threaded hybrid) algorithm of that implementation.  

---
---

# Task 2: Median Streaming (Sliding Window)

## Function Descriptions

| Function | Description | Complexity |
|----------|-------------|------------|
| `SlidingMedian` | Custom class to maintain a dynamic median using two heaps (max-heap for lower half, min-heap for upper half). Supports `add`, `remove`, and `median` operations efficiently. | `O(log k)` per add/remove, where `k` is window size |
| `median_sliding_window` | Sliding window median using `SlidingMedian` class. Adds new elements and removes old elements as the window slides. | `O(n log k)` total, `n = len(nums)` |
| `median_sliding_window_naive` | Naive implementation: sorts each window independently to compute the median. | `O(n k log k)` total |

---

## Notes on Implementation

- `SlidingMedian` uses delayed removals with a dictionary (`self.delayed`) to avoid costly heap deletions.
- For even window sizes, the median is computed as the average of the two middle elements.
- This algorithm supports both odd and even `k` efficiently.

---

## Benchmark Table (Minimum Time)

| n        | k   | Fast min (ms) | Slow min (ms) | Speedup (x) |
|----------|-----|---------------|---------------|-------------|
| 1,000    | 10  | 3.9283        | 3.0586        | 0.78×       |
| 1,000    | 100 | 3.8595        | 3.7404        | 0.97×       |
| 1,000    | 500 | 2.8011        | 4.4010        | 1.57×       |
| 10,000   | 10  | 49.3253       | 36.0850       | 0.73×       |
| 10,000   | 100 | 44.6906       | 40.4060       | 0.90×       |
| 10,000   | 500 | 42.1520       | 76.3250       | 1.81×       |
| 100,000  | 10  | 498.8611      | 338.4201      | 0.68×       |
| 100,000  | 100 | 472.6670      | 410.2786      | 0.87×       |
| 100,000  | 500 | 454.1813      | 780.1417      | 1.72×       |
| 1,000,000| 10  | 5489.6615     | 3645.0457     | 0.66×       |
| 1,000,000| 100 | 5420.5026     | 4311.0957     | 0.80×       |
| 1,000,000| 500 | 5212.4047     | 8458.4557     | 1.62×       |


> **Note:** Times are minimum over several iterations. Values in milliseconds.

---

## **Key Takeaways**

- For **small `k`**, the naive approach can be faster due to lower overhead.
- For **larger `k`**, `SlidingMedian` is clearly superior and scales much better.
