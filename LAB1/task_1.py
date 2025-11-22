import time
import numpy as np
from multiprocessing import Pool
from numba import njit
from prettytable import PrettyTable

@njit
def match_timestamps_numba_accelerated(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    """
    Timestamp matching function. It returns such array `matching` of length len(timestamps1),
    that for each index i of timestamps1 the element matching[i] contains
    the index j of timestamps2, so that the difference between
    timestamps2[j] and timestamps1[i] is minimal.
    Example:
        timestamps1 = [0, 0.091, 0.5]
        timestamps2 = [0.001, 0.09, 0.12, 0.6]
        => matching = [0, 1, 3]
    """

    n1 = timestamps1.size
    n2 = timestamps2.size
    matching = np.empty(n1, dtype=np.uint32)
    j = 0

    for i in range(n1):
        t = timestamps1[i]
        while j + 1 < n2 and abs(timestamps2[j + 1] - t) <= abs(timestamps2[j] - t):
            j += 1
        matching[i] = j

    return matching

def match_timestamps(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    """
    Timestamp matching function. It returns such array `matching` of length len(timestamps1),
    that for each index i of timestamps1 the element matching[i] contains
    the index j of timestamps2, so that the difference between
    timestamps2[j] and timestamps1[i] is minimal.
    Example:
        timestamps1 = [0, 0.091, 0.5]
        timestamps2 = [0.001, 0.09, 0.12, 0.6]
        => matching = [0, 1, 3]
    """

    n1 = timestamps1.size
    n2 = timestamps2.size
    matching = np.empty(n1, dtype=np.uint32)
    j = 0

    for i in range(n1):
        t = timestamps1[i]
        while j + 1 < n2 and abs(timestamps2[j + 1] - t) <= abs(timestamps2[j] - t):
            j += 1
        matching[i] = j

    return matching

def match_timestamps_hybrid(ts1, ts2):
    """
    Hybrid algorithm:
      1. Binary search to get initial pointer j
      2. Then slide j forward just like the O(n+m) algorithm
    """

    n1 = ts1.size
    n2 = ts2.size

    out = np.empty(n1, dtype=np.uint32)

    # step 1: initial j via binary search
    t0 = ts1[0]
    j = np.searchsorted(ts2, t0)

    if j >= n2:
        j = n2 - 1
    elif j > 0:
        # pick nearest
        if abs(ts2[j] - t0) >= abs(ts2[j - 1] - t0):
            j -= 1

    # step 2: standard O(n+m) logic for this chunk
    out[0] = j

    for i in range(1, n1):
        t = ts1[i]
        while j + 1 < n2 and abs(ts2[j + 1] - t) <= abs(ts2[j] - t):
            j += 1
        out[i] = j

    return out

def chunk_array(arr, chunk_size):
    for i in range(0, len(arr), chunk_size):
        yield arr[i:i+chunk_size]

def _process_wrapper(args):
    ts1, ts2 = args
    # return match_timestamps_binary(ts1, ts2)
    return match_timestamps_hybrid(ts1, ts2)

def match_timestamps_parallel(ts1, ts2, nproc=8):
    chunk_size = (len(ts1) // nproc) + 1
    chunks = [ts1[i:i+chunk_size] for i in range(0, len(ts1), chunk_size)]
    args = [(c, ts2) for c in chunks]

    with Pool(nproc) as p:
        results = p.map(_process_wrapper, args)

    return np.concatenate(results)


def make_timestamps(fps: int, st_ts: float, fn_ts: float) -> np.ndarray:
    """
    Create array of timestamps. This array is discretized with fps,
    but not evenly.
    Timestamps are assumed sorted nad unique.
    Parameters:
    - fps: int
        Average frame per second
    - st_ts: float
        First timestamp in the sequence
    - fn_ts: float
        Last timestamp in the sequence
    Returns:
        np.ndarray: synthetic timestamps
    """
    # generate uniform timestamps
    timestamps = np.linspace(st_ts, fn_ts, int((fn_ts - st_ts) * fps))
    # add an fps noise
    timestamps += np.random.randn(len(timestamps))
    timestamps = np.unique(np.sort(timestamps))
    return timestamps


def perf_measurement(match_function: callable, fps: int = 30, num_hours : int = 2, n_iter: int = 25, eval_method='min'):
    """
    Performance measurement procedure
    """
    st_ts = time.time()
    fn_ts = st_ts + 3600 * num_hours
    ts1 = make_timestamps(fps, st_ts, fn_ts)
    ts2 = make_timestamps(fps, st_ts + 200, fn_ts)

    # warmup
    for _ in range(10):
        match_function(ts1, ts2)
    e_time = 0
    if eval_method == 'avg':
        t0 = time.perf_counter()
        for _ in range(n_iter):
            s_t = time.perf_counter()
            match_function(ts1, ts2)
        e_time = (time.perf_counter() - t0) / n_iter
        print(f"Perf time: {e_time} seconds")


    elif eval_method == 'min':
        times_list = []
        for _ in range(n_iter):
            s_t = time.perf_counter()
            match_function(ts1, ts2)
            r_t = time.perf_counter() - s_t
            times_list.append(r_t)
        e_time = min(times_list)
        print(f"Min perf time: {e_time} seconds")

    return e_time


def main():
    """
    Setup:
        Say we have two videocameras, each filming the same scene. We make
        a prediction based on this scene (e.g. detect a human pose).
        To improve the robustness of the detection algorithm,
        we average the predictions from both cameras at each moment.
        The camera data is a pair (frame, timestamp), where the timestamp
        represents the moment when the frame was captured by the camera.

    Problem:
        For each frame of camera1, we need to find the index of the
        corresponding frame received by camera2. The frame i from camera2
        corresponds to the frame j from camera1, if
        abs(timestamps[i] - timestamps[j]) is minimal for all i.

    Estimation criteria:
        - The solution has to be optimal algorithmically. As an example, let's assume that
    the best solution has O(n^3) complexity. In this case, the O(n^3 * logn) solution will add -1 point penalty,
    O(n^4) will add -2 points and so on.
        - The solution has to be optimal python-wise.
    If it can be optimized ~x5 times by rewriting the algorithm in Python,
    this will add -1 point. x20 times optimization will result in -2 points.
    You may use any optimization library!
        - All corner cases must be handled correctly. A wrong solution will add -3 points.
        - The base score is 6.
        - Parallel implementation adds +1 point, provided it is effective (cannot be optimized x5 times)
        - 3 points for this homework are added by completing the second problem (the one with the medians).
    Optimize the solution to work with ~2-3 hours of data.
    Good luck!
    """
    # # generate timestamps for the first camera
    # timestamps1 = make_timestamps(30, time.time() - 100, time.time() + 3600 * 2)
    # # generate timestamps for the second camera
    # timestamps2 = make_timestamps(60, time.time() + 200, time.time() + 3600 * 2.5)
    # matching = match_timestamps(timestamps1, timestamps2)

    table = PrettyTable()
    table.field_names = ["Function", "FPS", "Hours", "Evaluation", "Time (s)"]

    for match_function in (
        match_timestamps,
        match_timestamps_hybrid,
        match_timestamps_parallel,
        match_timestamps_numba_accelerated,
    ):
        for fps in (30, 60, 120):
            for num_hours in (1, 2, 3):
                for eval_method in ('avg', 'min'):
                    print(f'Run: {match_function.__name__}, {fps=}, {num_hours=}, {eval_method=}')
                    cur_time = perf_measurement(
                        match_function=match_function,
                        fps=fps,
                        num_hours=num_hours,
                        eval_method=eval_method
                    )

                    table.add_row([
                    match_function.__name__,
                    fps,
                    num_hours,
                    eval_method,
                    f"{cur_time:.6f}",
                    ])
    
    print(table)

if __name__ == '__main__':
    main()
