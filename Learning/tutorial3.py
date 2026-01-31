"""
memory_detective.py - A program that performs various operations
Your task: Add memory tracking to answer the questions below!
"""

import numpy as np
import time
import psutil
import tracemalloc

process = psutil.Process()
tracemalloc.start()

def create_small_list():
    """Creates a small list of integers"""
    return [i for i in range(1000)]

def create_large_list():
    """Creates a large list of integers"""
    return [i for i in range(10_000_000)]

def create_numpy_array():
    """Creates a large numpy array"""
    return np.random.randn(5000, 5000)

def do_matrix_multiplication():
    """Performs matrix multiplication"""
    A = np.random.randn(1000, 1000)
    B = np.random.randn(1000, 1000)
    C = A @ B
    return C

def create_nested_structure():
    """Creates a nested dictionary structure"""
    data = {}
    for i in range(1000):
        data[f"key_{i}"] = {
            "values": [j for j in range(100)],
            "metadata": {"id": i, "name": f"item_{i}"}
        }
    return data

def main():
    """Main function - performs various operations"""
    
    print("Starting program...")
    time.sleep(0.5)
    mem_dict = {}
    # Operation 1: Small list
    mem_bef = process.memory_info().rss
    print("\n[Operation 1] Creating small list...")
    small_list = create_small_list()
    time.sleep(0.5)
    mem_aft = process.memory_info().rss
    mem_dict['Process 1'] = mem_aft - mem_bef
    
    # Operation 2: Large list
    mem_bef = process.memory_info().rss
    mem_before_by_traced_alloc = tracemalloc.get_traced_memory()[0]
    print("[Operation 2] Creating large list...")
    large_list = create_large_list()
    time.sleep(0.5)
    mem_aft = process.memory_info().rss
    mem_aft_by_traced_alloc = tracemalloc.get_traced_memory()[0]
    mem_dict['Process 2'] = mem_aft - mem_bef
    print("Process 2 memory by psutil = ", mem_dict['Process 2']/1024/1024)
    print("Process 2 memory by tracemalloc = ", (mem_aft_by_traced_alloc - mem_before_by_traced_alloc)/1024/1024)
    
    # Operation 3: Numpy array
    mem_bef = process.memory_info().rss
    print("[Operation 3] Creating numpy array...")
    array = create_numpy_array()
    time.sleep(0.5)
    mem_aft = process.memory_info().rss
    mem_dict['Process 3'] = mem_aft - mem_bef
    
    # Operation 4: Matrix multiplication
    mem_bef = process.memory_info().rss
    print("[Operation 4] Doing matrix multiplication...")
    result = do_matrix_multiplication()
    time.sleep(0.5)
    mem_aft = process.memory_info().rss
    mem_dict['Process 4'] = mem_aft - mem_bef

    # Operation 5: Nested structure
    mem_bef = process.memory_info().rss
    mem_before_by_traced_alloc = tracemalloc.get_traced_memory()[0]
    print("[Operation 5] Creating nested structure...")
    nested = create_nested_structure()
    time.sleep(0.5)
    mem_aft = process.memory_info().rss
    mem_aft_by_traced_alloc = tracemalloc.get_traced_memory()[0]
    mem_dict['Process 5'] = mem_aft - mem_bef
    # print(mem_dict['Process 5'])
    print("Process 5 memory by psutil = ", mem_dict['Process 5']/1024/1024)
    print("Process 5 memory by tracemalloc = ", (mem_aft_by_traced_alloc - mem_before_by_traced_alloc)/1024/1024)
    print("\nProgram complete!")
    
    # Keep everything in memory
    return small_list, large_list, array, result, nested, mem_dict

if __name__ == "__main__":
    
    mem_before = process.memory_info().rss / 1024 / 1024
    data = main()
    mem_after = process.memory_info().rss /1024 / 1024
    current_mem = tracemalloc.get_traced_memory()[0]
    peak_mem = tracemalloc.get_traced_memory()[1]
    print("Memory used by process = ", mem_after - mem_before)
    print(f"Current memory used by process = {current_mem/1024/1024} and peak memory = {peak_mem/1024/1024}")
    print("Process wise memory information : ")
    # print(type(data[-1]))
    for k in data[-1]:
        print(k, " : ", data[-1][k]/1024/1024)
