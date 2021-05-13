from Heuristics.new_alns_local_common_weights_separate_charg import ALNS
from path_manager import path_to_src
from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Gurobi.Model.run_model import run_model
import os
import time
import sys
os.chdir(path_to_src)
import multiprocessing as mp

def parallel_runs(filename, n):
    alns = ALNS(filename + ".pkl", acceptance_percentage=1)
    with mp.Pool(processes=n) as pool:
        args = [False for _ in range(n)]
        obj_vals = pool.map(alns.run, args)
    return obj_vals
    #obj_vals = alns.run(verbose=False)
    #return obj_vals

def run_parallell(filename, n):
    print(f"Running {n} processes in parallel\n")
    start = time.perf_counter()
    obj_vals = parallel_runs(filename, n)
    finish = time.perf_counter()
    print(f"Parallel: Finished {n} runs in {round(finish - start, 2)} seconds(s)")
    print(obj_vals)

def run_sequential(filename, n):
    print(f"Running {n} processes in sequence\n")
    start = time.perf_counter()
    obj_vals = []
    for _ in range(n):
        alns = ALNS(filename + ".pkl", acceptance_percentage=1)
        obj_val = alns.run(False)
        print(obj_val)
        obj_vals.append(obj_val)
    finish = time.perf_counter()
    print(f"Sequential: Finished {n} runs in {round(finish - start, 2)} seconds(s)")
    print(obj_vals)


if __name__ == "__main__":
    try:
        filename = "InstanceGenerator/InstanceFiles/30nodes/30-10-2-1_a"
        n = 10

        ### PARALLEL
        #run_parallell(filename, n)

        ### SEQUENTIAL
        run_sequential(filename, n)

    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
