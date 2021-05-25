from Heuristics.ALNS.alns import ALNS
from path_manager import path_to_src
import os
import time
import sys
os.chdir(path_to_src)
import multiprocessing as mp

def run_parallel(filename, n, param=None):
    try:
        num_processes = n * len(param)
        params = len(param)
    except:
        num_processes = n
        params = 1
    args = []

    for x in range(params):
        for i in range(n):
            if param is not None:
                args.append((filename + ".pkl", i + 1, param[x]))
            else:
                args.append((filename + ".pkl", i + 1))
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(run_process, args)


def run_process(filename, process_num, param=None):
    alns = ALNS(filename, param)
    alns.run(process_num)

def run_sequential(filename, n, verbose):
    print(f"Running {n} processes in sequence\n")
    start = time.perf_counter()
    obj_vals = []
    for _ in range(n):
        alns = ALNS(filename + ".pkl", acceptance_percentage=1)
        obj_val = alns.run(verbose)
        print(obj_val)
        obj_vals.append(obj_val)
    finish = time.perf_counter()
    print(f"Sequential: Finished {n} runs in {round(finish - start, 2)} seconds(s)")
    print(obj_vals)
    return alns


if __name__ == "__main__":
    files = []
    for n in [6]: #, 25, 30, 40, 50]:
        directory = f"./InstanceGenerator/InstanceFiles/{n}nodes/"
        for filename in os.listdir(directory):
            filename_list = filename.split(".")
            if filename_list[-1] == "pkl":
                files.append(os.path.join(directory, filename_list[0]))

    #for f in files:
    #    print(f)
    try:
        n = 10
        for filename in files:
            ### PARALLEL
            run_parallel(filename, n)

        '''
        ### SEQUENTIAL
        #alns = run_sequential(filename, 1, True)
        
        print("\n############## Evaluate solution ##############")
        gi = GurobiInstance(filename + ".yaml", employees=alns.solution.employees, optimize=False)
        run_model(gi)

        print("\n############## Optimal solution ##############")
        gi2 = GurobiInstance(filename + ".yaml")
        run_model(gi2, time_limit=300)
        '''
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
