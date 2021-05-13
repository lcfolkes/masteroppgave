from Heuristics.new_alns_local_common_weights_separate_charg import ALNS
from path_manager import path_to_src
from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Gurobi.Model.run_model import run_model
import os
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

if __name__ == "__main__":

    filename = "InstanceGenerator/InstanceFiles/30nodes/30-10-2-1_a"

    try:
       obj_vals = parallel_runs(filename, 10)
       print(obj_vals)

    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
