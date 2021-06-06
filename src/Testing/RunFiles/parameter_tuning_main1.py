from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Gurobi.Model.run_model import run_model
from HelperFiles.helper_functions import read_config
from Heuristics.ALNS.alns import ALNS
from InstanceGenerator.instance_generator import build_world, create_instance_from_world
from path_manager import path_to_src
import os
import time
import sys
os.chdir(path_to_src)
import multiprocessing as mp

def run_parallel(filenames, n, params):
    #num_processes = n * len(filenames)
    args = []
    for filename in filenames:
        for param in params:
            for i in range(n):
                args.append((filename + ".pkl", i + 1, param))
    with mp.Pool(processes=len(args)) as pool:
        pool.starmap(run_process, args)

def run_process(filename, process_num, param=None):
    alns = ALNS(filename, param)
    alns.run(f"Run: {process_num}")

def build_world_process(cf, process_num):
    print("\nWELCOME TO THE EXAMPLE CREATOR \n")
    world = build_world(instance_config=cf)
    create_instance_from_world(world, num_scenarios=cf['num_scenarios'], num_tasks=cf['tasks']['num_all'],
                               num_first_stage_tasks=cf['tasks']['num_first_stage'], version=process_num,
                               time_of_day=cf['time_of_day'], planning_period=cf['planning_period'])


if __name__ == "__main__":
    '''    files = []
    for n in [6, 8, 10]: #, 25, 30, 40, 50]:
        directory = f"./InstanceGenerator/InstanceFiles/{n}nodes/"
        for filename in os.listdir(directory):
            filename_list = filename.split(".")
            if filename_list[-1] == "pkl":
                files.append(os.path.join(directory, filename_list[0]))'''



    file = ["InstanceGenerator/InstanceFilesCompStudy/50nodes/50-25-2-1_a"]

    try:
        #[[10, 0], [20, 0], [30, 0], [40, 0]],
        #[[50, 0], [60, 0], [70, 0],
        '''n = 10
        params = [[30, 45, 60], [75, 90]]
        for param in params:
            ### PARALLEL
            run_parallel(file, n, param)
        '''
        cf = read_config('./InstanceGenerator/InstanceConfigs/instance_config100.yaml')
        args = ((cf, 1), (cf, 2), (cf, 3))
        with mp.Pool(processes=3) as pool:
            pool.starmap(build_world_process, args)


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
