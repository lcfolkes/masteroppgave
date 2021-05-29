from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Gurobi.Model.run_model import run_model
from Heuristics.ALNS.alns_no_local import ALNS
from Heuristics.helper_functions_heuristics import get_first_stage_solution
from path_manager import path_to_src
import os
import time
import sys

os.chdir(path_to_src)
import multiprocessing as mp

def run_parallel(filenames, n):
	num_processes = n * len(filenames)
	args = []
	for filename in filenames:
		for i in range(n):
			args.append((filename + ".pkl", i + 1))
	with mp.Pool(processes=num_processes) as pool:
		pool.starmap(run_process, args)

def run_process(filename, process_num, param=None):
	alns = ALNS(filename, param)
	alns.run(process_num)

def run_vss_parallel(filenames, n):
	num_processes = n * len(filenames)
	args = []
	for filename in filenames:
		for i in range(n):
			args.append((filename, i + 1))
	with mp.Pool(processes=num_processes) as pool:
		pool.starmap(run_vss_process, args)

def run_vss_process(filename, process_num):
	print(f"\n############## ALNS - Stochastic process {process_num} ##############")
	alns_stochastic = ALNS(filename + ".pkl")
	alns_stochastic.run(f"{process_num}\nProblem type: RP")

	print(f"\n############## ALNS - Deterministic process {process_num} ##############")
	filename_list = filename.split("-")
	filename_list[1] = '1'
	deterministic_filename = "-".join(filename_list)
	alns_deterministic = ALNS(deterministic_filename + ".pkl")
	alns_deterministic.run(f"{process_num}\nProblem type: Deterministic")

	print(f"\n############## ALNS - EEV process {process_num} ##############")
	alns_stochastic.solution.rebuild(get_first_stage_solution(
		alns_deterministic.best_solution[0], alns_stochastic.solution.num_first_stage_tasks), stage="first")
	results_str = f"{process_num}\nProblem type: EEV \n"
	results_str += f"Objective value: {str(alns_stochastic.solution.get_obj_val(true_objective=True, both=False))}\n"
	results_str += f"Cars charged: {str(alns_stochastic.solution.num_charging_moves)}\n" \
				   f"Cars in need of charging: {str(alns_stochastic.solution.num_cars_in_need)}\n\n"
	test_dir = f"./Testing/Results/" + filename.split('/')[-2] + "/"
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)
	filepath = test_dir + filename.split('/')[-1].split('.')[0]
	with open(filepath + "_results.txt", "a") as f:
		f.write(results_str)

	print(f"\n############## GUROBI - RP process {process_num} ##############")
	rp = GurobiInstance(filename + ".yaml", solution_dict=alns_stochastic.best_solution[0], first_stage_only=True, optimize=True)
	run_model(rp, mode="_rp", run=process_num)

	print(f"\n############## GUROBI - EEV process {process_num} ##############")
	eev = GurobiInstance(filename + ".yaml", solution_dict=alns_deterministic.best_solution[0], first_stage_only=True, optimize=True)
	run_model(eev, mode="_eev", run=process_num)


def run_gurobi_parallel(filenames):
	num_processes = len(filenames)
	with mp.Pool(processes=num_processes) as pool:
		pool.map(run_gurobi_process, filenames)


def run_gurobi_process(filename):
	gi = GurobiInstance(filename + ".yaml")
	run_model(gi, time_limit=3600)


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
	for n in [6, 8, 10]:  # , 25, 30, 40, 50]:
		directory = f"./InstanceGenerator/InstanceFiles/{n}nodes/"
		for filename in os.listdir(directory):
			filename_list = filename.split(".")
			if filename_list[-1] == "pkl":
				files.append(os.path.join(directory, filename_list[0]))

	# for f in files:
	#    print(f)
	files = [["./InstanceGenerator/InstanceFiles/15nodes/15-25-2-1_b", "./InstanceGenerator/InstanceFiles/15nodes/15-25-2-1_c"],
		["./InstanceGenerator/InstanceFiles/20nodes/20-25-2-1_a", "./InstanceGenerator/InstanceFiles/20nodes/20-25-2-1_b"],
		["./InstanceGenerator/InstanceFiles/20nodes/20-25-2-1_c", "./InstanceGenerator/InstanceFiles/25nodes/25-25-2-1_a"],
		["./InstanceGenerator/InstanceFiles/25nodes/25-25-2-1_b", "./InstanceGenerator/InstanceFiles/25nodes/25-25-2-1_c"]]

	try:
		n = 10
		for filenames in files:
			### PARALLEL
			run_vss_parallel(filenames, n)

	except KeyboardInterrupt:
		print('Interrupted')
		try:
			sys.exit(0)
		except SystemExit:
			os._exit(0)
