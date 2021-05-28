from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Gurobi.Model.run_model import run_model
from Heuristics.ALNS.alns_no_local import ALNS
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
	alns_stochastic.run(process_num)

	print(f"\n############## ALNS - Deterministic process {process_num} ##############")
	filename_list = filename.split("-")
	filename_list[1] = '1'
	deterministic_filename = "-".join(filename_list)
	alns_deterministic = ALNS(deterministic_filename + ".pkl")
	alns_deterministic.run(process_num)

	print(f"\n############## RP process {process_num} ##############")
	rp = GurobiInstance(filename + ".yaml", solution_dict=alns_stochastic.best_solution[0], first_stage_only=True, optimize=True)
	run_model(rp)

	print(f"\n############## EEV process {process_num} ##############")
	rp = GurobiInstance(filename + ".yaml", solution_dict=alns_deterministic.best_solution[0], first_stage_only=True, optimize=True)
	run_model(rp)


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
	files = [["./InstanceGenerator/InstanceFiles/6nodes/6-25-2-1_a", "./InstanceGenerator/InstanceFiles/6nodes/6-25-2-1_b"],
	        ["./InstanceGenerator/InstanceFiles/6nodes/6-25-2-1_c", "./InstanceGenerator/InstanceFiles/8nodes/8-25-2-1_a"],
	        ["./InstanceGenerator/InstanceFiles/8nodes/8-25-2-1_b", "./InstanceGenerator/InstanceFiles/8nodes/8-25-2-1_c"],
	        ["./InstanceGenerator/InstanceFiles/10nodes/10-25-2-1_a", "./InstanceGenerator/InstanceFiles/10nodes/10-25-2-1_b"],
	        ["./InstanceGenerator/InstanceFiles/10nodes/10-25-2-1_c", "./InstanceGenerator/InstanceFiles/15nodes/15-25-2-1_a"]]

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
