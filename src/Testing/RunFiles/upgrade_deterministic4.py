from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Gurobi.Model.run_model import run_model
from Heuristics.ALNS.alns_deterministic_upgrade import ALNS
from Heuristics.ALNS.construction_heuristic import ConstructionHeuristic
from Heuristics.helper_functions_heuristics import get_first_stage_solution
from path_manager import path_to_src
import os
import time
import sys

os.chdir(path_to_src)
import multiprocessing as mp


def run_upgrade_parallel(filenames, n):
	num_processes = n * len(filenames)
	args = []
	for filename in filenames:
		for i in range(n):
			args.append((filename, i + 1))
	with mp.Pool(processes=num_processes) as pool:
		pool.starmap(run_upgrade_process, args)


def run_upgrade_process(filename, process_num):
	print(f"\n############## ALNS - Stochastic process {process_num} ##############")
	alns = ALNS(filename + ".pkl")
	alns.run(f"Run: {process_num}\nProblem type: Stochastic")

	print(f"\n############## ALNS - Deterministic process {process_num} ##############")
	filename_list = filename.split("-")
	filename_list[1] = '1'
	deterministic_filename = "-".join(filename_list)
	alns_deterministic = ALNS(deterministic_filename + ".pkl")
	alns_deterministic.run(f"Run: {process_num}\nProblem type: Deterministic")

	print(f"\n############## ALNS - EEV process {process_num} ##############")
	stochastic_ch = ConstructionHeuristic(filename + ".pkl")
	start_ch = time.perf_counter()
	stochastic_ch.rebuild(get_first_stage_solution(
		alns_deterministic.best_solution[0], stochastic_ch.num_first_stage_tasks), stage="first")
	time_ch = time.perf_counter() - start_ch

	time_used = alns_deterministic.finish_time + time_ch
	print(f"\n############## ALNS - Upgrade process {process_num} ##############")
	alns_stochastic = ALNS(filename + ".pkl", construction_heuristic=stochastic_ch, start_time=time_used)
	alns_stochastic.run(f"Run: {process_num}\nProblem type: Upgrade")


if __name__ == "__main__":
	files = []
	for n in [40]:
		directory = f"./InstanceGenerator/InstanceFiles/{n}nodes/"
		instance_type_list = []
		for filename in os.listdir(directory):
			filename_list = filename.split(".")
			if filename_list[-1] == "pkl":
				instance_type_list.append(os.path.join(directory, filename_list[0]))
		files.append(instance_type_list)

	try:
		n = 10
		for filenames in files:
			### PARALLEL
			run_upgrade_parallel(filenames, n)

	except KeyboardInterrupt:
		print('Interrupted')
		try:
			sys.exit(0)
		except SystemExit:
			os._exit(0)
