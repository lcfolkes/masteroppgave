from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Gurobi.Model.run_model import run_model
from HelperFiles.helper_functions import read_config
from Heuristics.ALNS.alns import ALNS
from path_manager import path_to_src
import os
import time
import sys

os.chdir(path_to_src)
import multiprocessing as mp


def run_parallel(filenames, n, params):
	# num_processes = n * len(filenames)
	args = []
	for filename in filenames:
		for p in params:
			for i in range(n):
				args.append((filename + ".pkl", i + 1, p))
	with mp.Pool(processes=len(args)) as pool:
		pool.starmap(run_process, args)


def run_process(filename, process_num, param=None):
	alns = ALNS(filename, param)
	alns.run(f"Run: {process_num}")


if __name__ == "__main__":
	'''    files = []
	for n in [6, 8, 10]: #, 25, 30, 40, 50]:
		directory = f"./InstanceGenerator/InstanceFiles/{n}nodes/"
		for filename in os.listdir(directory):
			filename_list = filename.split(".")
			if filename_list[-1] == "pkl":
				files.append(os.path.join(directory, filename_list[0]))'''

	files = [["InstanceGenerator/InstanceFiles/150nodes/150-25-2-1_a",
			  "InstanceGenerator/InstanceFiles/150nodes/150-25-2-1_b",
			  "InstanceGenerator/InstanceFiles/150nodes/150-25-2-1_c"]]

	try:
		# [[10, 0], [20, 0], [30, 0], [40, 0]],
		# [[50, 0], [60, 0], [70, 0],
		n = 1
		for file in files:
			### PARALLEL
			run_parallel(file, n, [0.5, 0.6, 0.7, 0.8, 0.9])

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
