from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Gurobi.Model.run_model import run_model
from Heuristics.ALNS.alns import ALNS
from path_manager import path_to_src
import os
import time
import sys

os.chdir(path_to_src)
import multiprocessing as mp


def run_parallel(filenames, n, params):
	num_processes = n * len(filenames)
	args = []
	for filename in filenames:
		for param in params:
			for i in range(n):
				args.append((filename + ".pkl", i + 1, param))
	with mp.Pool(processes=num_processes) as pool:
		pool.starmap(run_process, args)


def run_process(filename, process_num, param=None):
	alns = ALNS(filename, param)
	alns.run(f"Run: {process_num}")

if __name__ == "__main__":
	'''files = []
	for n in [6, 8, 10]:  # , 25, 30, 40, 50]:
		directory = f"./InstanceGenerator/InstanceFiles/{n}nodes/"
		for filename in os.listdir(directory):
			filename_list = filename.split(".")
			if filename_list[-1] == "pkl":
				files.append(os.path.join(directory, filename_list[0]))'''

	file = ["InstanceGenerator/InstanceFiles/50nodes/50-25-2-1_b"]

	try:
		n = 10
		params = [[[40, 8], [50, 8], [60, 8]],
				  [[70, 8], [10, 10], [20, 10]],
				  [[30, 10], [40, 10], [50, 10]]]
		for param in params:
			### PARALLEL
			run_parallel(file, n, param)

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
