from abc import ABC, abstractmethod
import os
import random
import copy

#print(os.getcwd())
#os.chdir('../../InstanceGenerator')
#from Heuristics.construction_heuristic import ConstructionHeuristic


class Destroy(ABC):
	def __init__(self, solution, num_first_stage_tasks, neighborhood_size):
		'''
		:param solution: (s) assigned car_moves of constructed solution. solution[(k,s)], dictionary containing car_move assigned to employee in scenario s
		:param neighborhood_size: (n) number of car_moves to remove
		'''
		self.input_solution = solution
		self.num_first_stage_tasks = num_first_stage_tasks
		self.destroyed_solution = self._destroy(neighborhood_size)

	@abstractmethod
	def _destroy(self, solution):
		pass


	def to_string(self):
		print("\nDESTROY")
		print("input solution")
		for k, v in self.input_solution.items():
			print(k)
			for s in v:
				print([cm.car_move_id for cm in s])

		print("destroyed solution")
		for k, v in self.destroyed_solution.items():
			print(k)
			for s in v:
				print([cm.car_move_id for cm in s])


class RandomRemoval(Destroy):

	def __init__(self, solution, num_first_stage_tasks, neighborhood_size):
		super().__init__(solution, num_first_stage_tasks, neighborhood_size)

	def _destroy(self, neighborhood_size):
		solution = copy.deepcopy(self.input_solution)
		n_size = neighborhood_size
		while n_size > 0:
			k = random.choice(list(solution.keys()))

			# ensures list of chosen key is not empty
			if not any(solution[k]):
				continue
			i = random.randrange(0, len(solution[k]), 1)

			if i < self.num_first_stage_tasks:
				num_scenarios = len(solution[k])
				for s in range(num_scenarios):
					car_moves = solution[k][s]
					solution[k][s] = car_moves[:i] + car_moves[i+1:]

			else:
				s = random.randrange(0, len(solution[k]), 1)
				car_moves = solution[k][s]
				solution[k][s] = car_moves[:i] + car_moves[i+1:]

			n_size -= 1

		return solution

class WorstRemoval(Destroy):
	def __init__(self, solution, num_first_stage_tasks, neighborhood_size, randomization_degree):
		'''
		:param randomization_degree: (p) parameter that determines the degree of randomization
		'''
		super().__init__(solution, num_first_stage_tasks, neighborhood_size)

		self.randomization_degree = randomization_degree


class ShawRemoval(Destroy):
	def __init__(self, solution, num_first_stage_tasks, neighborhood_size, randomization_degree):
		super().__init__(solution, num_first_stage_tasks, neighborhood_size)
		'''
		:param randomization_degree: (p) parameter that determines the degree of randomization 
		'''
		self.randomization_degree = randomization_degree

	def _relatednes_measure(self, car_move_i, car_move_j):
		pass


if __name__ == "__main__":
	print("\n---- HEURISTIC ----")
	ch = ConstructionHeuristic("InstanceFiles/6nodes/6-3-1-1_a.pkl")
	ch.add_car_moves_to_employees()
	ch.print_solution()
	ch.get_objective_function_val()
	rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
					   neighborhood_size=2)
	rr.to_string()
