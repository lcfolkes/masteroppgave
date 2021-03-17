from abc import ABC, abstractmethod
import os
import random
import copy

print(os.getcwd())
os.chdir('../../InstanceGenerator')
from Heuristics.construction_heuristic import ConstructionHeuristic


class Destroy(ABC):
	def __init__(self, solution, num_first_stage_tasks, neighborhood_size):
		'''
		:param solution: (s) assigned car_moves of constructed solution. solution[(k,s)], dictionary containing car_move assigned to employee in scenario s
		:param neighborhood_size: (q) number of car_moves to remove
		'''
		self.num_scenarios = len(list(solution.items())[0][1])
		self.num_first_stage_tasks = num_first_stage_tasks

		self.removed_moves = []
		self.input_solution = solution
		self.first_stage_solution = self._get_first_stage_solution()
		self.neighborhood_size = neighborhood_size
		self.destroyed_solution = self._destroy()

	@abstractmethod
	def _destroy(self):
		pass

	def _get_first_stage_solution(self):
		removed_second_stage_moves = set()
		first_stage_solution = {}
		for k, v in self.input_solution.items():
			first_stage_solution[k] = set()
			for s in range(len(self.input_solution[k])):
				for i in range(self.num_first_stage_tasks):
					first_stage_solution[k].add(self.input_solution[k][s][i])
				for i in range(self.num_first_stage_tasks, len(self.input_solution[k][s])):
					removed_second_stage_moves.add(self.input_solution[k][s][i])
			first_stage_solution[k] = list(first_stage_solution[k])

		self.removed_moves = list(removed_second_stage_moves)

		print(self.input_solution)
		print(self.removed_moves)
		print(first_stage_solution)
		return first_stage_solution


	def to_string(self):
		print("\nDESTROY")
		print("input solution")
		for k, v in self.input_solution.items():
			print(k)
			for s in v:
				print([cm.car_move_id for cm in s])

		print("first stage solution")
		for k, v in self.first_stage_solution.items():
			print(k)
			print([cm.car_move_id for cm in v])

		print("destroyed solution")
		for k, v in self.destroyed_solution.items():
			print(k)
			print([cm.car_move_id for cm in v])


class RandomRemoval(Destroy):

	def __init__(self, solution, num_first_stage_tasks, neighborhood_size):
		super().__init__(solution, num_first_stage_tasks, neighborhood_size)

	def _destroy(self):

		solution = copy.deepcopy(self.first_stage_solution)

		n_size = self.neighborhood_size
		while n_size > 0:
			k = random.choice(list(solution.keys()))
			# ensures list of chosen key is not empty
			if not any(solution[k]):
				continue
			i = random.randrange(0, len(solution[k]), 1)
			self.removed_moves.append(solution[k][i])
			solution[k] = solution[k][:i] + solution[k][i + 1:]
			n_size -= 1
		print(self.removed_moves)

		return solution

class WorstRemoval(Destroy):
	def __init__(self, solution, num_first_stage_tasks, neighborhood_size, randomization_degree):
		'''
		Worst removal removes solutions that have a bad influence on the objective value. In this case, that means moves where the objective function decreases little when they are removed.
		:param randomization_degree: (p) parameter that determines the degree of randomization
		'''
		super().__init__(solution, num_first_stage_tasks, neighborhood_size)

		self.randomization_degree = randomization_degree






class ShawRemoval(Destroy):

	def __init__(self, solution, num_first_stage_tasks, neighborhood_size, randomization_degree):
		'''
		Shaw removal removes car-moves that are somewhat similar to each other. It takes in a solution, the number of car moves to remove, and a randomness parameter p >= 1.
		:param randomization_degree: (p) parameter that determines the degree of randomization
		'''
		super().__init__(solution, num_first_stage_tasks, neighborhood_size)
		'''
		:param randomization_degree: (p) parameter that determines the degree of randomization 
		'''
		self.randomization_degree = randomization_degree

	# The relatedness measure can e.g. measure how close the the start nodes are to each other, and the same for the
	# end nodes, when the car moves are started, and maybe whether only a few employees are able to perform both
	# requests. It should also probably only compare parking moves with parking moves and charging moves with
	# charging moves.
	def _relatednes_measure(self, car_move_i, car_move_j):
		pass

	def _destroy(self):
		pass



if __name__ == "__main__":
	print("\n---- HEURISTIC ----")
	ch = ConstructionHeuristic("InstanceFiles/6nodes/6-3-1-1_a.pkl")
	ch.add_car_moves_to_employees()
	ch.print_solution()
	ch.get_objective_function_val()
	rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
					   neighborhood_size=1)
	rr.to_string()
