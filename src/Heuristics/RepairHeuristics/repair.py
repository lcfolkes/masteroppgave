from abc import ABC, abstractmethod
import os
import random
import copy
from Heuristics.DestroyHeuristics.destroy import RandomRemoval
print(os.getcwd())
os.chdir('../../InstanceGenerator')
from Heuristics.construction_heuristic import ConstructionHeuristic




class Repair(ABC):
	def __init__(self, destroyed_solution, unused_car_moves, num_first_stage_tasks, neighborhood_size):
		'''
		:param destroyed_solution: dictionary of destroyed solution returned from a Destroy heuristic
		:param unused_car_moves: list of unused car_moves for each scenario e.g.: [[], [], []]
		:param num_first_stage_tasks: int
		:param neighborhood_size: int
		'''

		self.destroyed_solution = destroyed_solution
		self.unused_car_moves = unused_car_moves
		self.num_first_stage_tasks = num_first_stage_tasks
		self.neighborhood_size = neighborhood_size

	@abstractmethod
	def _repair(self):
		pass

	def _get_best_insertion(self, regret):
		

class GreedyInsertion(Repair):
	def __init__(self, destroyed_solution, unused_car_moves, num_first_stage_tasks, neighborhood_size):
		super().__init__(destroyed_solution, unused_car_moves, num_first_stage_tasks, neighborhood_size)

	def _repair(self):
		q = self.neighborhood_size
		current_solution = self.destroyed_solution
		while q > 0:
			best_car_move, best_employee, best_index = self._get_best_insertion(regret=1)
			current_solution = insert_car_move(current_solution, best_car_move, best_employee, best_index)
			q -= 1


class RegretInsertion(Repair):
	def __init__(self, destroyed_solution, unused_car_moves, num_first_stage_tasks, neighborhood_size):
		super().__init__(self, destroyed_solution, unused_car_moves, num_first_stage_tasks, neighborhood_size)





if __name__ == "__main__":
	print("\n---- HEURISTIC ----")
	ch = ConstructionHeuristic("InstanceFiles/6nodes/6-3-1-1_a.pkl")
	ch.add_car_moves_to_employees()
	ch.print_solution()
	ch.get_objective_function_val()
	rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
					   neighborhood_size=2)
	rr.to_string()

	gi = GreedyInsertion(destroyed_solution=rr.destroyed_solution, unused_car_moves=ch.unused_car_moves,
						 num_first_stage_tasks=ch.world_instance.first_stage_tasks, neighborhood_size=2)