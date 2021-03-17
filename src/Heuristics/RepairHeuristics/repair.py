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





		

class GreedyInsertion(Repair):
	'''
	The greedy insertion heuristic greedily inserts car_moves yielding the greatest improvement to the
	objective function value (similar to the construction heuristic)
	'''
	def __init__(self, destroyed_solution, unused_car_moves, num_first_stage_tasks, neighborhood_size):

		super().__init__(destroyed_solution, unused_car_moves, num_first_stage_tasks, neighborhood_size)

	def _repair(self):
		q = self.neighborhood_size
		current_solution = self.destroyed_solution
		while q > 0:
			best_car_move, best_employee, best_index = self._get_best_insertion(current_solution, regret=1)
			current_solution = insert_car_move(current_solution, best_car_move, best_employee, best_index)
			q -= 1
		#print(destroyed_solution)

	def _get_best_insertion(self, current_solution):
		best_car_move = None
		best_employee = None
		best_index = None
		best_obj_val = -1000
		for car_move in self.unused_car_moves:
			for employee in self.destroyed_solution:
				if len(employee) == 0:
					obj_val = get_obj_value_first_stage(current_solution, car_move, employee, task_number=1)
					if obj_val > best_obj_val:
						best_car_move = car_move
						best_employee = employee
						best_index = 1

				elif len(employee) < self.num_first_stage_tasks:
					best_obj_val_car_move = -1000
					best_index_car_move = None
					for index in range(1, len(employee)+2):
						obj_val = get_obj_value_first_stage(current_solution, car_move, employee, task_nr=index)
						if obj_val > best_obj_val_car_move:
							best_obj_val_car_move = obj_val
							best_index_car_move = index
					if best_obj_val_car_move > best_obj_val:
						best_car_move = car_move
						best_employee = employee
						best_index = best_index_car_move
						best_obj_val = best_obj_val_car_move
		return best_car_move, best_employee, best_index


class RegretInsertion(Repair):
	'''
	The regret insertion heuristic considers the alternative costs of inserting a car_move into gamma (assigned_car_moves).
	The
	'''

	def _repair(self):
		pass

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