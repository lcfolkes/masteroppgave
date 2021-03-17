import os
from path_manager import path_to_src
from abc import ABC, abstractmethod
import random
import copy
from Heuristics.DestroyHeuristics.destroy import RandomRemoval
from Heuristics.helper_functions_heuristics import insert_car_move, get_obj_val_of_car_move
from Heuristics.construction_heuristic import ConstructionHeuristic
from InstanceGenerator.instance_components import CarMove, ParkingNode
os.chdir(path_to_src)
print(os.getcwd())





class Repair(ABC):
	def __init__(self, destroyed_solution: {int: [CarMove]}, unused_car_moves: [CarMove], num_first_stage_tasks: int, neighborhood_size: int, parking_nodes: [ParkingNode]) -> {int: [CarMove]}:
		'''
		:param destroyed_solution: dictionary of destroyed solution returned from a Destroy heuristic
		:param unused_car_moves: list of unused car_moves for each scenario e.g.: [[], [], []]
		:param num_first_stage_tasks: int
		:param neighborhood_size: int
		:param parking_nodes: list of ParkingNode
		:return A repaired solution in the form of a dictionary with key employee and value list of first-stage car moves
		'''
		self.destroyed_solution = destroyed_solution
		self.unused_car_moves = unused_car_moves
		self.num_first_stage_tasks = num_first_stage_tasks
		self.neighborhood_size = neighborhood_size
		self.parking_nodes = parking_nodes
		self.repaired_solution = self._repair()

	@abstractmethod
	def _repair(self):
		pass

	def to_string(self):
		print("\nREPAIR")
		print("destroyed solution")
		for k, v in self.destroyed_solution.items():
			print(k)
			print([cm.car_move_id for cm in v])

		print("repaired solution")
		for k, v in self.repaired_solution.items():
			print(k)
			print([cm.car_move_id for cm in v])


class GreedyInsertion(Repair):
	'''
	The greedy insertion heuristic greedily inserts car_moves yielding the greatest improvement to the
	objective function value (similar to the construction heuristic)
	'''
	def __init__(self, destroyed_solution, unused_car_moves, num_first_stage_tasks, neighborhood_size, parking_nodes):

		super().__init__(destroyed_solution, unused_car_moves, num_first_stage_tasks, neighborhood_size, parking_nodes)

	def _repair(self):
		q = self.neighborhood_size
		current_solution = copy.deepcopy(self.destroyed_solution)
		while q > 0:
			best_car_move, best_employee = self._get_best_insertion(current_solution)
			current_solution = insert_car_move(current_solution, best_car_move, best_employee)
			q -= 1
		self.repaired_solution = current_solution
		return current_solution

	def _get_best_insertion(self, current_solution):
		best_car_move = None
		best_employee = None
		best_obj_val = -1000
		for car_move in self.unused_car_moves:
			for employee in self.destroyed_solution.values():
				if len(employee) < self.num_first_stage_tasks:
					solution_with_move = insert_car_move(current_solution, car_move, employee)
					obj_val = get_obj_val_of_car_move(self.parking_nodes, num_scenarios=1, first_stage_car_moves=solution_with_move)
					if obj_val > best_obj_val:
						best_obj_val = obj_val
						best_car_move = car_move
						best_employee = employee
		return best_car_move, best_employee


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
	ch = ConstructionHeuristic("InstanceGenerator/InstanceFiles/6nodes/6-3-1-1_a.pkl")
	ch.add_car_moves_to_employees()
	ch.print_solution()
	ch.get_objective_function_val()
	rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
					   neighborhood_size=2)
	rr.to_string()
	gi = GreedyInsertion(destroyed_solution=rr.destroyed_solution, unused_car_moves=rr.removed_moves,
						 num_first_stage_tasks=ch.world_instance.first_stage_tasks, neighborhood_size=2,
						 parking_nodes=ch.parking_nodes)
	gi.to_string()