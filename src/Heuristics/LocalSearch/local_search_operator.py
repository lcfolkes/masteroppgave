from abc import ABC, abstractmethod

import random
import copy
from Heuristics.DestroyAndRepairHeuristics.destroy import RandomRemoval
from Heuristics.DestroyAndRepairHeuristics.repair import GreedyInsertion
from Heuristics.construction_heuristic import ConstructionHeuristic
from Heuristics.feasibility_checker import FeasibilityChecker
from Heuristics.helper_functions_heuristics import get_first_and_second_stage_solution, reconstruct_solution_from_first_and_second_stage


class LocalSearchOperator(ABC):

	def __init__(self, solution, first_stage_tasks):
		'''
		:param solution: dictionary with employee object as key and a list of car moves as value
		'''
		self._current_solution = solution
		self._candidate_solution = None
		self.first_stage_tasks = first_stage_tasks
	'''
	def mutate(self, solution, employee, i, j, s):
		pass
	'''
	@property
	def current_solution(self):
		return self._current_solution

	@current_solution.setter
	def current_solution(self, value):
		self._current_solution = value

	@property
	def candidate_solution(self):
		return self._candidate_solution

	'''
	@candidate_solution.setter
	def candidate_solution(self, value):
		self._current_solution = value
	'''

	@property
	def hash_key(self):
		hash_dict = {}
		for k, v in self._candidate_solution.items():
			emp_moves = []
			for s in v:
				for cm in s:
					emp_moves.append(cm.car_move_id)
			hash_dict[k.employee_id] = emp_moves

		return hash(str(hash_dict))

	def to_string(self):
		print("Current solution")
		out_str = ""
		for k, v in self._current_solution.items():
			out_str += f"Emp {k.employee_id}: ("
			for s in v:
				out_str += f"{[cm.car_move_id for cm in s]}, "
			out_str = out_str[:-2]
			out_str += ")\n"
		print(out_str)
		try:
			print("Candidate solution")
			out_str = ""
			for k, v in self._candidate_solution.items():
				out_str += f"Emp {k.employee_id}: ("
				for s in v:
					out_str += f"{[cm.car_move_id for cm in s]}, "
				out_str = out_str[:-2]
				out_str += ")\n"
			print(out_str)
		except:
			pass
class IntraMove(LocalSearchOperator):
	'''
    An Intra Move LSO moves a car-move within the list of car-moves for one service employee
    '''
	# first or second stage:
	# acceptance: first best, best, simulated annealing

	def __init__(self, solution, first_stage_tasks):
		super().__init__(solution, first_stage_tasks)

	def mutate(self, employee, i, j, s=None):
		# No need to copy because get_first_and_second_stage_solution() creates new dict with same elements
		first_stage, second_stage = get_first_and_second_stage_solution(self._current_solution, self.first_stage_tasks)

		if s is None:
			moves = first_stage[employee]
		else:
			moves = second_stage[employee][s]

		moves[i], moves[j] = moves[j], moves[i]

		self._candidate_solution = reconstruct_solution_from_first_and_second_stage(first_stage, second_stage)



class InterSwap(LocalSearchOperator):
	'''
    An Inter Swap LSO swaps two car-moves between two service employees.
    '''

	def __init__(self, solution, first_stage_tasks):
		super().__init__(solution, first_stage_tasks)

	def mutate(self, emp1, emp2, s=None):
		e1, idx_1 = emp1
		e2, idx_2 = emp2
		first_stage, second_stage = get_first_and_second_stage_solution(self._current_solution, self.first_stage_tasks)
		if s is None:
			first_stage[e2][idx_2], first_stage[e1][idx_1] = first_stage[e1][idx_1], first_stage[e2][idx_2]
		else:
			first_stage[e2][s][idx_2], first_stage[e1][s][idx_1] = first_stage[e1][s][idx_1], first_stage[e2][s][idx_2]

		self._candidate_solution = reconstruct_solution_from_first_and_second_stage(first_stage, second_stage)


class InterMove(LocalSearchOperator):
	'''
    An Inter Move LSO moves a car-move from one service employee to another.
    '''

	def __init__(self, solution, feasibility_checker):
		super().__init__(solution, feasibility_checker)


class Inter2Move(LocalSearchOperator):
	'''
    An Inter 2-Move LSO moves two consecutive car-moves from one employee to another employee.
    '''

	def __init__(self, solution):
		super().__init__(solution)


class EjectionInsert(LocalSearchOperator):
	'''
    An Ejection Insert LSO picks a car-move from , and randomly inserts it into gamma
    '''

	def __init__(self, solution, unused_car_moves):
		self.unused_car_moves = unused_car_moves
		super().__init__(solution)


class EjectionRemove(LocalSearchOperator):
	'''
    An Ejection Remove LSO removes a car-move from gamma and place it into the pool of unused car-moves beta.
    '''

	def __init__(self, solution, unused_car_moves):
		self.unused_car_moves = unused_car_moves
		super().__init__(solution)


class EjectionReplace(LocalSearchOperator):
	'''
    An Ejection Replace LSO swaps a car-move from gamma with a car-move from beta associated with the same car.
    In other words, the destinations of a car-move in gamma is changed.
    '''

	def __init__(self, solution, unused_car_moves):
		self.unused_car_moves = unused_car_moves
		super().__init__(solution)


class EjectionSwap(LocalSearchOperator):
	'''
    An Ejection Swap LSO swaps a car-move from gamma with a car-move from beta, associated with
    different cars. An Ejection Swap differs from an Ejection Replace by swapping cars, while an
    Ejection Replace only replace a car-move with another car-move for the same car
    '''

	def __init__(self, solution, unused_car_moves):
		self.unused_car_moves = unused_car_moves
		super().__init__(solution)


if __name__ == "__main__":
	print("\n---- Local Search ----")
	filename = "InstanceGenerator/InstanceFiles/25nodes/25-2-2-1_a"
	ch = ConstructionHeuristic(filename + ".pkl")
	ch.add_car_moves_to_employees()
	#rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
	#				   neighborhood_size=2)
	# rr.to_string()
	#gi = GreedyInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
	#					 parking_nodes=ch.parking_nodes, world_instance=ch.world_instance)
	fc = FeasibilityChecker(ch.world_instance)
	print("Local Search")
	ls = IntraMove(ch.assigned_car_moves, ch.world_instance.first_stage_tasks)#, fc.feasibility_checker)
	emp = list(ch.assigned_car_moves.keys())
	ls.mutate(emp[0],0,1)
	ls.to_string()
	ls.current_solution = ls.candidate_solution
	ls.mutate(emp[1],0,1)
	ls.to_string()
	print(ls.candidate_solution is ls.current_solution)
