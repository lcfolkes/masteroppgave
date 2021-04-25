from abc import ABC, abstractmethod

from Heuristics.construction_heuristic import ConstructionHeuristic
from Heuristics.feasibility_checker import FeasibilityChecker
from Heuristics.helper_functions_heuristics import get_first_and_second_stage_solution, reconstruct_solution_from_first_and_second_stage, get_first_stage_solution
from Heuristics.objective_function import get_obj_val_of_solution_dict
import itertools


class LocalSearchOperator(ABC):

	def __init__(self, solution, first_stage_tasks, feasibility_checker):
		'''
		:param solution: dictionary with employee object as key and a list of car moves as value
		'''
		self._current_solution = solution
		self._candidate_solution = None
		self.first_stage_tasks = first_stage_tasks
		self.feasibility_checker = feasibility_checker
		self.visited_list = []


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

	def search(self, strategy, shuffle):
		pass
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
				scen_moves = []
				for cm in s:
					scen_moves.append(cm.car_move_id)
				emp_moves.append(scen_moves)
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

	def __init__(self, solution, first_stage_tasks, feasibility_checker):
		super().__init__(solution, first_stage_tasks, feasibility_checker)

	def _mutate(self, employee, i, j, s=None):
		# No need to copy because get_first_and_second_stage_solution() creates new dict with same elements
		first_stage, second_stage = get_first_and_second_stage_solution(self._current_solution, self.first_stage_tasks)

		if s is None:
			moves = first_stage[employee]
		else:
			moves = second_stage[employee][s]

		moves[i], moves[j] = moves[j], moves[i]

		self._candidate_solution = reconstruct_solution_from_first_and_second_stage(first_stage, second_stage)

	def search(self, strategy, shuffle=False):
		current_obj_val = get_obj_val_of_solution_dict(self._current_solution, self.feasibility_checker.world_instance, True)
		best_solution = self._current_solution

		search_space = self._get_search_space(self._current_solution)

		for k, i, j in search_space:
			self._mutate(k, i, j)
			self.visited_list.append(self.hash_key)
			if not self.feasibility_checker.is_solution_feasible(self._candidate_solution):
				continue
			#intra_move.to_string()
			candidate_obj_val = get_obj_val_of_solution_dict(self._candidate_solution, self.feasibility_checker.world_instance, True)
			#print(f"current_obj_val {current_obj_val}")
			#print(f"candidate_obj_val {candidate_obj_val}")
			if candidate_obj_val > current_obj_val:
				print("New best solution found!")
				self.to_string()
				current_obj_val = candidate_obj_val
				best_solution = self._candidate_solution
				if strategy == "best_first":
					break
		self._current_solution = best_solution
		return best_solution

	def _get_search_space(self, solution, second_stage=False):
		if not second_stage:
			solution = get_first_stage_solution(solution, self.feasibility_checker.world_instance.first_stage_tasks)
		search_space = []
		for k, car_moves in solution.items():
			if len(car_moves) > 1:
				idx = list(range(len(car_moves)))
				for i, j in itertools.combinations(idx, 2):
					search_space.append((k,i,j))

		return search_space


class InterSwap(LocalSearchOperator):
	'''
    An Inter Swap LSO swaps two car-moves between two service employees.
    '''

	def __init__(self, solution, first_stage_tasks, feasibility_checker):
		super().__init__(solution, first_stage_tasks, feasibility_checker)

	def _mutate(self, emp1, emp2, s=None):
		e1, idx_1 = emp1
		e2, idx_2 = emp2
		first_stage, second_stage = get_first_and_second_stage_solution(self._current_solution, self.first_stage_tasks)
		if s is None:
			if first_stage[e1] and not first_stage[e2]:
				first_stage[e2].append(first_stage[e1][idx_1])

			elif first_stage[e2] and not first_stage[e1]:
				first_stage[e1].append(first_stage[e2][idx_2])
			else:
				first_stage[e2][idx_2], first_stage[e1][idx_1] = first_stage[e1][idx_1], first_stage[e2][idx_2]
		else:
			second_stage[e2][s][idx_2], second_stage[e1][s][idx_1] = second_stage[e1][s][idx_1], \
																	 second_stage[e2][s][idx_2]

		self._candidate_solution = reconstruct_solution_from_first_and_second_stage(first_stage, second_stage)

	def search(self, strategy, shuffle=False):
		# TODO: support for second stage solutions
		current_obj_val = get_obj_val_of_solution_dict(self._current_solution,
													   self.feasibility_checker.world_instance, True)
		best_solution = self._current_solution
		emp_pairs = self._get_search_space(self._current_solution)

		for emp1, emp2 in emp_pairs:
			self._mutate(emp1, emp2)
			self.visited_list.append(self.hash_key)
			if not self.feasibility_checker.is_solution_feasible(self._candidate_solution):
				continue
			# inter_swap.to_string()
			candidate_obj_val = get_obj_val_of_solution_dict(self._candidate_solution, self.feasibility_checker.world_instance, True)
			##print(f"current_obj_val {current_obj_val}")
			# print(f"candidate_obj_val {candidate_obj_val}")

			if candidate_obj_val > current_obj_val:
				print("New best solution found!")
				self.to_string()
				current_obj_val = candidate_obj_val
				best_solution = self.candidate_solution
				if strategy == "best_first":
					break
		self._current_solution = best_solution
		return best_solution

	def _get_search_space(self, solution, second_stage=False):
		if not second_stage:
			solution = get_first_stage_solution(solution, self.feasibility_checker.world_instance.first_stage_tasks)
		emp_pair_lists = []
		for k, v in solution.items():
			emp = []
			for i, cm in enumerate(v):
				emp.append((k, i))
			emp_pair_lists.append(emp)
		emp_pairs = []
		for i, j in itertools.combinations(range(len(emp_pair_lists)), 2):
			emp_pairs.append(itertools.product(emp_pair_lists[i], emp_pair_lists[j]))
		emp_pairs = [emp_pair for itertools_obj in emp_pairs for emp_pair in itertools_obj]
		return emp_pairs


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
	print("\n---- Construction Heuristic ----")
	filename = "InstanceGenerator/InstanceFiles/25nodes/25-2-2-1_a"
	ch = ConstructionHeuristic(filename + ".pkl")
	ch.add_car_moves_to_employees()
	print(ch.get_obj_val(true_objective=True, both=True))
	fc = FeasibilityChecker(ch.world_instance)
	print("\n---- Local Search ----")

	print("IntraMove")
	intra_move = IntraMove(ch.assigned_car_moves, ch.world_instance.first_stage_tasks, fc)
	solution = intra_move.search("best_first")
	print("InterSwap")
	inter_swap = InterSwap(solution, ch.world_instance.first_stage_tasks, fc)
	solution = inter_swap.search("best_first")
	ch.rebuild(solution, "second_stage")
	print(ch.get_obj_val(true_objective=True, both=True))


