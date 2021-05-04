from abc import ABC, abstractmethod

from Heuristics.construction_heuristic import ConstructionHeuristic
from Heuristics.feasibility_checker import FeasibilityChecker
from Heuristics.helper_functions_heuristics import get_first_and_second_stage_solution, \
	reconstruct_solution_from_first_and_second_stage, get_first_stage_solution, get_second_stage_solution_dict
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

	def search(self, strategy, first_stage, shuffle=False):
		self.visited_list = []
		new_solution_found = False
		#current_obj_val = get_obj_val_of_solution_dict(self._current_solution, self.feasibility_checker.world_instance, True)
		_, current_inter_node_travel_time = self.feasibility_checker.is_solution_feasible(self._current_solution, return_inter_node_travel_time=True)
		best_solution = self._current_solution
		search_space = self._get_search_space(self._current_solution, first_stage)

		for neighbor in search_space:
			if first_stage:
				k, i, j = neighbor
				self._mutate(k, i, j)
			else:
				k, i, j, s = neighbor
				self._mutate(k, i, j, s)
			self.visited_list.append(self.hash_key)
			feasible, candidate_inter_node_travel_time = self.feasibility_checker.is_solution_feasible(self._candidate_solution, return_inter_node_travel_time=True)
			if not feasible:
				continue
			#intra_move.to_string()
			#candidate_obj_val = get_obj_val_of_solution_dict(self._candidate_solution, self.feasibility_checker.world_instance, True)
			#print(f"current_obj_val {current_obj_val}")
			#print(f"candidate_obj_val {candidate_obj_val}")
			if candidate_inter_node_travel_time < current_inter_node_travel_time:
				new_solution_found = True
				print("New best solution found!")
				self.to_string()
				current_inter_node_travel_time = candidate_inter_node_travel_time
				best_solution = self._candidate_solution
				if strategy == "best_first":
					break
		self._current_solution = best_solution
		return best_solution, new_solution_found

	def _get_search_space(self, solution, first_stage):
		search_space = []
		if first_stage:
			solution = get_first_stage_solution(solution, self.feasibility_checker.world_instance.first_stage_tasks)
			for k, car_moves in solution.items():
				if len(car_moves) > 1:
					idx = list(range(len(car_moves)))
					# check if not e.g. 0,1 and 1,0
					for i, j in itertools.combinations(idx, 2):
						search_space.append((k, i, j))
		else:
			solution = get_second_stage_solution_dict(solution, self.feasibility_checker.world_instance.first_stage_tasks)
			for k, scenarios in solution.items():
				for s, car_moves in enumerate(scenarios):
					if len(car_moves) > 1:
						idx = list(range(len(car_moves)))
						# check if not e.g. 0,1 and 1,0
						for i, j in itertools.combinations(idx, 2):
							search_space.append((k, i, j, s))


		return search_space

	def _mutate(self, employee, i, j, s=None):
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

	def __init__(self, solution, first_stage_tasks, feasibility_checker):
		super().__init__(solution, first_stage_tasks, feasibility_checker)



	def search(self, strategy, first_stage, shuffle=False):
		_, current_inter_node_travel_time = self.feasibility_checker.is_solution_feasible(self._current_solution, return_inter_node_travel_time=True)

		best_solution = self._current_solution
		search_space = self._get_search_space(self._current_solution, first_stage)
		new_solution_found = False

		if first_stage:
			for neighbor in search_space:
				emp1, emp2 = neighbor
				self._mutate(emp1, emp2)
				self.visited_list.append(self.hash_key)
				feasible, candidate_inter_node_travel_time = self.feasibility_checker.is_solution_feasible(
					self._candidate_solution, return_inter_node_travel_time=True)

				if not feasible:
					continue

				if candidate_inter_node_travel_time < current_inter_node_travel_time:
					print("New best solution found!")
					new_solution_found = True
					self.to_string()
					current_inter_node_travel_time = candidate_inter_node_travel_time
					best_solution = self.candidate_solution
					if strategy == "best_first":
						break
		else:
			for s, scenario_search_space in enumerate(search_space):
				for neighbor in scenario_search_space:
					emp1, emp2 = neighbor
					self._mutate(emp1, emp2, s)
					self.visited_list.append(self.hash_key)
					feasible, candidate_inter_node_travel_time = self.feasibility_checker.is_solution_feasible(
						self._candidate_solution, return_inter_node_travel_time=True)

					if not feasible:
						continue

					if candidate_inter_node_travel_time < current_inter_node_travel_time:
						print("New best solution found!")
						new_solution_found = True
						self.to_string()
						current_inter_node_travel_time = candidate_inter_node_travel_time
						best_solution = self.candidate_solution
						if strategy == "best_first":
							break

		self._current_solution = best_solution
		return best_solution, new_solution_found

	def _get_search_space(self, solution, first_stage):
		emp_pair_lists = []
		if first_stage:
			solution = get_first_stage_solution(solution, self.feasibility_checker.world_instance.first_stage_tasks)
			for k, car_moves in solution.items():
				emp = []
				for i, cm in enumerate(car_moves):
					emp.append((k, i))
				if not car_moves:
					emp.append((k, 0))
				emp_pair_lists.append(emp)
			emp_pairs = []
			for i, j in itertools.combinations(range(len(emp_pair_lists)), 2):
				emp_pairs.append(itertools.product(emp_pair_lists[i], emp_pair_lists[j]))
			emp_pairs = [emp_pair for itertools_obj in emp_pairs for emp_pair in itertools_obj]
		else:
			solution = get_second_stage_solution_dict(solution, self.feasibility_checker.world_instance.first_stage_tasks)
			for _ in range(self.feasibility_checker.world_instance.num_scenarios):
				emp_pair_lists.append([])
			for k, scenarios in solution.items():
				for s, car_moves in enumerate(scenarios):
					emp = []
					for i, cm in enumerate(car_moves):
						emp.append((k, i))
					if not car_moves:
						emp.append((k, 0))
					emp_pair_lists[s].append(emp)
			emp_pairs = []
			for s in range(len(emp_pair_lists)):
				emp_pairs_scenario = []
				for i, j in itertools.combinations(range(len(emp_pair_lists[s])), 2):
					emp_pairs_scenario.append(itertools.product(emp_pair_lists[s][i], emp_pair_lists[s][j]))
				emp_pairs_scenario = [emp_pair for itertools_obj in emp_pairs_scenario for emp_pair in itertools_obj]
				emp_pairs.append(emp_pairs_scenario)
		return emp_pairs

	def _mutate(self, emp1, emp2, s=None):
		e1, idx_1 = emp1
		e2, idx_2 = emp2
		first_stage, second_stage = get_first_and_second_stage_solution(self._current_solution, self.first_stage_tasks)


		if s is None:
			if first_stage[e1] and not first_stage[e2]:
				car_move = first_stage[e1].pop(idx_1)
				first_stage[e2].append(car_move)

			elif first_stage[e2] and not first_stage[e1]:
				car_move = first_stage[e2].pop(idx_2)
				first_stage[e1].append(car_move)
			elif not second_stage[e1] and not second_stage[e2]:
				pass
			else:
				first_stage[e2][idx_2], first_stage[e1][idx_1] = first_stage[e1][idx_1], first_stage[e2][idx_2]
		else:

			if second_stage[e1][s] and not second_stage[e2][s]:
				car_move = second_stage[e1][s].pop(idx_1)
				second_stage[e2][s].append(car_move)
			elif second_stage[e2][s] and not second_stage[e1][s]:
				car_move = second_stage[e2][s].pop(idx_2)
				second_stage[e1][s].append(car_move)
			elif not second_stage[e1][s] and not second_stage[e2][s]:
				pass
			else:
				second_stage[e2][s][idx_2], second_stage[e1][s][idx_1] = second_stage[e1][s][idx_1], second_stage[e2][s][idx_2]

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
	print("\n---- Construction Heuristic ----")
	filename = "InstanceGenerator/InstanceFiles/25nodes/25-2-2-1_a"
	ch = ConstructionHeuristic(filename + ".pkl")
	ch.construct()
	print(ch.get_obj_val(true_objective=True, both=True))
	fc = FeasibilityChecker(ch.world_instance)
	print("\n---- Local Search ----")

	print("IntraMove")
	intra_move = IntraMove(ch.assigned_car_moves, ch.world_instance.first_stage_tasks, fc)
	solution = intra_move.search("best_first", True)
	print("InterSwap")
	inter_swap = InterSwap(solution, ch.world_instance.first_stage_tasks, fc)
	solution = inter_swap.search("best_first", True)
	ch.rebuild(solution, "second_stage")
	print(ch.get_obj_val(true_objective=True, both=True))


