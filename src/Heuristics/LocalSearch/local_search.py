from Heuristics.LocalSearch.local_search_operator import IntraMove, InterSwap
from Heuristics.construction_heuristic import ConstructionHeuristic
from Heuristics.feasibility_checker import FeasibilityChecker
import itertools

from Heuristics.helper_functions_heuristics import get_first_and_second_stage_solution_list_from_dict
from Heuristics.objective_function import get_obj_val_of_solution_dict


class LocalSearch:
	def __init__(self, solution, first_stage_tasks, feasibility_checker):
		self.first_stage_tasks = first_stage_tasks
		self.feasibility_checker = feasibility_checker
		self.visited_list = []
		self.solution = solution

	def search(self, strategy="best_first"):
		print("IntraMove")
		intra_move = IntraMove(self.solution, self.feasibility_checker.world_instance.first_stage_tasks)
		solution = self._intra_move_search(intra_move, strategy)
		inter_swap = InterSwap(solution, self.feasibility_checker.world_instance.first_stage_tasks)
		print("InterSwap")
		solution = self._inter_swap_search(inter_swap, strategy)
		self.solution = solution
		return solution

	def _intra_move_search(self, intra_move, strategy, shuffle=False):
		current_obj_val = get_obj_val_of_solution_dict(intra_move.current_solution, self.feasibility_checker.world_instance, True)
		best_solution = intra_move.current_solution
		for k, v in self.solution.items():
			idx = range(len(v))
			#TODO: add shuffle feature
			for i, j in itertools.combinations(idx, 2):
				intra_move.mutate(k, i, j)
				self.visited_list.append(intra_move.hash_key)
				if not self.feasibility_checker.is_solution_feasible(intra_move.candidate_solution):
					continue
				intra_move.to_string()
				candidate_obj_val = get_obj_val_of_solution_dict(intra_move.candidate_solution, self.feasibility_checker.world_instance, True)
				if candidate_obj_val > current_obj_val:
					current_obj_val = candidate_obj_val
					best_solution = intra_move.candidate_solution
					if strategy == "best_first":
						break
		return best_solution

	def _inter_swap_search(self, inter_swap, strategy):
		current_obj_val = get_obj_val_of_solution_dict(inter_swap.current_solution, self.feasibility_checker.world_instance, True)
		best_solution = inter_swap.current_solution
		emp_pairs = self._get_emp_pairs_inter_swap(inter_swap.current_solution)

		for emp1, emp2 in emp_pairs:
			inter_swap.mutate(emp1, emp2)
			self.visited_list.append(inter_swap.hash_key)
			if not self.feasibility_checker.is_solution_feasible(inter_swap.candidate_solution):
				continue
			inter_swap.to_string()
			candidate_obj_val = get_obj_val_of_solution_dict(inter_swap.candidate_solution, self.feasibility_checker.world_instance, True)
			if candidate_obj_val > current_obj_val:
				current_obj_val = candidate_obj_val
				best_solution = inter_swap.candidate_solution
				if strategy == "best_first":
					break
		return best_solution

	def _get_emp_pairs_inter_swap(self, solution):
		emp_pair_lists = []
		for k, v in solution.items():
			emp = []
			for i in range(len(v)):
				emp.append((k, i))
			emp_pair_lists.append(emp)
		emp_pairs = []
		for i, j in itertools.combinations(range(len(emp_pair_lists)), 2):
			emp_pairs.append(itertools.product(emp_pair_lists[i], emp_pair_lists[j]))
		emp_pairs = [emp_pair for itertools_obj in emp_pairs for emp_pair in itertools_obj]
		return emp_pairs


if __name__ == "__main__":
	print("\n---- Local Search ----")
	filename = "InstanceGenerator/InstanceFiles/25nodes/25-2-2-1_a"
	ch = ConstructionHeuristic(filename + ".pkl")
	ch.add_car_moves_to_employees()
	print(ch.get_obj_val(true_objective=True, both=True))
	#rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
	#				   neighborhood_size=2)
	# rr.to_string()
	#gi = GreedyInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
	#					 parking_nodes=ch.parking_nodes, world_instance=ch.world_instance)
	fc = FeasibilityChecker(ch.world_instance)
	local_search = LocalSearch(ch.assigned_car_moves, ch.world_instance.first_stage_tasks, fc)
	local_search.search("best_first")


	#TODO: find out why old calculation of cost_travel_time_between_car_moves is incorrect

