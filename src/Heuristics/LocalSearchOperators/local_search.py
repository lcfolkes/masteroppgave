from Heuristics.LocalSearchOperators.local_search_operator import IntraMove, InterSwap
from Heuristics.construction_heuristic import ConstructionHeuristic
from Heuristics.feasibility_checker import FeasibilityChecker
from itertools import combinations

from Heuristics.helper_functions_heuristics import get_first_and_second_stage_solution_list_from_dict
from Heuristics.objective_function import get_obj_val_of_solution_dict


class LocalSearch:
	def __init__(self, solution, first_stage_tasks, feasibility_checker, strategy=None):
		self.solution = solution
		self.first_stage_tasks = first_stage_tasks
		self.feasibility_checker = feasibility_checker
		self.strategy = strategy
		self.intra_move = IntraMove(solution, feasibility_checker.world_instance.first_stage_tasks)
		self.visited_list = []

	def search(self):
		current_obj_val = get_obj_val_of_solution_dict(self.intra_move.current_solution, self.feasibility_checker.world_instance, True)
		print(current_obj_val)
		for k, v in self.solution.items():
			idx = list(range(len(v)))
			idx_pairs = list(combinations(idx, 2))
			for i, j in idx_pairs:
				self.intra_move.mutate(k, i, j)
				self.intra_move.to_string()
				self.visited_list.append(self.intra_move.hash_key)
				if not self.feasibility_checker.is_solution_feasible(self.intra_move.candidate_solution):
					continue
				candidate_obj_val = get_obj_val_of_solution_dict(self.intra_move.candidate_solution, self.feasibility_checker.world_instance, True)
				print(candidate_obj_val)

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
	local_search.search()


	#TODO: find out why old calculation of cost_travel_time_between_car_moves is incorrect

