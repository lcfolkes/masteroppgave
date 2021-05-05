from Heuristics.LocalSearch.local_search_operator import IntraMove, InterSwap
from Heuristics.new_construction_heuristic import ConstructionHeuristic
from Heuristics.feasibility_checker import FeasibilityChecker


class LocalSearch:
    def __init__(self, solution, first_stage_tasks, feasibility_checker):
        self.first_stage_tasks = first_stage_tasks
        self.feasibility_checker = feasibility_checker
        self.visited_list = []
        self.solution = solution

    '''
    def search(self, strategy="best_first"):
        #print("\n---- Local Search ----")

        print("IntraMove")
        intra_move = IntraMove(self.solution, self.first_stage_tasks, self.feasibility_checker)
        solution = intra_move.search(strategy, True)
        self.visited_list += intra_move.visited_list

        print("InterSwap")
        inter_swap = InterSwap(solution, self.first_stage_tasks, self.feasibility_checker)
        solution = inter_swap.search(strategy, True)
        self.visited_list += inter_swap.visited_list


        print("IntraMove")
        intra_move = IntraMove(solution, self.first_stage_tasks, self.feasibility_checker)
        solution = intra_move.search(strategy, False)
        self.visited_list += intra_move.visited_list

        print("InterSwap")
        inter_swap = InterSwap(solution, self.first_stage_tasks, self.feasibility_checker)
        solution = inter_swap.search(strategy, False)
        self.visited_list += inter_swap.visited_list

        self.solution = solution
        return solution
    '''

    def search(self, strategy="best_first"):
        # print("\n---- Local Search ----")


        solution = self.solution
        intra_active_first_stage = True
        inter_active_first_stage = True
        intra_active_second_stage = True
        inter_active_second_stage = True

        while intra_active_first_stage or intra_active_second_stage or inter_active_first_stage or \
                inter_active_second_stage:

            new_solution_found_intra_first_stage = True
            new_solution_found_inter_first_stage = True
            new_solution_found_intra_second_stage = True
            new_solution_found_inter_second_stage = True

            intra_fs_counter = 0
            while new_solution_found_intra_first_stage:
                #print("IntraMove")
                intra_move = IntraMove(solution, self.first_stage_tasks, self.feasibility_checker)
                solution, new_solution_found_intra_first_stage = intra_move.search(strategy, first_stage=True)
                feasible, inter_travel_time = self.feasibility_checker.is_solution_feasible(solution,
                                                                                            return_inter_node_travel_time=True)
                self.visited_list += intra_move.visited_list
                intra_fs_counter += 1

            if intra_fs_counter == 1:
                intra_active_first_stage = False

            intra_ss_counter = 0
            while new_solution_found_intra_second_stage:
                #print("IntraMove")
                intra_move = IntraMove(solution, self.first_stage_tasks, self.feasibility_checker)
                solution, new_solution_found_intra_second_stage = intra_move.search(strategy, first_stage=False)
                feasible, inter_travel_time = self.feasibility_checker.is_solution_feasible(solution,
                                                                                            return_inter_node_travel_time=True)
                self.visited_list += intra_move.visited_list
                intra_ss_counter += 1

            if intra_ss_counter == 1:
                intra_active_second_stage = False

            inter_fs_counter = 0
            while new_solution_found_inter_first_stage:
                #print("InterSwap")
                inter_swap = InterSwap(solution, self.first_stage_tasks, self.feasibility_checker)
                solution, new_solution_found_inter_first_stage = inter_swap.search(strategy, first_stage=True)
                feasible, inter_travel_time = self.feasibility_checker.is_solution_feasible(solution,
                                                                                            return_inter_node_travel_time=True)
                self.visited_list += inter_swap.visited_list
                inter_fs_counter += 1

            if inter_fs_counter == 1:
                inter_active_first_stage = False

            inter_ss_counter = 0
            while new_solution_found_inter_second_stage:
                #print("InterSwap")
                inter_swap = InterSwap(solution, self.first_stage_tasks, self.feasibility_checker)
                solution, new_solution_found_inter_second_stage = inter_swap.search(strategy, first_stage=False)
                feasible, inter_travel_time = self.feasibility_checker.is_solution_feasible(solution,
                                                                                            return_inter_node_travel_time=True)
                self.visited_list += inter_swap.visited_list
                inter_ss_counter += 1

            if inter_ss_counter == 1:
                inter_active_second_stage = False

        self.solution = solution
        return solution


if __name__ == "__main__":
    print("\n---- Local Search ----")
    filename = "InstanceGenerator/InstanceFiles/20nodes/20-10-2-1_a"
    ch = ConstructionHeuristic(filename + ".pkl")
    ch.construct()
    print(ch.get_obj_val(true_objective=True, both=True))
    # rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
    #				   neighborhood_size=2)
    # rr.to_string()
    # gi = GreedyInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
    #					 parking_nodes=ch.parking_nodes, world_instance=ch.world_instance)
    fc = FeasibilityChecker(ch.world_instance)

    local_search = LocalSearch(ch.assigned_car_moves, ch.world_instance.first_stage_tasks, fc)
    local_search.search("best_first")
    ch.rebuild(local_search.solution, "second_stage")
    ch.print_solution()
