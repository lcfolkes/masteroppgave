import copy
from collections import OrderedDict
import random
from DestroyAndRepairHeuristics.destroy import Destroy, RandomRemoval, WorstRemoval, ShawRemoval
from DestroyAndRepairHeuristics.repair import Repair, GreedyInsertion, RegretInsertion
from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Gurobi.Model.run_model import run_model
from Heuristics.helper_functions_heuristics import safe_zero_division, get_solution_list
from Heuristics.objective_function import get_obj_val_of_car_moves
from construction_heuristic_new import ConstructionHeuristic
from path_manager import path_to_src
import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir(path_to_src)

os.chdir(path_to_src)

_IS_BEST = 3.0
_IS_BETTER = 2.0
_IS_ACCEPTED = 1.0
_IS_REJECTED = 0.0

'''

_IS_BEST (sigma_1): The last remove-insert operation resulted in a new global best solution
_IS_BETTER (sigma_2): The last remove-insert operation resulted in a solution that has not been accepted before.
         The cost of the new solution is better than the cost of the current solution.

_IS_ACCEPTED (sigma_3): The last remove-insert operation resulted in a solution that has not been accepted before. The cost of
            the new solution is worse than the cost of current solution, but the solution was accepted.
_IS_REJECTED

'''

#TODO: Need to keep track of visited solutions by assigning a hash key to each solution and storing the key in a hash table



class ALNS():

    def __init__(self, filename):
        self.filename = filename
        print(filename)

        self.destroy_operators = OrderedDict({'random': 1.0, 'worst': 1.0, 'shaw': 1.0})
        self.repair_operators = OrderedDict({'greedy': 1.0, 'regret2': 1.0, 'regret3': 1.0})
        self.destroy_operators_record = OrderedDict({'random': [1.0, 0.0], 'worst': [1.0, 0.0], 'shaw': [1.0, 0.0]})
        self.repair_operators_record = OrderedDict({'greedy': [1.0, 0.0], 'regret2': [1.0, 0.0], 'regret3': [1.0, 0.0]})

        self._callbacks = {}

        self.best_solution = None
        self.best_solutions = None
        self.best_obj_val = 0
        self.run()

    def run(self):
        visited_hash_keys = set()

        solution = ConstructionHeuristic(self.filename)
        solution.add_car_moves_to_employees()
        true_obj_val, best_obj_val = solution.get_obj_val(both=True)
        current_obj_val = best_obj_val
        true_obj_vals = [true_obj_val]
        heuristic_obj_vals = [best_obj_val]
        best_solution = (copy.deepcopy(solution), true_obj_val)
        print(f"Construction heuristic obj. val {true_obj_val}")
        current_solution = copy.deepcopy(solution)
        visited_hash_keys.add(current_solution.hash_key)
        # TODO: this is the old objective function val


        temperature = 1000 # Start temperature must be set differently
        cooling_rate = 0.9 # cooling_rate in (0,1)

        # SEGMENTS
        for i in range(10):
            for j in range(10):
                print(f"Iteration {i*10 + j}")
                print(f"Best objective value {best_obj_val}")

                solution = copy.deepcopy(current_solution)
                destroy = self._get_destroy_operator(solution=solution.assigned_car_moves,
                                               num_first_stage_tasks=solution.world_instance.first_stage_tasks,
                                               neighborhood_size=2, randomization_degree=1,
                                               parking_nodes=solution.parking_nodes)

                repair = self._get_repair_operator(destroyed_solution_object=destroy,
                                             unused_car_moves=solution.unused_car_moves,
                                             parking_nodes=solution.parking_nodes,
                                             world_instance=solution.world_instance)
                solution.rebuild(repair.repaired_solution)

                if solution.hash_key in visited_hash_keys:
                    #print("Solution already visited")
                    pass
                else:
                    #update scores for repair and destroy
                    visited_hash_keys.add(solution.hash_key)
                    true_obj_val, current_obj_val = solution.get_obj_val(both=True)
                    heuristic_obj_vals.append(current_obj_val)
                    true_obj_vals.append(true_obj_val)

                    if self._accept(current_obj_val, best_obj_val, temperature):
                        if current_obj_val > best_obj_val:
                            best_obj_val = current_obj_val
                            best_solution = (copy.deepcopy(solution), true_obj_val)
                            self._update_weight_record(_IS_BEST, destroy, repair)
                        elif current_obj_val > current_obj_val:
                            self._update_weight_record(_IS_BETTER, destroy, repair)
                        else:
                            self._update_weight_record(_IS_ACCEPTED, destroy, repair)

                        current_solution = copy.deepcopy(solution)
                    else:
                        print("not accepted")

                temperature *= cooling_rate


            self._update_score_adjustment_parameters()


        self.best_solution = best_solution
        self.best_obj_val = best_obj_val
        plt.plot(heuristic_obj_vals, c="red", label="Heuristic obj. val")
        plt.plot(true_obj_vals, c="green", label="True obj. val")
        plt.suptitle('Objective value comparison')
        plt.xlabel('instance')
        plt.ylabel('obj. val')
        plt.legend(loc="upper right")
        plt.show()
        #print(obj_vals)
        print(self.destroy_operators)
        print(self.repair_operators)
        print("best solution")
        print("obj_val", best_solution[1])
        best_solution[0].print_solution()
        #best_solution.print_solution()

    def _accept(self, current_obj_val, best_obj_val, temperature) -> bool:
        if current_obj_val > best_obj_val:
            acceptance_probability = 1
        else:
            p = np.exp(- (best_obj_val - current_obj_val) / temperature)
            acceptance_probability = p

        accept = acceptance_probability > random.random()
        return accept

    def _get_destroy_operator(self, solution, num_first_stage_tasks, neighborhood_size, randomization_degree,
                             parking_nodes) -> Destroy:
        w_sum_destroy = sum(w for o, w in self.destroy_operators.items())
        w_dist_destroy = [w / w_sum_destroy for o, w in self.destroy_operators.items()]
        operator = random.choices(list(self.destroy_operators), w_dist_destroy)[0]
        self.destroy_operators_record[operator][1] += 1

        if operator == "random":
            return RandomRemoval(solution, num_first_stage_tasks, neighborhood_size)
        elif operator == "worst":
            return WorstRemoval(solution, num_first_stage_tasks, neighborhood_size, randomization_degree, parking_nodes)
        elif operator == "shaw":
            return ShawRemoval(solution, num_first_stage_tasks, neighborhood_size, randomization_degree)
        else:
            exit("Destroy operator does not exist")


    def _get_repair_operator(self, destroyed_solution_object, unused_car_moves, parking_nodes, world_instance) \
            -> Repair:
        w_sum_repair = sum(w for o, w in self.repair_operators.items())
        w_dist_repair = [w / w_sum_repair for o, w in self.repair_operators.items()]
        operator = random.choices(list(self.repair_operators), w_dist_repair)[0]
        self.repair_operators_record[operator][1] += 1

        if operator == "greedy":
            return GreedyInsertion(destroyed_solution_object, unused_car_moves, parking_nodes, world_instance)
        elif operator == "regret2":
            return RegretInsertion(destroyed_solution_object, unused_car_moves, parking_nodes, world_instance, regret_nr=2)
        elif operator == "regret3":
            return RegretInsertion(destroyed_solution_object, unused_car_moves, parking_nodes, world_instance, regret_nr=3)
        else:
            exit("Repair operator does not exist")



    def _update_weight_record(self, operator_score, destroy, repair):

        # DESTROY
        if isinstance(destroy, RandomRemoval):
            self.destroy_operators_record['random'][0] += operator_score
        elif isinstance(destroy, WorstRemoval):
            self.destroy_operators_record['worst'][0] += operator_score
        elif isinstance(destroy, ShawRemoval):
            self.destroy_operators_record['shaw'][0] += operator_score

        # REPAIR
        if isinstance(repair, GreedyInsertion):
            self.repair_operators_record['greedy'][0] += operator_score
        elif isinstance(repair, RegretInsertion):
            if repair.regret_nr == 2:
                self.repair_operators_record['regret2'][0] += operator_score
            elif repair.regret_nr == 3:
                self.repair_operators_record['regret3'][0] += operator_score

    def _update_score_adjustment_parameters(self):
        reaction_factor = 0.5

        for k, v in self.destroy_operators.items():
            self.destroy_operators[k] = self.destroy_operators[k]*(1.0-reaction_factor) + \
                                        safe_zero_division(reaction_factor, self.destroy_operators_record[k][1]) * \
                                        self.destroy_operators_record[k][0]
            self.destroy_operators_record[k][0] = self.destroy_operators[k]
            self.destroy_operators_record[k][0] = 0

        for k, v in self.repair_operators.items():
            self.repair_operators[k] = self.repair_operators[k]*(1.0-reaction_factor) + \
                                        safe_zero_division(reaction_factor, self.repair_operators_record[k][1]) * \
                                       self.repair_operators_record[k][0]
            self.repair_operators_record[k][0] = self.repair_operators[k]
            self.repair_operators_record[k][0] = 0



if __name__ == "__main__":
    filename = "InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_special_case"
    alns = ALNS(filename + ".pkl")

    print("\n############## Evaluate solution ##############")
    gi = GurobiInstance(filename + ".yaml", employees=alns.best_solution[0].employees, optimize=False)
    run_model(gi)
    print("\n############## Optimal solution ##############")
    gi2 = GurobiInstance(filename + ".yaml")
    run_model(gi2)


    # TODO: check objective function


    '''
    filename = "InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_b"
    print("\n---- HEURISTIC ----")
    ch = ConstructionHeuristic(filename+".pkl")
    ch.add_car_moves_to_employees()
    ch.print_solution()
    rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
                       neighborhood_size=2)
    rr.to_string()
    gi = GreedyInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
                         parking_nodes=ch.parking_nodes, world_instance=ch.world_instance)
    gi.to_string()
    #ri = RegretInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
    #                     parking_nodes=ch.parking_nodes, world_instance=ch.world_instance, regret_nr=1)
    #ri.to_string()

    fc = FeasibilityChecker(ch.world_instance)
    print("feasibilityChecker")
    print(fc.is_first_stage_solution_feasible(gi.repaired_solution))

    ch.rebuild(gi.repaired_solution, True)
    # gi = GurobiInstance(filename + ".yaml")
    # run_model(gi)

    print("\n---- GUROBI ----")
    print("Verify in Gurobi")
    gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=False)
    run_model(gi)

    ch.add_car_moves_to_employees()
    ch.print_solution()

    print("\n---- GUROBI ----")
    print("Verify in Gurobi")
    gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=False)
    run_model(gi)

    print("\nOptimize in Gurobi")
    gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=True)
    run_model(gi)
    '''
