import copy
import sys
from collections import OrderedDict
import random
from DestroyAndRepairHeuristics.destroy import Destroy, RandomRemoval, WorstRemoval, ShawRemoval, ChargeRemoval
from DestroyAndRepairHeuristics.repair import Repair, GreedyInsertion, RegretInsertion, ChargeInsertion
from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Gurobi.Model.run_model import run_model
from Heuristics.LocalSearch.local_search import LocalSearch
from Heuristics.helper_functions_heuristics import safe_zero_division, get_first_stage_solution
from Heuristics.best_construction_heuristic import ConstructionHeuristic

from path_manager import path_to_src
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
os.chdir(path_to_src)

_IS_BEST = 33.0
_IS_BETTER = 9.0
_IS_ACCEPTED = 13.0

'''
_IS_BEST (sigma_1): The last remove-insert operation resulted in a new global best solution
_IS_BETTER (sigma_2): The last remove-insert operation resulted in a solution that has not been accepted before.
         The cost of the new solution is better than the cost of the current solution.
_IS_ACCEPTED (sigma_3): The last remove-insert operation resulted in a solution that has not been accepted before. The cost of
            the new solution is worse than the cost of current solution, but the solution was accepted.
_IS_REJECTED
'''


# TODO: Need to keep track of visited solutions by assigning a hash key to each solution and storing the key in a hash table


class ALNS():

    def __init__(self, filename):
        self.filename = filename

        self.best_solution = None
        self.best_solutions = None
        self.best_obj_val = 0

        solution = ConstructionHeuristic(self.filename)
        self.num_employees = len(solution.employees)
        self.operator_pairs = self._initialize_operators()
        self.operators_record = self._initialize_operator_records()
        self.run(solution)


    def run(self, solution):
        # TODO: in order to save time, this could be implemented as a queue (as in tabu search)
        start = time.perf_counter()
        visited_hash_keys = set()

        solution.construct(verbose=True)
        solution.print_solution()
        true_obj_val, best_obj_val = solution.get_obj_val(both=True)
        current_obj_val = best_obj_val
        true_obj_vals = [true_obj_val]
        heuristic_obj_vals = [best_obj_val]
        best_solution = (solution, true_obj_val)
        print(f"Construction heuristic true obj. val {true_obj_val}")
        print(f"Heuristic obj. val {heuristic_obj_vals[0]}")
        print(f"Heuristic obj. val {heuristic_obj_vals[0]}")
        current_solution = solution
        visited_hash_keys.add(current_solution.hash_key)
        MODE = "LOCAL"

        # temperature = 1000  # Start temperature must be set differently. High temperature --> more randomness
        temperature = (-np.abs(heuristic_obj_vals[0])) * 0.05 / np.log(0.5)
        # cooling_rate = 0.5  # cooling_rate in (0,1)
        cooling_rate = np.exp(np.log(0.002) / 200)
        iterations_alns = 50
        iterations_segment = 10

        # SEGMENTS
        try:

            for i in range(iterations_alns):
                # print(i)

                # print(f"Iteration {i * 10}")
                # print(f"Best objective value {best_solution[1]}")
                # print(f"Best heuristic objective value {max(heuristic_obj_vals)}")
                loop = tqdm(range(iterations_segment), total=iterations_segment, leave=True)
                loop.set_description(f"Segment[{i}/{100}]")

                loop.set_postfix(current_obj_val=current_obj_val, best_obj_val=best_obj_val,
                                 best_true_obj_val=best_solution[1])
                output_text = "\n"
                counter = 1

                for j in loop:
                    # print(f"Iteration {i*10 + j}")
                    candidate_solution = copy.deepcopy(current_solution)

                    if MODE == "LOCAL_FIRST":
                        print("\n----- LOCAL SEARCH FIRST BEST -----")
                        local_search = LocalSearch(candidate_solution.assigned_car_moves,
                                                   candidate_solution.world_instance.first_stage_tasks,
                                                   candidate_solution.feasibility_checker)
                        local_search.search("best_first")
                        candidate_solution.rebuild(local_search.solution, "second_stage")
                        visited_hash_keys.update(local_search.visited_list)

                    elif MODE == "LOCAL_FULL":
                        print("\n----- LOCAL SEARCH FULL -----")
                        local_search = LocalSearch(candidate_solution.assigned_car_moves,
                                                   candidate_solution.world_instance.first_stage_tasks,
                                                   candidate_solution.feasibility_checker)
                        local_search.search("full")
                        candidate_solution.rebuild(local_search.solution, "second_stage")
                        visited_hash_keys.update(local_search.visited_list)


                    elif MODE == "LNS":
                        # print("\n----- LARGE NEIGHBORHOOD SEARCH -----")
                        destroy_heuristic, operator_pair = self._get_destroy_operator(
                            solution=candidate_solution.assigned_car_moves,
                            neighborhood_size=1, randomization_degree=40,
                            world_instance=candidate_solution.world_instance)
                        destroy_heuristic.destroy()
                        # print("Destroy: ", destroy_heuristic, destroy_heuristic.solution)
                        # destroy_heuristic.to_string()
                        # print(destroy)

                        repair_heuristic = self._get_repair_operator(destroyed_solution_object=destroy_heuristic,
                                                                     unused_car_moves=candidate_solution.
                                                                     unused_car_moves,
                                                                     world_instance=candidate_solution.world_instance,
                                                                     operator_pair=operator_pair)
                        repair_heuristic.repair()

                        # print("Repair: ", repair_heuristic, repair_heuristic.solution)
                        # repair_heuristic.to_string()
                        hash_key = repair_heuristic.hash_key
                        if hash_key in visited_hash_keys:
                            output_text += str(counter) + " Already visited solution\n"
                            counter += 1
                            continue
                        visited_hash_keys.add(hash_key)
                        candidate_solution.rebuild(repair_heuristic.solution)
                        '''
                        hash_key = candidate_solution.hash_key
                        if hash_key in visited_hash_keys:
                            continue
                        visited_hash_keys.add(hash_key)
                        '''

                    true_obj_val, candidate_obj_val = candidate_solution.get_obj_val(both=True)
                    true_obj_vals.append(true_obj_val)
                    # print(f"true_obj_val {true_obj_val}")
                    # print(f"\ncurrent_obj_val {current_obj_val}")
                    # print(f"heuristic_obj_val {obj_val}")
                    heuristic_obj_vals.append(candidate_obj_val)

                    if self._accept(candidate_obj_val, current_obj_val, temperature):

                        # IMPROVING
                        if candidate_obj_val > current_obj_val:
                            # print("current_obj_val: ", current_obj_val)
                            # print("candidate_obj_val: ", candidate_obj_val)
                            # NEW GLOBAL BEST
                            if candidate_obj_val > best_obj_val:
                                best_obj_val = candidate_obj_val
                                best_solution = (candidate_solution, true_obj_val)
                                output_text += str(counter) + f" New best solution globally: {candidate_obj_val}\n"
                                counter += 1
                                if MODE == "LNS":
                                    self._update_weight_record(_IS_BEST, destroy_heuristic, repair_heuristic)
                            # NEW LOCAL BEST
                            else:
                                output_text += str(counter) + f" New local best solution locally: {candidate_obj_val}\n"
                                counter += 1
                                if MODE == "LNS":
                                    self._update_weight_record(_IS_BETTER, destroy_heuristic, repair_heuristic)

                            MODE = "LOCAL_FULL"

                        # NON-IMPROVING BUT ACCEPTED
                        else:
                            p = np.exp(- (current_obj_val - candidate_obj_val) / temperature)
                            output_text += str(
                                counter) + f" New accepted solution: {candidate_obj_val}, acceptance probability was {p}, temperature was {temperature} \n"
                            counter += 1
                            if MODE == "LNS":
                                self._update_weight_record(_IS_ACCEPTED, destroy_heuristic, repair_heuristic)
                                MODE = "LOCAL_FIRST"
                            else:
                                MODE = "LNS"

                        # current_solution = copy.deepcopy(solution)
                        current_obj_val = candidate_obj_val
                        current_solution = candidate_solution

                    else:
                        output_text += str(counter) + " Not accepted solution\n"
                        counter += 1
                        MODE = "LNS"
                print(output_text)

                # print statistics
                # loop.set_description(f"Segment[{i}/{100}]")
                loop.set_postfix(current_obj_val=current_obj_val, best_obj_val=best_obj_val,
                                 best_true_obj_val=best_solution[1])

                temperature *= cooling_rate
                self._update_score_adjustment_parameters()

            finish = time.perf_counter()
            print(f"Finished in {round(finish - start, 2)} seconds(s)")

        except KeyboardInterrupt:
            print("Keyboard Interrupt")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        finally:
            self.best_solution = best_solution
            self.best_obj_val = best_obj_val
            plt.plot(heuristic_obj_vals, c="red", label="Heuristic obj. val")
            plt.plot(true_obj_vals, c="green", label="True obj. val")
            plt.suptitle('Objective value comparison')
            plt.xlabel('iteration')
            plt.ylabel('obj. val')
            plt.legend(bbox_to_anchor=(.75, 1.0))
            plt.show()
            # print(obj_vals)
            # print(self.operators_pairs)
            # print(self.repair_operators)
            print(self.operator_pairs)


    # best_solution.print_solution()

    def _initialize_operators(self):
        if self.num_employees < 3:
            '''operators = OrderedDict(
                {'random_greedy': 1.0, 'random_regret2': 1.0, 'random_charge': 3.0,
                 'worst_greedy': 1.0, 'worst_regret2': 1.0, 'worst_charge': 3.0,
                 'shaw_greedy': 1.0, 'shaw_regret2': 1.0, 'shaw_charge': 3.0,
                 'charge_greedy': 3.0, 'charge_regret2': 3.0, 'charge_charge': 20.0})'''
            operators = OrderedDict(
                {'random_greedy': 1.0, 'random_regret2': 1.0,
                 'worst_greedy': 1.0, 'worst_regret2': 1.0,
                 'shaw_greedy': 1.0, 'shaw_regret2': 1.0})
        elif self.num_employees < 4:
            '''operators = OrderedDict(
                {'random_greedy': 1.0, 'random_regret2': 1.0, 'random_regret3': 1.0, 'random_charge': 3.0,
                 'worst_greedy': 1.0, 'worst_regret2': 1.0, 'worst_regret3': 1.0, 'worst_charge': 3.0,
                 'shaw_greedy': 1.0, 'shaw_regret2': 1.0, 'shaw_regret3': 1.0, 'shaw_charge': 3.0,
                 'charge_greedy': 3.0, 'charge_regret2': 3.0, 'charge_regret3': 3.0, 'charge_charge': 20.0})'''
            operators = OrderedDict(
                {'random_greedy': 1.0, 'random_regret2': 1.0, 'random_regret3': 1.0,
                 'worst_greedy': 1.0, 'worst_regret2': 1.0, 'worst_regret3': 1.0,
                 'shaw_greedy': 1.0, 'shaw_regret2': 1.0, 'shaw_regret3': 1.0})
        else:
            '''operators = OrderedDict(
                {'random_greedy': 1.0, 'random_regret2': 1.0, 'random_regret3': 1.0, 'random_regret4': 1.0,
                 'random_charge': 3.0,
                 'worst_greedy': 1.0, 'worst_regret2': 1.0, 'worst_regret3': 1.0, 'worst_regret4': 1.0,
                 'worst_charge': 3.0,
                 'shaw_greedy': 1.0, 'shaw_regret2': 1.0, 'shaw_regret3': 1.0, 'shaw_regret4': 1.0,
                 'shaw_charge': 3.0,
                 'charge_greedy': 3.0, 'charge_regret2': 3.0, 'charge_regret3': 3.0, 'charge_regret4': 3.0,
                 'charge_charge': 20.0})
            '''
            operators = OrderedDict(
                {'random_greedy': 1.0, 'random_regret2': 1.0, 'random_regret3': 1.0, 'random_regret4': 1.0,
                 'worst_greedy': 1.0, 'worst_regret2': 1.0, 'worst_regret3': 1.0, 'worst_regret4': 1.0,
                 'shaw_greedy': 1.0, 'shaw_regret2': 1.0, 'shaw_regret3': 1.0, 'shaw_regret4': 1.0})

        return operators

    def _initialize_operator_records(self):
        if self.num_employees < 3:
            operators_record = OrderedDict(
                {'random_greedy': [1.0, 0], 'random_regret2': [1.0, 0], 'random_charge': [1.0, 0],
                 'worst_greedy': [1.0, 0], 'worst_regret2': [1.0, 0], 'worst_charge': [1.0, 0],
                 'shaw_greedy': [1.0, 0], 'shaw_regret2': [1.0, 0], 'shaw_charge': [1.0, 0],
                 'charge_greedy': [1.0, 0], 'charge_regret2': [1.0, 0], 'charge_charge': [1.0, 0]})
        elif self.num_employees < 4:
            operators_record = OrderedDict(
                {'random_greedy': [1.0, 0], 'random_regret2': [1.0, 0], 'random_regret3': [1.0, 0],
                 'random_charge': [1.0, 0],
                 'worst_greedy': [1.0, 0], 'worst_regret2': [1.0, 0], 'worst_regret3': [1.0, 0],
                 'worst_charge': [1.0, 0],
                 'shaw_greedy': [1.0, 0], 'shaw_regret2': [1.0, 0], 'shaw_regret3': [1.0, 0],
                 'shaw_charge': [1.0, 0],
                 'charge_greedy': [1.0, 0], 'charge_regret2': [1.0, 0], 'charge_regret3': [1.0, 0],
                 'charge_charge': [1.0, 0]})
        else:
            operators_record = OrderedDict(
                {'random_greedy': [1.0, 0], 'random_regret2': [1.0, 0], 'random_regret3': [1.0, 0],
                 'random_regret4': [1.0, 0], 'random_charge': [1.0, 0],
                 'worst_greedy': [1.0, 0], 'worst_regret2': [1.0, 0], 'worst_regret3': [1.0, 0],
                 'worst_regret4': [1.0, 0], 'worst_charge': [1.0, 0],
                 'shaw_greedy': [1.0, 0], 'shaw_regret2': [1.0, 0], 'shaw_regret3': [1.0, 0], 'shaw_regret4': [1.0, 0],
                 'shaw_charge': [1.0, 0],
                 'charge_greedy': [1.0, 0], 'charge_regret2': [1.0, 0], 'charge_regret3': [1.0, 0],
                 'charge_regret4': [1.0, 0], 'charge_charge': [1.0, 0]})

        return operators_record

    def _accept(self, new_obj_val, current_obj_val, temperature) -> bool:
        if new_obj_val > current_obj_val:
            acceptance_probability = 1
        else:
            p = np.exp(- (current_obj_val - new_obj_val) / temperature)
            acceptance_probability = p

        accept = acceptance_probability > random.random()
        return accept

    def _get_destroy_operator(self, solution, neighborhood_size, randomization_degree, world_instance) -> \
            (Destroy, str):
        w_sum = sum(w for o, w in self.operator_pairs.items())
        # dist = distribution
        w_dist = [w / w_sum for o, w in self.operator_pairs.items()]
        operator_pair = random.choices(list(self.operator_pairs), w_dist)[0]
        self.operators_record[operator_pair][1] += 1

        if operator_pair == "random_greedy" or operator_pair == "random_regret2" or operator_pair == "random_regret3" \
                or operator_pair == "random_regret4" or operator_pair == "random_charge":
            return RandomRemoval(solution, world_instance, neighborhood_size), operator_pair
        elif operator_pair == "worst_greedy" or operator_pair == "worst_regret2" or operator_pair == "worst_regret3" \
                or operator_pair == "worst_regret4" or operator_pair == "worst_charge":
            return WorstRemoval(solution, world_instance, neighborhood_size, randomization_degree), operator_pair
        elif operator_pair == "shaw_greedy" or operator_pair == "shaw_regret2" or operator_pair == "shaw_regret3" \
                or operator_pair == "shaw_regret4" or operator_pair == "shaw_charge":
            return ShawRemoval(solution, world_instance, neighborhood_size, randomization_degree), operator_pair
        elif operator_pair == "charge_greedy" or operator_pair == "charge_regret2" or \
                operator_pair == "charge_regret3" or operator_pair == "charge_regret4" \
                or operator_pair == "charge_charge":
            return ChargeRemoval(solution, world_instance, neighborhood_size), operator_pair
        else:
            exit("Destroy operator does not exist")

    def _get_repair_operator(self, destroyed_solution_object, unused_car_moves, world_instance, operator_pair) \
            -> Repair:

        if operator_pair == "random_greedy" or operator_pair == "worst_greedy" or operator_pair == "shaw_greedy" or \
                operator_pair == "charge_greedy":
            return GreedyInsertion(destroyed_solution_object, unused_car_moves, world_instance)
        elif operator_pair == "random_regret2" or operator_pair == "worst_regret2" or operator_pair == "shaw_regret2" \
                or operator_pair == "charge_regret2":
            return RegretInsertion(destroyed_solution_object, unused_car_moves, world_instance, regret_nr=2)
        elif operator_pair == "random_regret3" or operator_pair == "worst_regret3" or operator_pair == "shaw_regret3" \
                or operator_pair == "charge_regret3":
            return RegretInsertion(destroyed_solution_object, unused_car_moves, world_instance, regret_nr=3)
        elif operator_pair == "random_regret4" or operator_pair == "worst_regret4" or operator_pair == "shaw_regret4" \
                or operator_pair == "charge_regret4":
            return RegretInsertion(destroyed_solution_object, unused_car_moves, world_instance, regret_nr=4)
        elif operator_pair == "random_charge" or operator_pair == 'worst_charge' or operator_pair == 'shaw_charge':
            return ChargeInsertion(destroyed_solution_object, unused_car_moves, world_instance)
        elif operator_pair == 'charge_charge':
            return ChargeInsertion(destroyed_solution_object, unused_car_moves, world_instance,
                                   moves_not_insert=destroyed_solution_object.removed_moves_in_this_operation)
        else:
            exit(f"Repair operator {operator_pair} does not exist")

    def _update_weight_record(self, operator_score, destroy, repair):

        # DESTROY
        if isinstance(repair, RegretInsertion):
            if repair.regret_nr == 2:
                if isinstance(destroy, RandomRemoval):
                    self.operators_record['random_regret2'][0] += operator_score
                elif isinstance(destroy, WorstRemoval):
                    self.operators_record['worst_regret2'][0] += operator_score
                elif isinstance(destroy, ShawRemoval):
                    self.operators_record['shaw_regret2'][0] += operator_score
                elif isinstance(destroy, ChargeRemoval):
                    self.operators_record['charge_regret2'][0] += operator_score

            elif repair.regret_nr == 3:
                if isinstance(destroy, RandomRemoval):
                    self.operators_record['random_regret3'][0] += operator_score
                elif isinstance(destroy, WorstRemoval):
                    self.operators_record['worst_regret3'][0] += operator_score
                elif isinstance(destroy, ShawRemoval):
                    self.operators_record['shaw_regret3'][0] += operator_score
                elif isinstance(destroy, ChargeRemoval):
                    self.operators_record['charge_regret3'][0] += operator_score

            elif repair.regret_nr == 4:
                if isinstance(destroy, RandomRemoval):
                    self.operators_record['random_regret4'][0] += operator_score
                elif isinstance(destroy, WorstRemoval):
                    self.operators_record['worst_regret4'][0] += operator_score
                elif isinstance(destroy, ShawRemoval):
                    self.operators_record['shaw_regret4'][0] += operator_score
                elif isinstance(destroy, ChargeRemoval):
                    self.operators_record['charge_regret4'][0] += operator_score

        elif isinstance(repair, GreedyInsertion):
            if isinstance(destroy, RandomRemoval):
                self.operators_record['random_greedy'][0] += operator_score
            elif isinstance(destroy, WorstRemoval):
                self.operators_record['worst_greedy'][0] += operator_score
            elif isinstance(destroy, ShawRemoval):
                self.operators_record['shaw_greedy'][0] += operator_score
            elif isinstance(destroy, ChargeRemoval):
                self.operators_record['charge_greedy'][0] += operator_score

        elif isinstance(repair, ChargeRemoval):
            if isinstance(destroy, RandomRemoval):
                self.operators_record['random_charge'][0] += operator_score
            elif isinstance(destroy, WorstRemoval):
                self.operators_record['worst_charge'][0] += operator_score
            elif isinstance(destroy, ShawRemoval):
                self.operators_record['shaw_charge'][0] += operator_score
            elif isinstance(destroy, ChargeRemoval):
                self.operators_record['charge_charge'][0] += operator_score

    def _update_score_adjustment_parameters(self):
        reaction_factor = 0.1

        for k, v in self.operator_pairs.items():
            self.operator_pairs[k] = self.operator_pairs[k] * (1.0 - reaction_factor) + \
                                     safe_zero_division(reaction_factor, self.operators_record[k][1]) * \
                                     self.operators_record[k][0]
            self.operators_record[k][0] = self.operator_pairs[k]
            self.operators_record[k][0] = 0


if __name__ == "__main__":
    from pyinstrument import Profiler
    import time
    from best_objective_function import get_parking_nodes_in_out

    filename = "InstanceGenerator/InstanceFiles/20nodes/20-25-2-1_a"

    # code you want to profile


    try:
        profiler = Profiler()
        profiler.start()
        alns = ALNS(filename + ".pkl")
        profiler.stop()
        print("best solution")
        print("obj_val", alns.best_solution[1])
        alns.best_solution[0].print_solution()
        print(profiler.output_text(unicode=True, color=True))
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
    # print("\n############## Optimal solution ##############")
    # gi2 = GurobiInstance(filename + ".yaml")
    # run_model(gi2, time_limit=300)

