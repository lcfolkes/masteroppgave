import copy
import sys
import time
from collections import OrderedDict
import random

from termcolor import colored

from Heuristics.DestroyAndRepairHeuristics.destroy import Destroy, RandomRemoval, WorstRemoval, ShawRemoval, ChargeRemoval
from Heuristics.DestroyAndRepairHeuristics.repair import Repair, GreedyInsertion, RegretInsertion, ChargeInsertion
from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Gurobi.Model.run_model import run_model
from Heuristics.LocalSearch.local_search import LocalSearch
from Heuristics.helper_functions_heuristics import safe_zero_division, get_first_stage_solution, copy_solution_dict, \
    copy_unused_car_moves_2d_list, get_first_and_second_stage_solution
from Heuristics.best_construction_heuristic import ConstructionHeuristic
from Heuristics.heuristics_constants import HeuristicsConstants
#from Heuristics.parallel_construction_heuristic import ConstructionHeuristic
from path_manager import path_to_src
import numpy as np
import traceback
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

os.chdir(path_to_src)

_IS_BEST = HeuristicsConstants.BEST
_IS_BETTER = HeuristicsConstants.BETTER
_IS_ACCEPTED = HeuristicsConstants.ACCEPTED

'''

_IS_BEST (sigma_1): The last remove-insert operation resulted in a new global best solution
_IS_BETTER (sigma_2): The last remove-insert operation resulted in a solution that has not been accepted before.
         The cost of the new solution is better than the cost of the current solution.
_IS_ACCEPTED (sigma_3): The last remove-insert operation resulted in a solution that has not been accepted before. The cost of
            the new solution is worse than the cost of current solution, but the solution was accepted.
_IS_REJECTED
'''


class ALNS():

    def __init__(self, filename, acceptance_percentage=1.0):
        self.filename = filename

        self.best_solution = None
        self.best_solutions = None
        self.best_obj_val = 0

        self.solution = ConstructionHeuristic(self.filename, acceptance_percentage)
        self._num_employees = len(self.solution.employees)
        self._num_first_stage_tasks = self.solution.num_first_stage_tasks
        self._feasibility_checker = self.solution.feasibility_checker
        self._world_instance = self.solution.world_instance
        self.operator_pairs = self._initialize_operators()
        self.operators_record = self._initialize_operator_records()

    def _initialize_new_iteration(self, current_unused_car_moves, current_solution):

        candidate_unused_car_moves = copy_unused_car_moves_2d_list(current_unused_car_moves)

        candidate_solution = copy_solution_dict(current_solution)

        for cn in self._world_instance.charging_nodes:
            cn.reset()
        return candidate_unused_car_moves, candidate_solution

    def run(self, verbose=True):
        start = time.perf_counter()
        visited_hash_keys = set()

        iterations_alns = HeuristicsConstants.ITERATIONS_ALNS
        iterations_segment = HeuristicsConstants.ITERATIONS_SEGMENT
        time_limit = HeuristicsConstants.TIME_LIMIT

        finish_times_segments = []
        first_checkpoint = HeuristicsConstants.FIRST_CHECKPOINT
        second_checkpoint = HeuristicsConstants.SECOND_CHECKPOINT
        first_checkpoint_reached = False
        second_checkpoint_reached = False
        obj_val_first_checkpoint = None
        heur_val_first_checkpoint = None
        obj_val_second_checkpoint = None
        heur_val_second_checkpoint = None
        finish = None

        self.solution.construct(verbose=verbose)
        self.solution.print_solution()
        true_obj_val, best_obj_val = self.solution.get_obj_val(both=True)
        current_obj_val = best_obj_val
        true_obj_vals = [true_obj_val]
        heuristic_obj_vals = [best_obj_val]
        iterations = [0]
        best_solution = (copy_solution_dict(self.solution.assigned_car_moves), true_obj_val)
        if verbose:
            self.solution.print_solution()
            print(f"Construction heuristic true obj. val {true_obj_val}")
            print(f"Heuristic obj. val {heuristic_obj_vals[0]}")
            print(f"Heuristic obj. val {heuristic_obj_vals[0]}")
        current_solution = copy_solution_dict(self.solution.assigned_car_moves)
        current_unused_car_moves = copy_unused_car_moves_2d_list(self.solution.unused_car_moves)
        visited_hash_keys.add(self.solution.hash_key)
        # MODE = "LOCAL"
        MODE = "LOCAL_FIRST"

        # temperature = 1000  # Start temperature must be set differently. High temperature --> more randomness
        temperature = (-np.abs(heuristic_obj_vals[0])) * 0.05 / np.log(0.5)
        # cooling_rate = 0.5  # cooling_rate in (0,1)
        cooling_rate = np.exp(np.log(0.002) / 200)

        # SEGMENTS
        try:
            for i in range(iterations_alns):
                # print(i)

                # print(f"Iteration {i * 10}")
                # print(f"Best objective value {best_solution[1]}")
                # print(f"Best heuristic objective value {max(heuristic_obj_vals)}")
                loop = tqdm(range(iterations_segment), total=iterations_segment, leave=True)
                loop.set_description(f"Segment[{i}/{iterations_alns}]")
                loop.set_postfix(current_obj_val=current_obj_val, best_obj_val=best_obj_val,
                             best_true_obj_val=best_solution[1])

                output_text = "\n"
                counter = 1

                for j in loop:
                    # print(f"Iteration {i*10 + j}")
                    candidate_unused_car_moves, candidate_solution = self._initialize_new_iteration(
                        current_unused_car_moves, current_solution)

                    if MODE == "LOCAL_FIRST":
                        # print("\n----- LOCAL SEARCH FIRST BEST -----")
                        local_search = LocalSearch(candidate_solution,
                                                   self._num_first_stage_tasks,
                                                   self._feasibility_checker)
                        local_search.search("best_first")
                        self.solution.rebuild(local_search.solution, "second_stage")
                        visited_hash_keys.update(local_search.visited_list)

                    elif MODE == "LOCAL_FULL":
                        # print("\n----- LOCAL SEARCH FULL -----")
                        local_search = LocalSearch(candidate_solution,
                                                   self._num_first_stage_tasks,
                                                   self._feasibility_checker)
                        local_search.search("full")
                        self.solution.rebuild(local_search.solution, "second_stage")
                        visited_hash_keys.update(local_search.visited_list)


                    elif MODE == "LNS":
                        # print("\n----- LARGE NEIGHBORHOOD SEARCH -----")
                        destroy_heuristic, operator_pair = self._get_destroy_operator(
                            solution=candidate_solution,
                            world_instance=self._world_instance)
                        destroy_heuristic.destroy()

                        # print(f"Destroy: {destroy_heuristic}\n{destroy_heuristic.solution}\n{destroy_heuristic.to_string()}")
                        # print(destroy)

                        repair_heuristic = self._get_repair_operator(destroyed_solution_object=destroy_heuristic,
                                                                     unused_car_moves=candidate_unused_car_moves,
                                                                     world_instance=self._world_instance,
                                                                     operator_pair=operator_pair)
                        repair_heuristic.repair()

                        # print("Repair: ", repair_heuristic, repair_heuristic.solution)
                        # repair_heuristic.to_string()
                        hash_key = repair_heuristic.hash_key
                        if hash_key in visited_hash_keys:
                            output_text += str(counter) + f" {colored('Already visited solution', 'yellow')}\n"
                            counter += 1
                            continue
                        visited_hash_keys.add(hash_key)
                        self.solution.rebuild(repair_heuristic.solution)


                    true_obj_val, candidate_obj_val = self.solution.get_obj_val(both=True)
                    true_obj_vals.append(true_obj_val)
                    heuristic_obj_vals.append(candidate_obj_val)

                    iterations.append((j + 1) + (i * iterations_segment))

                    if self._accept(candidate_obj_val, current_obj_val, temperature):

                        # IMPROVING
                        if round(candidate_obj_val, 2) > round(current_obj_val, 2):
                            # print("current_obj_val: ", current_obj_val)
                            # print("candidate_obj_val: ", candidate_obj_val)
                            # NEW GLOBAL BEST
                            if round(candidate_obj_val, 2) > round(best_obj_val, 2):
                                best_obj_val = candidate_obj_val
                                best_solution = (copy_solution_dict(self.solution.assigned_car_moves), true_obj_val)
                                output_text += str(
                                    counter) + f" {colored('New best solution globally:', 'green')} {colored(round(candidate_obj_val, 2), 'green')}\n"
                                counter += 1
                                # print("NEW GLOBAL SOLUTION")
                                # self.solution.print_solution()
                                if MODE == "LNS":
                                    self._update_weight_record(_IS_BEST, destroy_heuristic, repair_heuristic)
                            # NEW LOCAL BEST
                            else:
                                output_text += str(
                                    counter) + f" {colored('New best solution locally:', 'blue')} {colored(round(candidate_obj_val, 2), 'blue')}\n"
                                counter += 1
                                if MODE == "LNS":
                                    self._update_weight_record(_IS_BETTER, destroy_heuristic, repair_heuristic)

                            MODE = "LOCAL_FULL"

                        # NON-IMPROVING BUT ACCEPTED
                        else:
                            p = np.exp(- (current_obj_val - candidate_obj_val) / temperature)
                            output_text += str(
                                counter) + f" New accepted solution: {round(candidate_obj_val, 2)}, acceptance probability was {round(p, 2)}, temperature was {round(temperature, 2)} \n"
                            counter += 1
                            if MODE == "LNS":
                                self._update_weight_record(_IS_ACCEPTED, destroy_heuristic, repair_heuristic)
                                MODE = "LOCAL_FIRST"
                            else:
                                MODE = "LNS"

                        # current_solution = copy.deepcopy(solution)
                        current_obj_val = candidate_obj_val
                        current_solution = candidate_solution
                        current_unused_car_moves = candidate_unused_car_moves

                    else:
                        output_text += str(
                            counter) + f" {colored('Not accepted solution:', 'red')} {colored(round(candidate_obj_val, 2), 'red')}\n"
                        counter += 1
                        MODE = "LNS"

                    if time.perf_counter() > time_limit:
                        print("Time limit reached!")
                        finish = time.perf_counter()
                        return

                    if time.perf_counter() > first_checkpoint and not first_checkpoint_reached:
                        first_checkpoint_reached = True
                        heur_val_first_checkpoint = best_obj_val
                        obj_val_first_checkpoint = best_solution[1]
                    if time.perf_counter() > second_checkpoint and not second_checkpoint_reached:
                        second_checkpoint_reached = True
                        heur_val_second_checkpoint = best_obj_val
                        obj_val_second_checkpoint = best_solution[1]

                print(output_text)
                time_segment = time.perf_counter()
                finish_times_segments.append(time_segment)

                if verbose:
                    print(output_text)
                loop.set_postfix(current_obj_val=current_obj_val, best_obj_val=best_obj_val,
                             best_true_obj_val=best_solution[1])

                temperature *= cooling_rate
                self._update_score_adjustment_parameters()

            finish = time.perf_counter()
            if verbose:
                print(f"Finished in {round(finish - start, 2)} seconds(s)")

        except KeyboardInterrupt:
            print("Keyboard Interrupt")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print(traceback.format_exc())
            raise
        finally:
            if finish is None:
                finish = time.perf_counter()
            self.best_solution = best_solution
            self.best_obj_val = best_obj_val
            '''
            plt.plot(heuristic_obj_vals, c="red", label="Heuristic obj. val")
            plt.plot(true_obj_vals, c="green", label="True obj. val")
            plt.suptitle('Objective value comparison')
            plt.xlabel('iteration')
            plt.ylabel('obj. val')
            plt.legend(bbox_to_anchor=(.75, 1.0))
            plt.show()
            '''

            f, (ax1, ax2) = plt.subplots(2, 1)
            #print("iterations", iterations)
            #print("true_obj_vals", true_obj_vals)
            obj_plot = ax1.plot(iterations, true_obj_vals, c="green", label="True obj. val")
            heur_plot = ax2.plot(iterations, heuristic_obj_vals, c="red", label="Heur. obj. val")

            ax1.set_title("True obj. value")
            ax2.set_title("Heuristic obj. value")

            f.subplots_adjust(hspace=0.4)

            lns = obj_plot + heur_plot
            labs = [l.get_label() for l in lns]
            plt.legend(lns, labs, bbox_to_anchor=(1.03, 2.7),
                       fancybox=True, shadow=False)

            f.show()

            # Strings to save to file
            obj_val_txt = f"Objective value: {str(best_solution[1])}\n"
            heur_val_txt = f"Heuristic value: {str(best_obj_val)}\n"
            first_checkpoint_txt1 = f"Objective value after {first_checkpoint} s: {obj_val_first_checkpoint}\n"
            first_checkpoint_txt2 = f"Heuristic value after {first_checkpoint} s: {heur_val_first_checkpoint}\n"
            second_checkpoint_txt1 = f"Objective value after {second_checkpoint} s: {obj_val_second_checkpoint}\n"
            second_checkpoint_txt2 = f"Heuristic value after {second_checkpoint} s: {heur_val_second_checkpoint}\n"
            time_spent_txt = f"Time used: {finish}\n"
            finish_times_segments_txt = f"Finish time segments:\n{finish_times_segments}\n"
            iterations_done_txt = f"Iterations completed: {len(finish_times_segments) * iterations_segment} iterations in {len(finish_times_segments)} segments\n\n"

            # Write to file
            f = open(filename + "_results.txt", "a")
            f.writelines([obj_val_txt, heur_val_txt, first_checkpoint_txt1, first_checkpoint_txt2,
                          second_checkpoint_txt1, second_checkpoint_txt2, time_spent_txt, finish_times_segments_txt,
                          iterations_done_txt])
            f.close()

            return best_solution[1]

    def _initialize_operators(self):
        if self._num_employees < 3:
            operators = OrderedDict(
                {'random_greedy': 1.0, 'random_regret2': 1.0, 'random_charge': 3.0,
                 'worst_greedy': 1.0, 'worst_regret2': 1.0, 'worst_charge': 3.0,
                 'shaw_greedy': 1.0, 'shaw_regret2': 1.0, 'shaw_charge': 3.0,
                 'charge_greedy': 3.0, 'charge_regret2': 3.0, 'charge_charge': 20.0})

        elif self._num_employees < 4:
            operators = OrderedDict(
                {'random_greedy': 1.0, 'random_regret2': 1.0, 'random_regret3': 1.0, 'random_charge': 3.0,
                 'worst_greedy': 1.0, 'worst_regret2': 1.0, 'worst_regret3': 1.0, 'worst_charge': 3.0,
                 'shaw_greedy': 1.0, 'shaw_regret2': 1.0, 'shaw_regret3': 1.0, 'shaw_charge': 3.0,
                 'charge_greedy': 3.0, 'charge_regret2': 3.0, 'charge_regret3': 3.0, 'charge_charge': 20.0})

        else:
            operators = OrderedDict(
                {'random_greedy': 1.0, 'random_regret2': 1.0, 'random_regret3': 1.0, 'random_regret4': 1.0,
                 'random_charge': 3.0,
                 'worst_greedy': 1.0, 'worst_regret2': 1.0, 'worst_regret3': 1.0, 'worst_regret4': 1.0,
                 'worst_charge': 3.0,
                 'shaw_greedy': 1.0, 'shaw_regret2': 1.0, 'shaw_regret3': 1.0, 'shaw_regret4': 1.0,
                 'shaw_charge': 3.0,
                 'charge_greedy': 3.0, 'charge_regret2': 3.0, 'charge_regret3': 3.0, 'charge_regret4': 3.0,
                 'charge_charge': 20.0})

        return operators

    def _initialize_operator_records(self):
        if self._num_employees < 3:
            operators_record = OrderedDict(
                {'random_greedy': [1.0, 0], 'random_regret2': [1.0, 0], 'random_charge': [1.0, 0],
                 'worst_greedy': [1.0, 0], 'worst_regret2': [1.0, 0], 'worst_charge': [1.0, 0],
                 'shaw_greedy': [1.0, 0], 'shaw_regret2': [1.0, 0], 'shaw_charge': [1.0, 0],
                 'charge_greedy': [1.0, 0], 'charge_regret2': [1.0, 0], 'charge_charge': [1.0, 0]})
        elif self._num_employees < 4:
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
        if round(new_obj_val, 2) > round(current_obj_val, 2):
            acceptance_probability = 1
        else:
            p = np.exp(- (current_obj_val - new_obj_val) / temperature)
            acceptance_probability = p

        accept = acceptance_probability > random.random()
        return accept

    def _get_destroy_operator(self, solution, world_instance) -> \
            (Destroy, str):

        neighborhood_size = int(self._num_employees * self._num_first_stage_tasks * random.uniform(
            HeuristicsConstants.DESTROY_REPAIR_FACTOR[0], HeuristicsConstants.DESTROY_REPAIR_FACTOR[1]))
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
            return WorstRemoval(solution, world_instance, neighborhood_size,
                                HeuristicsConstants.DETERMINISM_PARAMETER_WORST), operator_pair
        elif operator_pair == "shaw_greedy" or operator_pair == "shaw_regret2" or operator_pair == "shaw_regret3" \
                or operator_pair == "shaw_regret4" or operator_pair == "shaw_charge":
            return ShawRemoval(solution, world_instance, neighborhood_size,
                               HeuristicsConstants.DETERMINISM_PARAMETER_RELATED), operator_pair
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
        for k, v in self.operator_pairs.items():
            self.operator_pairs[k] = self.operator_pairs[k] * (1.0 - HeuristicsConstants.REWARD_DECAY_PARAMETER) + \
                                     safe_zero_division(HeuristicsConstants.REWARD_DECAY_PARAMETER, self.operators_record[k][1]) * \
                                     self.operators_record[k][0]

            if self.operator_pairs[k] < HeuristicsConstants.LOWER_THRESHOLD:
                self.operator_pairs[k] = HeuristicsConstants.LOWER_THRESHOLD
            self.operators_record[k][0] = self.operator_pairs[k]
            self.operators_record[k][0] = 0


if __name__ == "__main__":
    from pyinstrument import Profiler
    filename = "InstanceGenerator/InstanceFiles/30nodes/30-10-2-1_a"

    try:
        #profiler = Profiler()
        #profiler.start()
        alns = ALNS(filename + ".pkl", acceptance_percentage=0.7)
        alns.run()

        #profiler.stop()
        print("best solution")
        print("obj_val", alns.best_solution[1])
        alns.solution.rebuild(alns.best_solution[0], "second_stage")
        alns.solution.print_solution()
        #print(profiler.output_text(unicode=True, color=True))

        print("\n############## Evaluate solution ##############")
        gi = GurobiInstance(filename + ".yaml", employees=alns.solution.employees, optimize=False)
        run_model(gi)
        '''
        print("\n############## Reoptimized solution ##############")
        gi = GurobiInstance(filename + ".yaml", employees=alns.solution.employees, optimize=True)
        run_model(gi)
        print("\n############## Optimal solution ##############")
        gi2 = GurobiInstance(filename + ".yaml")
        run_model(gi2, time_limit=300)'''
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
