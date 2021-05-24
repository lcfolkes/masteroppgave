import sys
import time
from collections import OrderedDict
import random
from termcolor import colored
from Heuristics.DestroyAndRepairHeuristics.destroy import Destroy, RandomRemoval, WorstRemoval, ShawRemoval, ChargeRemoval
from Heuristics.DestroyAndRepairHeuristics.repair import Repair, RegretInsertion, ChargeInsertion, \
    GreedyRandomInsertion
from Heuristics.LocalSearch.local_search import LocalSearch
from Heuristics.helper_functions_heuristics import safe_zero_division, copy_solution_dict, \
    copy_unused_car_moves_2d_list
from Heuristics.ALNS.construction_heuristic import ConstructionHeuristic
from Heuristics.ALNS.heuristics_constants import HeuristicsConstants
from path_manager import path_to_src
import numpy as np
import traceback
import os
from datetime import datetime
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

    def __init__(self, filename, param):
        self.filename = filename
        self.best_solution = None
        self.best_solutions = None
        self.best_obj_val = 0
        self.solution = ConstructionHeuristic(self.filename)
        self._num_employees = len(self.solution.employees)
        self._num_first_stage_tasks = self.solution.num_first_stage_tasks
        self._feasibility_checker = self.solution.feasibility_checker
        self._world_instance = self.solution.world_instance
        self.operator_pairs = self._initialize_operators()
        self.operators_record = self._initialize_operator_records()
        self.determinism_parameter = param

    def _initialize_new_iteration(self, current_unused_car_moves, current_solution):
        candidate_unused_car_moves = copy_unused_car_moves_2d_list(current_unused_car_moves)

        candidate_solution = copy_solution_dict(current_solution)

        for cn in self._world_instance.charging_nodes:
            cn.reset()
        return candidate_unused_car_moves, candidate_solution

    def run(self, run=0, verbose=False):

        start = time.perf_counter()
        visited_hash_keys = set()

        iterations_alns = HeuristicsConstants.ITERATIONS_ALNS
        iterations_segment = HeuristicsConstants.ITERATIONS_SEGMENT
        time_limit = HeuristicsConstants.TIME_LIMIT

        finish_times_segments = []
        #first_checkpoint = HeuristicsConstants.FIRST_CHECKPOINT
        #second_checkpoint = HeuristicsConstants.SECOND_CHECKPOINT


        checkpoints_reached = [False for _ in range(10)]
        time_checkpoints = [60*(x+1) for x in range(10)]

        true_obj_val_checkpoints = [None for _ in range(10)]
        heur_obj_val_checkpoints = [None for _ in range(10)]

        construction_true_obj_val = None
        construction_heur_obj_val = None

        best_obj_val_found_time = None

        finish = None

        self.solution.construct(verbose=verbose)
        best_obj_val_found_time = time.perf_counter() - start
        true_obj_val, best_obj_val = self.solution.get_obj_val(both=True)
        construction_heur_obj_val = best_obj_val
        construction_true_obj_val = true_obj_val

        current_obj_val = best_obj_val
        true_obj_vals = [true_obj_val]
        heuristic_obj_vals = [best_obj_val]
        iterations = [0]
        best_solution = (copy_solution_dict(self.solution.assigned_car_moves), true_obj_val, self.solution.num_charging_moves)
        if verbose:
            self.solution.print_solution()
            print(f"Construction heuristic true obj. val {true_obj_val}")
            print(f"Heuristic obj. val {heuristic_obj_vals[0]}")
        current_solution = copy_solution_dict(self.solution.assigned_car_moves)
        current_unused_car_moves = copy_unused_car_moves_2d_list(self.solution.unused_car_moves)
        visited_hash_keys.add(self.solution.hash_key)
        # MODE = "LOCAL"
        MODE = "LOCAL_FIRST"

        temperature = (-np.abs(heuristic_obj_vals[0])) * 0.05 / np.log(0.5)
        #cooling_rate = 0.5  # cooling_rate in (0,1)
        cooling_rate = np.exp(np.log(0.02) / iterations_alns*iterations_segment)

        # SEGMENTS
        try:
            for i_alns in range(iterations_alns):
                loop = tqdm(range(iterations_segment), total=iterations_segment, leave=True, ascii=True)
                loop.set_description(f"Segment[{i_alns}/{iterations_alns}]")
                loop.set_postfix(current_obj_val=current_obj_val, best_obj_val=best_obj_val,
                             best_true_obj_val=best_solution[1])

                output_text = "\n"
                counter = 1

                for i_segment in loop:
                    # print(f"Iteration {i*10 + j}")
                    candidate_unused_car_moves, candidate_solution = self._initialize_new_iteration(
                        current_unused_car_moves, current_solution)

                    if MODE == "LOCAL_FIRST":
                        #print("\n----- LOCAL SEARCH FIRST BEST -----")
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
                        #print("\n----- LARGE NEIGHBORHOOD SEARCH -----")
                        destroy_heuristic, operator_pair = self._get_destroy_operator(
                            solution=candidate_solution,
                            world_instance=self._world_instance)
                        #print(f"Destroy before: {destroy_heuristic}\n{destroy_heuristic.to_string()}")
                        destroy_heuristic.destroy()


                        #print(f"Destroy: {destroy_heuristic}\n{destroy_heuristic.to_string()}")

                        repair_heuristic = self._get_repair_operator(destroyed_solution_object=destroy_heuristic,
                                                                     unused_car_moves=candidate_unused_car_moves,
                                                                     world_instance=self._world_instance,
                                                                     operator_pair=operator_pair)
                        repair_heuristic.repair()

                        #print(f"Repair: {repair_heuristic} {repair_heuristic.to_string()}")


                        hash_key = repair_heuristic.hash_key
                        if hash_key in visited_hash_keys:
                            output_text += str(counter) + f" {colored('Already visited solution', 'yellow')}\n"
                            counter += 1
                            continue
                        visited_hash_keys.add(hash_key)
                        self.solution.rebuild(repair_heuristic.solution)
                        #self.solution.print_solution()


                    true_obj_val, candidate_obj_val = self.solution.get_obj_val(both=True)
                    true_obj_vals.append(true_obj_val)
                    heuristic_obj_vals.append(candidate_obj_val)

                    iterations.append((i_segment + 1) + (i_alns * iterations_segment))

                    if self._accept(candidate_obj_val, current_obj_val, temperature):

                        # IMPROVING
                        if round(candidate_obj_val, 2) > round(current_obj_val, 2):
                            # print("current_obj_val: ", current_obj_val)
                            # print("candidate_obj_val: ", candidate_obj_val)
                            # NEW GLOBAL BEST
                            if round(candidate_obj_val, 2) > round(best_obj_val, 2):
                                current_time = time.perf_counter() - start
                                best_obj_val_found_time = current_time
                                best_obj_val = candidate_obj_val

                                best_solution = (copy_solution_dict(self.solution.assigned_car_moves), true_obj_val, self.solution.num_charging_moves)
                                output_text += str(
                                    counter) + f" {colored('New best solution globally:', 'green')} {colored(round(candidate_obj_val, 2), 'green')}{colored(', found by ', 'green')}{colored(MODE, 'green')}\n"
                                counter += 1
                                # print("NEW GLOBAL SOLUTION")
                                # self.solution.print_solution()
                                if MODE == "LNS":
                                    self._update_weight_record(_IS_BEST, destroy_heuristic, repair_heuristic)
                            # NEW LOCAL BEST
                            else:
                                output_text += str(
                                    counter) + f" {colored('New best solution locally:', 'blue')} {colored(round(candidate_obj_val, 2), 'blue')}{colored(', found by ', 'blue')}{colored(MODE, 'blue')}\n"
                                counter += 1
                                if MODE == "LNS":
                                    self._update_weight_record(_IS_BETTER, destroy_heuristic, repair_heuristic)

                            MODE = "LOCAL_FULL"

                        # NON-IMPROVING BUT ACCEPTED
                        else:
                            p = np.exp(- safe_zero_division(current_obj_val - candidate_obj_val, temperature))
                            output_text += str(
                                counter) + f"{colored(' New accepted solution: ', 'magenta')}{colored(round(candidate_obj_val, 2), 'magenta')}{colored(', found by ', 'magenta')}{colored(MODE, 'magenta')}{colored(', acceptance probability was' , 'magenta')} {colored(round(p, 2), 'magenta')}{colored(', temperature was ', 'magenta')}{colored(round(temperature, 2), 'magenta')} \n"
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
                            counter) + f" {colored('Not accepted solution:', 'red')} {colored(round(candidate_obj_val, 2), 'red')}{colored(', found by ', 'red')}{colored(MODE, 'red')}\n"
                        counter += 1
                        MODE = "LNS"

                    current_time = time.perf_counter() - start
                    if current_time > time_limit:
                        print("Time limit reached!")
                        finish = current_time
                        return

                    for i in range(len(time_checkpoints)):
                        if current_time >= time_checkpoints[i] and not checkpoints_reached[i]:
                            checkpoints_reached[i] = True
                            true_obj_val_checkpoints[i] = best_solution[1]
                            heur_obj_val_checkpoints[i] = best_obj_val
                            break

                    temperature *= cooling_rate

                time_segment = time.perf_counter() - start
                finish_times_segments.append(time_segment)

                if verbose:
                    print(output_text)
                    output_text = "\n"

                loop.set_postfix(current_obj_val=current_obj_val, best_obj_val=best_obj_val,
                             best_true_obj_val=best_solution[1])

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
            '''
            # Strings to save to file
            dateTimeObj = datetime.now()
            timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
            run_txt = f"\nRun: {str(run)}\n"
            date_time_txt = f"DateTime: {timestampStr}\n"
            obj_val_found_txt = f"Best objective value found after (s): {best_obj_val_found_time}\n"
            obj_val_txt = f"Objective value: {str(best_solution[1])}\n"
            heur_val_txt = f"Heuristic value: {str(best_obj_val)}\n"
            charging_txt = f"Cars charged: {str(best_solution[2])}\nCars in need of charging: {self.solution.num_cars_in_need}\n"
            construction_heur_txt = f"Construction heuristic, true objective value: {str(construction_true_obj_val)}\n"
            construction_heur_txt += f"Construction heuristic, heuristic objective value: {str(construction_heur_obj_val)}\n"
            check_points_txt = ""
            for t in range(len(time_checkpoints)):
                check_points_txt += f"Objective value after {time_checkpoints[t]} s: {true_obj_val_checkpoints[t]}\n"
                check_points_txt += f"Heuristic value after {time_checkpoints[t]} s: {heur_obj_val_checkpoints[t]}\n"

            time_spent_txt = f"Time used: {finish}\n"
            finish_times_segments_txt = f"Finish time segments:\n{finish_times_segments}\n"
            iterations_done_txt = f"Iterations completed: {i_alns*iterations_segment + i_segment} iterations in {i_alns+1} segments\n"
            parameter_tuning_txt = f"Acceptance percentage: {self.solution.acceptance_percentage}\n" \
                                   f"Travel time threshold: {self.solution.travel_time_threshold}\n" \
                                   f"Neighborhood Size: {HeuristicsConstants.DESTROY_REPAIR_FACTOR}\n" \
                                   f"Reward decay parameter: {HeuristicsConstants.REWARD_DECAY_PARAMETER}\n" \
                                   f"Determinism Worst: {HeuristicsConstants.DETERMINISM_PARAMETER_WORST}\n" \
                                   f"Determinism Related: {HeuristicsConstants.DETERMINISM_PARAMETER_RELATED}\n" \
                                   f"Determinism Greedy: {self.determinism_parameter}\n" \
                                   f"Adaptive Weight Rewards (Best, Better, Accepted): ({_IS_BEST}, {_IS_BETTER}, {_IS_ACCEPTED})\n\n"

            # Write to file
            test_dir = "./Testing/Results/" + self.filename.split('/')[-2] + "/"
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            filename = self.filename.split('/')[-1].split('.')[0]
            filepath = test_dir + filename
            f = open(filepath + "_results.txt", "a")
            f.writelines([run_txt, date_time_txt, obj_val_found_txt, obj_val_txt, heur_val_txt, charging_txt, construction_heur_txt,
                          check_points_txt, time_spent_txt, iterations_done_txt, parameter_tuning_txt])
            f.close()

            if verbose:
                #print("BEST SOLUTION")
                #print(self.best_solution[0])
                self.solution.rebuild(self.best_solution[0], "second_stage")
                self.solution.print_solution()
            return f"obj_val: {best_solution[1]}, n_iterations: {i_alns*iterations_segment + i_segment}"

    def _initialize_operators(self):
        if self._num_employees < 3:
            operators = OrderedDict(
                {'random_greedy': 1.0,  'random_regret2': 1.0, 'random_charge': 1.0,
                 'worst_greedy': 1.0,  'worst_regret2': 1.0, 'worst_charge': 1.0,
                 'shaw_greedy': 1.0,      'shaw_regret2': 1.0, 'shaw_charge': 1.0,
                 'charge_greedy': 1.0,  'charge_regret2': 1.0, 'charge_charge': 1.0})

        elif self._num_employees < 4:
            operators = OrderedDict(
                {'random_greedy': 1.0,  'random_regret2': 1.0, 'random_regret3': 1.0, 'random_charge': .0,
                 'worst_greedy': 1.0, 'worst_regret2': 1.0, 'worst_regret3': 1.0, 'worst_charge': 1.0,
                 'shaw_greedy': 1.0, 'shaw_regret2': 1.0, 'shaw_regret3': 1.0, 'shaw_charge': 1.0,
                 'charge_greedy': 1.0, 'charge_regret2': 1.0, 'charge_regret3': 1.0, 'charge_charge': .0})

        else:
            operators = OrderedDict(
                {'random_greedy': 1.0, 'random_regret2': 1.0, 'random_regret3': 1.0, 'random_regret4': 1.0, 'random_charge': 1.0,
                 'worst_greedy': 1.0, 'worst_regret2': 1.0, 'worst_regret3': 1.0, 'worst_regret4': 1.0, 'worst_charge': 1.0,
                 'shaw_greedy': 1.0, 'shaw_regret2': 1.0, 'shaw_regret3': 1.0, 'shaw_regret4': 1.0, 'shaw_charge': 1.0,
                 'charge_greedy': 1.0, 'charge_regret2': 1.0, 'charge_regret3': 1.0, 'charge_regret4': 1.0, 'charge_charge': 1.0})

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
                {'random_greedy': [1.0, 0], 'random_regret2': [1.0, 0], 'random_regret3': [1.0, 0], 'random_charge': [1.0, 0],
                 'worst_greedy': [1.0, 0], 'worst_regret2': [1.0, 0], 'worst_regret3': [1.0, 0], 'worst_charge': [1.0, 0],
                 'shaw_greedy': [1.0, 0],  'shaw_regret2': [1.0, 0], 'shaw_regret3': [1.0, 0], 'shaw_charge': [1.0, 0],
                 'charge_greedy': [1.0, 0], 'charge_regret2': [1.0, 0], 'charge_regret3': [1.0, 0], 'charge_charge': [1.0, 0]})
        else:
            operators_record = OrderedDict(
                {'random_greedy': [1.0, 0], 'random_regret2': [1.0, 0], 'random_regret3': [1.0, 0], 'random_regret4': [1.0, 0], 'random_charge': [1.0, 0],
                 'worst_greedy': [1.0, 0], 'worst_regret2': [1.0, 0], 'worst_regret3': [1.0, 0], 'worst_regret4': [1.0, 0], 'worst_charge': [1.0, 0],
                 'shaw_greedy': [1.0, 0], 'shaw_regret2': [1.0, 0], 'shaw_regret3': [1.0, 0], 'shaw_regret4': [1.0, 0], 'shaw_charge': [1.0, 0],
                 'charge_greedy': [1.0, 0], 'charge_regret2': [1.0, 0], 'charge_regret3': [1.0, 0], 'charge_regret4': [1.0, 0], 'charge_charge': [1.0, 0]})

        return operators_record

    def _accept(self, new_obj_val, current_obj_val, temperature) -> bool:
        if round(new_obj_val, 2) > round(current_obj_val, 2):
            acceptance_probability = 1
        else:
            p = np.exp(- safe_zero_division(current_obj_val - new_obj_val, temperature))
            acceptance_probability = p

        accept = acceptance_probability > random.random()
        return accept

    def _get_destroy_operator(self, solution, world_instance) -> \
            (Destroy, str):

        neighborhood_size = random.uniform(HeuristicsConstants.DESTROY_REPAIR_FACTOR[0], HeuristicsConstants.DESTROY_REPAIR_FACTOR[1])
        if neighborhood_size == 0:
            neighborhood_size = 1
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
            return GreedyRandomInsertion(destroyed_solution_object, unused_car_moves, world_instance,
                                         self.determinism_parameter)
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

        elif isinstance(repair, GreedyRandomInsertion):
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
    filename = "./InstanceGenerator/InstanceFiles/25nodes/25-25-2-1_b"

    try:
        #profiler = Profiler()
        #profiler.start()
        alns = ALNS(filename + ".pkl",0.5)
        alns.run(verbose=True)

        #profiler.stop()
        #alns.solution.rebuild(alns.best_solution[0], "second_stage")
        #alns.solution.print_solution()
        #print(profiler.output_text(unicode=True, color=True))

        '''
        print("\n############## Evaluate solution ##############")
        gi = GurobiInstance(filename + ".yaml", employees=alns.solution.employees)
        run_model(gi)


        print("\n############## Optimal solution ##############")
        gi = GurobiInstance(filename + ".yaml")
        run_model(gi)
        print("\n############## Evaluate solution ##############")
        gi = GurobiInstance(filename + ".yaml", employees=alns.solution.employees)
        run_model(gi)
        
        print("\n############## Reoptimized solution ##############")
        gi = GurobiInstance(filename + ".yaml", employees=alns.solution.employees, optimize=True)
        run_model(gi)
        
        print("\n############## Optimal solution ##############")
        gi2 = GurobiInstance(filename + ".yaml")
        run_model(gi2, time_limit=300)
        '''
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
