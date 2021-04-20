from abc import ABC, abstractmethod
import os
import random
import copy
from path_manager import path_to_src
from Heuristics.helper_functions_heuristics import get_first_stage_solution_list_from_dict, \
    get_first_stage_solution_and_removed_moves
from Heuristics.objective_function import get_obj_val_of_car_moves
from Heuristics.heuristics_constants import HeuristicsConstants
import numpy as np
from src.InstanceGenerator.instance_components import CarMove

os.chdir(path_to_src)


class Destroy(ABC):
    def __init__(self, solution, num_first_stage_tasks, neighborhood_size=2):
        """
        :param solution: (s) assigned car_moves of constructed solution. solution[(k,s)], dictionary containing car_move
        assigned to employee in scenario s
        :param neighborhood_size: (q) number of car_moves to remove
        """
        self.num_scenarios = len(list(solution.items())[0][1])
        self.num_first_stage_tasks = num_first_stage_tasks
        #self.input_solution = solution
        self.solution, self.removed_moves = get_first_stage_solution_and_removed_moves(solution, num_first_stage_tasks)
        #self.to_string("first_stage")
        self.neighborhood_size = neighborhood_size
        self._destroy()
        #print("removed moves: ", [cm.car_move_id for cm in self.removed_moves])

    @abstractmethod
    def _destroy(self):
        pass

    def to_string(self, solution_type=None):
        '''print("\nDESTROY")
        print("input solution")
        for k, v in self.input_solution.items():
            print(k.employee_id)
            for s in v:
                print([cm.car_move_id for cm in s])'''

        if solution_type == "first_stage":
            print("input first stage solution")
        else:
            print("destroyed solution")
        #print(self.solution)
        for k, v in self.solution.items():
            print(k.employee_id)
            print([cm.car_move_id for cm in v])
        for k, v in self.solution.items():
            # print(k.employee_id)
            # print([cm.car_move_id for cm in v])
            for cm in v:
                print(cm.to_string())


class RandomRemoval(Destroy):

    def __init__(self, solution, num_first_stage_tasks, neighborhood_size):
        super().__init__(solution, num_first_stage_tasks, neighborhood_size)

    def _destroy(self):
        #solution = copy.deepcopy(self.first_stage_solution)
        solution = self.solution

        n_size = self.neighborhood_size
        while n_size > 0:
            k = random.choice(list(solution.keys()))
            # ensures list of chosen key is not empty
            if not any(solution[k]):
                continue
            i = random.randrange(0, len(solution[k]), 1)
            # Charging node states are updated and employees are removed
            solution[k][i].reset()
            self.removed_moves.append(solution[k][i])
            solution[k] = solution[k][:i] + solution[k][i + 1:]
            n_size -= 1





class WorstRemoval(Destroy):

    def __init__(self, solution, num_first_stage_tasks, neighborhood_size, randomization_degree, parking_nodes):
        '''
		Worst removal removes solutions that have a bad influence on the objective value.
		In this case, that means moves where the objective function decreases little when they are removed.
		:param randomization_degree: (p) parameter that determines the degree of randomization, p>=1.
				Low value of p corresponds to much randomness
		'''
        self.parking_nodes = parking_nodes
        self.randomization_degree = randomization_degree
        super().__init__(solution, num_first_stage_tasks, neighborhood_size)

    def _destroy(self):
        solution_dict = self.solution
        solution_list = get_first_stage_solution_list_from_dict(solution_dict)
        #first_stage_solution_dict = self.first_stage_solution #copy.deepcopy(self.first_stage_solution)

        obj_val = {}  # {index: obj val}

        for i in range(len(solution_list)):
            solution_copy = solution_list[:i] + solution_list[i + 1:]
            obj_val_remove_cm = get_obj_val_of_car_moves(parking_nodes=self.parking_nodes,
                                                         num_scenarios=self.num_scenarios,
                                                         first_stage_car_moves=solution_copy)
            obj_val[i] = obj_val_remove_cm

        n_size = self.neighborhood_size
        obj_val_list = sorted(obj_val.items(), key=lambda x: x[1],
                              reverse=True)  # e.g. [(index, obj_val] = [(1, 89.74), (0, 85.96)]

        removed_car_moves_by_id = []
        removed_car_moves = []
        while n_size > 0 and len(obj_val_list) > 0:
            # Handle randomization (y^p*|L|)
            index = np.floor(np.power(random.random(), self.randomization_degree) * len(obj_val_list)).astype(int)

            try:
                idx = obj_val_list[index][0]
                obj_val_list.pop(index)
                removed_car_moves_by_id.append(solution_list[idx].car_move_id)

                # Charging node states are updated and employees are removed
                solution_list[idx].reset()
                removed_car_moves.append(solution_list[idx])
            except:
                # TODO: find out why we sometimes get index out of range.
                #  I suspect it happens when len of list is 0 and index becomes 0
                # seems to be correct. What do we do if it does not work? just continue?
                print("destroy.py/worst")
                print(f"index {index}")
                print(f"list_len {len(obj_val_list)}")

            n_size -= 1

        # From ropke and pisinger in 3.1.3 Worst Removal: The request is not moved to the request bank, but removed completely
        self.removed_moves += removed_car_moves

        for employee, car_moves in solution_dict.items():
            solution_dict[employee] = [cm for cm in car_moves if cm.car_move_id
                                            not in removed_car_moves_by_id]
        # print(first_stage_solution_dict)
        #return first_stage_solution_dict


class ShawRemoval(Destroy):
    # The relatedness measure can e.g. measure how close the the start nodes are to each other, and the same for the
    # end nodes, when the car moves are started, and maybe whether only a few employees are able to perform both
    # requests. It should also probably only compare parking moves with parking moves and charging moves with
    # charging moves.
    @classmethod
    def relatedness_measure(cls, car_move_i: CarMove, car_move_j: CarMove):
        '''
		:param car_move_i: a car move object
		:param car_move_j: another car move object
		:return:
		'''
        relatedness = 0
        relatedness += HeuristicsConstants.FROM_NODE_WEIGHT * (
            0 if car_move_i.start_node.node_id == car_move_j.start_node.node_id else 1)
        relatedness += HeuristicsConstants.TO_NODE_WEIGHT * (
            0 if car_move_i.end_node.node_id == car_move_j.end_node.node_id else 1)
        relatedness += HeuristicsConstants.IS_CHARGING_WEIGHT * (
            0 if car_move_i.is_charging_move == car_move_j.is_charging_move else 1)
        relatedness += HeuristicsConstants.TRAVEL_DISTANCE_WEIGHT * abs(
            car_move_i.handling_time - car_move_j.handling_time)
        relatedness += HeuristicsConstants.START_TIME_WEIGHT * abs(
            np.mean(car_move_i.start_time) - np.mean(car_move_j.start_time))
        return relatedness

    def __init__(self, solution, num_first_stage_tasks, neighborhood_size, randomization_degree):
        """
        Shaw removal removes car-moves that are somewhat similar to each other. It takes in a solution, the number of car moves to remove, and a randomness parameter p >= 1.
        :param randomization_degree: (p) parameter that determines the degree of randomization
        """
        self.randomization_degree = randomization_degree
        super().__init__(solution, num_first_stage_tasks, neighborhood_size)

    def _destroy(self):
        solution_dict = self.solution
        solution_list = get_first_stage_solution_list_from_dict(solution_dict)
        #first_stage_solution_dict = copy.deepcopy(self.first_stage_solution)
        removed_list = []
        rand_index = random.randrange(0, len(solution_list), 1)

        # Charging node states are updated and employees are removed
        solution_list[rand_index].reset()
        removed_list.append(solution_list[rand_index])
        car_moves_not_removed = [cm for cm in solution_list if cm not in removed_list]

        while len(removed_list) < self.neighborhood_size and len(car_moves_not_removed) > 0:
            rand_index = random.randrange(0, len(removed_list), 1)
            # print(rand_index)
            removed_car_move = removed_list[rand_index]
            car_moves_not_removed = [cm for cm in solution_list if cm not in removed_list]
            car_moves_not_removed = sorted(
                car_moves_not_removed,
                key=lambda cm: ShawRemoval.relatedness_measure(cm, removed_car_move),
                reverse=False)
            # Handle randomization (y^p*|L|)
            try:
                index = np.floor(
                    np.power(random.random(), self.randomization_degree) * len(car_moves_not_removed)).astype(int)

                # Charging node states are updated and employees are removed
                car_moves_not_removed[index].reset()
                removed_list.append(car_moves_not_removed[index])
            except:
                # TODO: find out why we sometimes get index out of range.
                # I suspect it happens when len of list is 0 and index becomes 0
                print("destroy.py/shaw")
                print(f"index {index}")
                print(f"list_len {len(car_moves_not_removed)}")

        self.removed_moves += removed_list
        removed_list = [cm.car_move_id for cm in removed_list]
        for employee, car_moves in solution_dict.items():
            solution_dict[employee] = [cm for cm in car_moves if cm.car_move_id not in removed_list]

        #return first_stage_solution_dict

'''
if __name__ == "__main__":
    from Heuristics.objective_function import get_objective_function_val

    print("\n---- HEURISTIC ----")
    ch = ConstructionHeuristic("./InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_special_case.pkl")
    ch.print_solution()
    get_objective_function_val(ch.parking_nodes, ch.employees, ch.num_scenarios)
    # rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
    #				   neighborhood_size=1)
    # rr.to_string()

    # wr = WorstRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
    #				  neighborhood_size=1, randomization_degree=10, parking_nodes=ch.parking_nodes)
    # wr.to_string()
    print("solution\n", ch.assigned_car_moves)
    sr = ShawRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
                     neighborhood_size=2, randomization_degree=10)
    sr.to_string()
'''