import os
import random

from Heuristics.feasibility_checker import FeasibilityChecker
from path_manager import path_to_src
from abc import ABC, abstractmethod
import copy
from Heuristics.helper_functions_heuristics import insert_car_move, get_first_stage_solution_list_from_dict, \
    remove_all_car_moves_of_car_in_car_move, remove_car_move_from_employee_from_solution
from Heuristics.objective_function import get_obj_val_of_car_moves
from InstanceGenerator.instance_components import CarMove, ParkingNode, Employee
from InstanceGenerator.world import World
from Heuristics.DestroyAndRepairHeuristics.destroy import Destroy, RandomRemoval

print(path_to_src)
os.chdir(path_to_src)


class Repair(ABC):

    @classmethod
    def get_unused_moves(cls, destroyed_solution, moves_by_scenario, removed_moves):
        moves = set(removed_moves)
        for scenario in moves_by_scenario:
            for cm in scenario:
                moves.add(cm)

        used_cars = [cm.car.car_id for cm in get_first_stage_solution_list_from_dict(destroyed_solution)]
        moves = [cm for cm in moves if cm.car.car_id not in used_cars]
        return moves

    def __init__(self, destroyed_solution_object: Destroy, unused_car_moves: [[CarMove]], parking_nodes: [ParkingNode],
                 world_instance: World) -> {int: [CarMove]}:
        """
        :param destroyed_solution_object: object from destroyed solution
        :param construction_heuristic: construction heuristic object
        :param destroyed_solution: dictionary of destroyed solution returned from a Destroy heuristic
        :param unused_car_moves: list of unused car_moves for each scenario e.g.: [[], [], []]
        :param num_first_stage_tasks: int
        :param neighborhood_size: int
        :param parking_nodes: list of ParkingNode
        :return A repaired solution in the form of a dictionary with key employee and value list of first-stage car moves
        """
        self.solution = destroyed_solution_object.solution
        self.unused_car_moves = Repair.get_unused_moves(
            self.solution, unused_car_moves, destroyed_solution_object.removed_moves)
        self.num_first_stage_tasks = destroyed_solution_object.num_first_stage_tasks
        self.neighborhood_size = destroyed_solution_object.neighborhood_size
        self.parking_nodes = parking_nodes
        self.feasibility_checker = FeasibilityChecker(world_instance)
        self._repair()

    @abstractmethod
    def _repair(self):
        pass

    @abstractmethod
    def _get_best_insertion(self, solution, regret_nr):
        pass

    @property
    def hash_key(self):
        hash_dict = {}
        for k, v in self.solution.items():
            emp_moves = []
            for cm in v:
                emp_moves.append(cm.car_move_id)
            hash_dict[k.employee_id] = emp_moves

        #self.hash_key = hash(str(hash_dict))
        return hash(str(hash_dict))

    def to_string(self):
        print("\nREPAIR")
        print(f"\nrepaired solution: {type(self)}")
        for k, v in self.solution.items():
            print(k.employee_id)
            print([cm.car_move_id for cm in v])

        for k, v in self.solution.items():
            # print(k.employee_id)
            # print([cm.car_move_id for cm in v])
            for cm in v:
                print(cm.to_string())
        repaired_solution = get_first_stage_solution_list_from_dict(self.solution)
        print("Objective value: ", round(get_obj_val_of_car_moves(self.parking_nodes, num_scenarios=1,
                                                            first_stage_car_moves=repaired_solution), 2))


class GreedyInsertion(Repair):

    def __init__(self, destroyed_solution_object: Destroy, unused_car_moves: [[CarMove]], parking_nodes: [ParkingNode], world_instance: World):
        super().__init__(destroyed_solution_object,
              unused_car_moves, parking_nodes, world_instance)

    def _repair(self) -> {Employee: [CarMove]}:
        """
        Greedily assigns car moves to employees and returns the repaired solution
        :return: a repaired solution in the form of a dictionary with key: employee id and value: a list of car moves
        """
        q = self.neighborhood_size
        current_solution = self.solution

        while q > 0:
            best_car_move, best_employee, best_idx = self._get_best_insertion(current_solution)

            # Handles cases when you cannot insert q car_moves to the solution
            if None in (best_car_move, best_employee):
                # print(f"Cannot insert more than {self.neighborhood_size-q} move(s)")
                break

            insert_car_move(current_solution, best_car_move, best_employee, best_idx)
            self.unused_car_moves = remove_all_car_moves_of_car_in_car_move(best_car_move, self.unused_car_moves)
            q -= 1
        #return current_solution

    def _get_best_insertion(self, current_solution: {Employee: [CarMove]}, regret_nr=None) -> (CarMove, Employee):
        """
        Finds the best car_move to insert, and the id of the employee that should perform it
        :param current_solution: a dictionary with key: employee id and value: list of car moves
        :return best car move, best employee
        """
        best_car_move = None
        best_employee = None
        best_idx = None
        input_solution = get_first_stage_solution_list_from_dict(current_solution)
        best_obj_val = get_obj_val_of_car_moves(parking_nodes=self.parking_nodes, num_scenarios=1,
                                                first_stage_car_moves=input_solution)

        for car_move in self.unused_car_moves:

            # Checks if best car move is a charging move to a node where the remaining charging capacity is zero
            if car_move.is_charging_move:
                if car_move.end_node.capacity == car_move.end_node.num_charging[0]:
                    self.unused_car_moves.remove(car_move)
                    continue

            for employee, employee_moves in current_solution.items():
                if len(employee_moves) < self.num_first_stage_tasks:
                    for idx in range(len(employee_moves)+1):
                        insert_car_move(current_solution, car_move, employee, idx)
                        # the dictionary checks solution only for employee the relevant employee
                        if self.feasibility_checker.is_first_stage_solution_feasible({employee: current_solution[employee]}):
                            solution_with_move = get_first_stage_solution_list_from_dict(current_solution)
                            obj_val = get_obj_val_of_car_moves(self.parking_nodes, num_scenarios=1, first_stage_car_moves=solution_with_move)

                            if obj_val > best_obj_val:
                                best_obj_val = obj_val
                                best_car_move = car_move
                                best_employee = employee
                                best_idx = idx
                        remove_car_move_from_employee_from_solution(current_solution, car_move, employee)
        # TODO: find out why no new best_car_move is found. Not even the removed moves are inserted
        return best_car_move, best_employee, best_idx


class RegretInsertion(Repair):

    def __init__(self, destroyed_solution_object: Destroy, unused_car_moves: [[CarMove]], parking_nodes: [ParkingNode],
                 world_instance: World, regret_nr: int):
        self.regret_nr = regret_nr
        super().__init__(destroyed_solution_object, unused_car_moves, parking_nodes,
        world_instance)
        """
        The regret insertion heuristic considers the alternative costs of inserting a car_move into gamma (assigned_car_moves).
        """

    def _repair(self) -> {int: [CarMove]}:
        """
        Assigns car moves to employees and returns the repaired solution
        :return: a repaired solution in the form of a dictionary with key: employee id and value: a list of car moves
        """
        if self.regret_nr > len(self.solution):
            print("Regret number cannot be higher than the number of employees!")
            exit()

        q = self.neighborhood_size
        #current_solution = copy.deepcopy(self.solution)
        current_solution = self.solution
        while q > 0:
            best_car_move, best_employee, idx = self._get_best_insertion(current_solution, self.regret_nr)
            if None in (best_car_move, best_employee):
                # print(f"Cannot insert more than {self.neighborhood_size-q} move(s)")
                break

            insert_car_move(current_solution, best_car_move, best_employee, idx)
            self.unused_car_moves = remove_all_car_moves_of_car_in_car_move(best_car_move, self.unused_car_moves)

            q -= 1

    def _get_best_insertion(self, current_solution: {int: [CarMove]}, regret_nr: int) -> (CarMove, int):
        """
        Finds the best car_move to insert, and the id of the employee that should perform it,  based on the regret
        value.
        :param current_solution: a dictionary with key: employee id and value: list of car moves
        :param regret_nr: the regret value
        :return best car move, best employee
        """
        # For each car move, we create a list with the employees the car move can be assigned to, which is sorted so
        # that the first element in the list is the employee for which the objective function increases the most if
        # assigned to the car move. The regret value is the difference between the total objective increase of
        # inserting the car move in its best, second best, ..., (k-1)th best position, and in its kth (regret_nr)
        # position. We do this for each car move, and then select the one with the highest regret value.
        best_car_move = None
        best_employee = None
        best_idx = None
        highest_obj_val_diff = -0.1
        # cars_used = [cm.car.car_id for cm in current_solution]

        for car_move in self.unused_car_moves:
            # Checks if best car move is a charging move to a node where the remaining charging capacity is zero
            if car_move.is_charging_move:
                if car_move.end_node.capacity == car_move.end_node.num_charging[0]:
                    self.unused_car_moves.remove(car_move)
                    continue

            obj_val_dict = {}
            #car_move_feasible = False
            feasible_idx = []
            for employee, employee_moves in current_solution.items():
                if len(employee_moves) < self.num_first_stage_tasks:
                    for idx in range(len(employee_moves)+1):
                        insert_car_move(current_solution, car_move, employee, idx)
                        if self.feasibility_checker.is_first_stage_solution_feasible({employee: current_solution[employee]}):
                            #car_move_feasible = True
                            feasible_idx.append(idx)
                            solution_with_move = get_first_stage_solution_list_from_dict(current_solution)

                            obj_val = get_obj_val_of_car_moves(self.parking_nodes, num_scenarios=1,
                                                               first_stage_car_moves=solution_with_move)
                            obj_val_dict[(employee, idx)] = obj_val

                        remove_car_move_from_employee_from_solution(current_solution, car_move, employee)

            #if not car_move_feasible:
            if not feasible_idx:
                continue

            obj_values_sorted = sorted(obj_val_dict.values(), reverse=True)
            if len(obj_values_sorted) >= regret_nr:
                obj_val_diff = 0
                for i in range(regret_nr-1):
                    obj_val_diff += obj_values_sorted[i]-obj_values_sorted[regret_nr-1]
                if obj_val_diff > highest_obj_val_diff:
                    best_car_move = car_move
                    best_employees_and_idx = []
                    for key, value in obj_val_dict.items():
                        if value == obj_values_sorted[0]:
                            best_employees_and_idx.append(key)
                    best_employee, best_idx = random.choice(best_employees_and_idx)

        return best_car_move, best_employee, best_idx


if __name__ == "__main__":
    from Heuristics.construction_heuristic import ConstructionHeuristic
    from Heuristics.DestroyAndRepairHeuristics.destroy import Destroy, RandomRemoval
    from Heuristics.helper_functions_heuristics import get_first_stage_solution_list_from_dict, \
        get_first_stage_solution_and_removed_moves

    print("\n---- HEURISTIC ----")
    filename = "InstanceGenerator/InstanceFiles/25nodes/25-2-2-1_a.pkl"

    ch = ConstructionHeuristic(filename)
    ch.add_car_moves_to_employees()
    first_stage_solution, ch_removed_moves = get_first_stage_solution_and_removed_moves(ch.assigned_car_moves, ch.world_instance.first_stage_tasks)
    feasibility_checker = FeasibilityChecker(ch.world_instance)
    print("input solution")
    feasibility_checker.is_first_stage_solution_feasible(first_stage_solution, True)
    rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
                       neighborhood_size=2)
    rr.to_string()
    '''from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()
    '''
    #gi = GreedyInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
    #    parking_nodes=ch.parking_nodes, world_instance=ch.world_instance)

    gi = RegretInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
                         parking_nodes=ch.parking_nodes, world_instance=ch.world_instance, regret_nr=4)

    gi.to_string()
    '''profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
    '''
    #gi.to_string()

    # gi = RegretInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
    #                     parking_nodes=ch.parking_nodes, world_instance=ch.world_instance, regret_nr=1)
    # gi.to_string()