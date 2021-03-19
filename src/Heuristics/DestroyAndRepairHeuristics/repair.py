import os
import random

from path_manager import path_to_src
from abc import ABC, abstractmethod
import copy
from Heuristics.DestroyAndRepairHeuristics.destroy import RandomRemoval
from Heuristics.helper_functions_heuristics import insert_car_move, get_obj_val_of_car_moves, \
    get_first_stage_solution_list_from_dict
from Heuristics.construction_heuristic import ConstructionHeuristic
from InstanceGenerator.instance_components import CarMove, ParkingNode

os.chdir(path_to_src)


class Repair(ABC):
    def __init__(self, destroyed_solution: {int: [CarMove]}, unused_car_moves: [CarMove], num_first_stage_tasks: int,
                 neighborhood_size: int, parking_nodes: [ParkingNode]) -> {int: [CarMove]}:
        """
        :param destroyed_solution: dictionary of destroyed solution returned from a Destroy heuristic
        :param unused_car_moves: list of unused car_moves for each scenario e.g.: [[], [], []]
        :param num_first_stage_tasks: int
        :param neighborhood_size: int
        :param parking_nodes: list of ParkingNode
        :return A repaired solution in the form of a dictionary with key employee and value list of first-stage car moves
        """
        self.destroyed_solution = destroyed_solution
        self.unused_car_moves = unused_car_moves
        self.num_first_stage_tasks = num_first_stage_tasks
        self.neighborhood_size = neighborhood_size
        self.parking_nodes = parking_nodes
        self.repaired_solution = self._repair()

    @abstractmethod
    def _repair(self):
        pass

    def to_string(self):
        print("\nREPAIR")
        print("destroyed solution")
        for k, v in self.destroyed_solution.items():
            print(k)
            print([cm.car_move_id for cm in v])

        print("repaired solution")
        for k, v in self.repaired_solution.items():
            print(k)
            print([cm.car_move_id for cm in v])


class GreedyInsertion(Repair):

    def _repair(self) -> {int: [CarMove]}:
        """
        Greedily assigns car moves to employees and returns the repaired solution
        :return: a repaired solution in the form of a dictionary with key: employee id and value: a list of car moves
        """
        q = self.neighborhood_size
        current_solution = copy.deepcopy(self.destroyed_solution)
        while q > 0:
            best_car_move, best_employee = self._get_best_insertion(current_solution)
            current_solution = insert_car_move(current_solution, best_car_move, best_employee)
            self.unused_car_moves.remove(best_car_move)
            q -= 1
        self.repaired_solution = current_solution
        return current_solution

    def _get_best_insertion(self, current_solution: {int: [CarMove]}) -> (CarMove, int):
        """
        Finds the best car_move to insert, and the id of the employee that should perform it
        :param current_solution: a dictionary with key: employee id and value: list of car moves
        :return best car move, best employee
        """
        best_car_move = None
        best_employee = None
        best_obj_val = -1000
        for car_move in self.unused_car_moves:
            for employee_id, employee_moves in current_solution.items():
                if len(employee_moves) < self.num_first_stage_tasks:
                    solution_with_move = insert_car_move(current_solution, car_move, employee_id)
                    solution_with_move = get_first_stage_solution_list_from_dict(solution_with_move)
                    obj_val = get_obj_val_of_car_moves(self.parking_nodes, num_scenarios=1,
                                                       first_stage_car_moves=solution_with_move)
                    if obj_val > best_obj_val:
                        best_obj_val = obj_val
                        best_car_move = car_move
                        best_employee = employee_id
        return best_car_move, best_employee


class RegretInsertion(Repair):

    def __init__(self, destroyed_solution, unused_car_moves, num_first_stage_tasks, neighborhood_size, parking_nodes, regret_nr):
        """
        The regret insertion heuristic considers the alternative costs of inserting a car_move into gamma (assigned_car_moves).
        """
        self.regret_nr = regret_nr

        super().__init__(destroyed_solution, unused_car_moves, num_first_stage_tasks, neighborhood_size, parking_nodes)

    def _repair(self) -> {int: [CarMove]}:
        """
        Assigns car moves to employees and returns the repaired solution
        :return: a repaired solution in the form of a dictionary with key: employee id and value: a list of car moves
        """
        if self.regret_nr > len(self.destroyed_solution)-1:
            print("Regret number cannot be higher than one less than the number of employees!")
            exit()

        q = self.neighborhood_size
        current_solution = copy.deepcopy(self.destroyed_solution)
        while q > 0:
            best_car_move, best_employee = self._get_best_insertion_regret(current_solution, self.regret_nr)
            current_solution = insert_car_move(current_solution, best_car_move, best_employee)
            self.unused_car_moves.remove(best_car_move)
            q -= 1
        self.repaired_solution = current_solution
        return current_solution

    def _get_best_insertion_regret(self, current_solution: {int: [CarMove]}, regret_nr: int) -> (CarMove, int):
        """
        Finds the best car_move to insert, and the id of the employee that should perform it,  based on the regret
        value.
        :param current_solution: a dictionary with key: employee id and value: list of car moves
        :param regret_nr: the regret value
        :return best car move, best employee
        """
        # For each car move, we create a list with the employees the car move can be assigned to, which is sorted so
        # that the first element in the list is the employee for which the objective function increases the most if
        # assigned to the car move. The regret value is the difference between inserting the car move in its best
        # position and in its kth (regret_nr) position. We do this for each car move, and then select the one with the
        # highest regret value.

        best_car_move = None
        best_employee = None
        highest_obj_val_diff = -0.1
        for car_move in self.unused_car_moves:
            obj_val_dict = {}
            for employee_id, employee_moves in current_solution.items():
                if len(employee_moves) < self.num_first_stage_tasks:
                    solution_with_move = insert_car_move(current_solution, car_move, employee_id)
                    solution_with_move = get_first_stage_solution_list_from_dict(solution_with_move)
                    obj_val = get_obj_val_of_car_moves(self.parking_nodes, num_scenarios=1,
                                                       first_stage_car_moves=solution_with_move)
                    obj_val_dict[employee_id] = obj_val
            obj_values_sorted = sorted(obj_val_dict.values())
            if (len(obj_values_sorted) <= regret_nr):
                best_car_move = car_move
                best_employees = []
                for key, value in obj_val_dict.items():
                    if value == obj_values_sorted[0]:
                        best_employees.append(key)
                        best_employee = random.choice(best_employees)
            elif (obj_values_sorted[0] - obj_values_sorted[regret_nr]) > highest_obj_val_diff:
                best_car_move = car_move
                best_employees = []
                for key, value in obj_val_dict.items():
                    if value == obj_values_sorted[0]:
                        best_employees.append(key)
                best_employee = random.choice(best_employees)
        return best_car_move, best_employee



if __name__ == "__main__":
    print("\n---- HEURISTIC ----")
    ch = ConstructionHeuristic("InstanceGenerator/InstanceFiles/6nodes/6-3-1-1_c.pkl")
    ch.add_car_moves_to_employees()
    ch.print_solution()
    ch.get_objective_function_val()
    rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
                       neighborhood_size=2)
    rr.to_string()
    gi = RegretInsertion(destroyed_solution=rr.destroyed_solution, unused_car_moves=rr.removed_moves,
                         num_first_stage_tasks=ch.world_instance.first_stage_tasks, neighborhood_size=2,
                         parking_nodes=ch.parking_nodes, regret_nr=2)
    gi.to_string()
