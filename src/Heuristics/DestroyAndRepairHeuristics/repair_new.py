import os
import random

from Heuristics.feasibility_checker import FeasibilityChecker
from path_manager import path_to_src
from abc import ABC, abstractmethod
import copy
from Heuristics.DestroyAndRepairHeuristics.destroy import RandomRemoval
from Heuristics.helper_functions_heuristics import insert_car_move_wo_deep_copy, get_first_stage_solution_list_from_dict, \
    remove_car_move, remove_car_move_from_employee_from_solution
from Heuristics.objective_function import get_obj_val_of_car_moves
from Heuristics.construction_heuristic_new import ConstructionHeuristic

from InstanceGenerator.instance_components import CarMove, ParkingNode, Employee
from InstanceGenerator.world import World
from Heuristics.DestroyAndRepairHeuristics.destroy import Destroy

print(path_to_src)
os.chdir(path_to_src)


class Repair(ABC):

    @classmethod
    def get_unused_moves(cls, destroyed_solution, moves_by_scenario, removed_moves):
        moves = set(removed_moves)
        for scenario in moves_by_scenario:
            for cm in scenario:
                moves.add(cm)

        used_cars = [cm.car.car_id for cm in get_first_stage_solution_list_from_dict(
            destroyed_solution)]
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
        self.destroyed_solution_object = destroyed_solution_object
        self.destroyed_solution = destroyed_solution_object.destroyed_solution
        self.unused_car_moves = Repair.get_unused_moves(
            self.destroyed_solution, unused_car_moves, destroyed_solution_object.removed_moves)
        self.num_first_stage_tasks = destroyed_solution_object.num_first_stage_tasks
        self.neighborhood_size = destroyed_solution_object.neighborhood_size
        self.parking_nodes = parking_nodes
        self.feasibility_checker = FeasibilityChecker(world_instance)
        self.repaired_solution = self._repair()

    @abstractmethod
    def _repair(self):
        pass

    def to_string(self):
        print("\nREPAIR")
        print(f"destroyed solution: {type(self.destroyed_solution_object)}")
        for k, v in self.destroyed_solution.items():
            # print(k.employee_id)
            # print([cm.car_move_id for cm in v])
            for cm in v:
                print(cm.to_string())
        destroyed_solution = get_first_stage_solution_list_from_dict(
            self.destroyed_solution)
        print("Objective value: ", round(get_obj_val_of_car_moves(self.parking_nodes, num_scenarios=1,
                                                            first_stage_car_moves=destroyed_solution), 2))
        print(f"\nrepaired solution: {type(self)}")
        for k, v in self.repaired_solution.items():
            # print(k.employee_id)
            # print([cm.car_move_id for cm in v])
            for cm in v:
                print(cm.to_string())
        repaired_solution = get_first_stage_solution_list_from_dict(
            self.repaired_solution)
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
        # current_solution = copy.copy(self.destroyed_solution)
        current_solution = copy.deepcopy(
            self.destroyed_solution)  # consider dropping deepcopy
        while q > 0:
            best_car_move, best_employee = self._get_best_insertion(
                current_solution)
            # Handles cases when you cannot insert q car_moves to the solution
            if None in (best_car_move, best_employee):
                # print(f"Cannot insert more than {self.neighborhood_size-q} move(s)")
                break
            insert_car_move_wo_deep_copy(
                current_solution, best_car_move, best_employee.employee_id)
            self.feasibility_checker.is_first_stage_solution_feasible(
                current_solution)
            self.unused_car_moves = remove_car_move(
                best_car_move, self.unused_car_moves)
            q -= 1
        # self.repaired_solution = current_solution
        return current_solution

    def _get_best_insertion(self, current_solution: {Employee: [CarMove]}) -> (CarMove, Employee):
        """
        Finds the best car_move to insert, and the id of the employee that should perform it
        :param current_solution: a dictionary with key: employee id and value: list of car moves
        :return best car move, best employee
        """
        best_car_move = None
        best_employee = None
        input_solution = get_first_stage_solution_list_from_dict(
            current_solution)
        best_obj_val = get_obj_val_of_car_moves(parking_nodes=self.parking_nodes, num_scenarios=1,
                                                first_stage_car_moves=input_solution)

        # cars_used = [cm.car.car_id for cm in input_solution]

        for car_move in self.unused_car_moves:
            # if car_move.car.car_id in cars_used:
            #    continue
            for employee, employee_moves in current_solution.items():
                if len(employee_moves) < self.num_first_stage_tasks:
                    # TODO: representation of solution must be reconsidered here. Maybe it is not necessary to create a new object every time

                    insert_car_move_wo_deep_copy(current_solution, car_move, employee.employee_id)
                    if self.feasibility_checker.is_first_stage_solution_feasible(current_solution):
                        solution_with_move = get_first_stage_solution_list_from_dict(current_solution)
                        obj_val = get_obj_val_of_car_moves(self.parking_nodes, num_scenarios=1, first_stage_car_moves=solution_with_move)

                        if obj_val > best_obj_val:
                            best_obj_val = obj_val
                            best_car_move = car_move
                            best_employee = employee
                    remove_car_move_from_employee_from_solution(current_solution, car_move, employee.employee_id)
        return best_car_move, best_employee


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
        if self.regret_nr > len(self.destroyed_solution)-1:
            print("Regret number cannot be higher than one less than the number of employees!")
            exit()

        q = self.neighborhood_size
        current_solution = copy.deepcopy(self.destroyed_solution)
        while q > 0:
            best_car_move, best_employee = self._get_best_insertion_regret(current_solution, self.regret_nr)
            if None in (best_car_move, best_employee):
                # print(f"Cannot insert more than {self.neighborhood_size-q} move(s)")
                break
            current_solution = insert_car_move(current_solution, best_car_move, best_employee.employee_id)
            self.unused_car_moves = remove_car_move(best_car_move, self.unused_car_moves)
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
        # cars_used = [cm.car.car_id for cm in current_solution]

        for car_move in self.unused_car_moves:
            obj_val_dict = {}
            car_move_feasible = False
            for employee, employee_moves in current_solution.items():
                if len(employee_moves) < self.num_first_stage_tasks:

                    # TODO: representation of solution must be reconsidered here. Maybe it is not necessary to create a new object every time
                    solution_with_move = insert_car_move(current_solution, car_move, employee.employee_id)
                    if self.feasibility_checker.is_first_stage_solution_feasible(solution_with_move):
                        car_move_feasible = True
                        solution_with_move = get_first_stage_solution_list_from_dict(solution_with_move)

                        obj_val = get_obj_val_of_car_moves(self.parking_nodes, num_scenarios=1,
                                                           first_stage_car_moves=solution_with_move)
                        obj_val_dict[employee] = obj_val

            if not car_move_feasible:
                continue

            obj_values_sorted = sorted(obj_val_dict.values(), reverse=True)
            if len(obj_values_sorted) and ((0 > highest_obj_val_diff) <= regret_nr):
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
    ch = ConstructionHeuristic("./InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_a.pkl")
    ch.add_car_moves_to_employees()
    rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
                       neighborhood_size=2)

    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()

    gi = GreedyInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
        parking_nodes=ch.parking_nodes, world_instance=ch.world_instance)

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
    #gi.to_string()

    # gi = RegretInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
    #                     parking_nodes=ch.parking_nodes, world_instance=ch.world_instance, regret_nr=1)
    # gi.to_string()

    fc = FeasibilityChecker(ch.world_instance)
    print("feasibilityChecker")
    print(fc.is_first_stage_solution_feasible(gi.repaired_solution))
