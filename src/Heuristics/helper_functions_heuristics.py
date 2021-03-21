import copy
import os
import numpy as np

from Heuristics.feasibility_checker import FeasibilityChecker
from Heuristics.objective_function import get_obj_val_of_car_moves
from path_manager import path_to_src
from src.InstanceGenerator.instance_components import ParkingNode, Employee, CarMove, ChargingNode
from InstanceGenerator.world import World

os.chdir(path_to_src)


def get_first_stage_solution_list_from_dict(first_stage_solution: {int: [CarMove]}) -> [CarMove]:
    """
    :param first_stage_solution: first stage solution dictionary, {e1: [cm1, cm2], e2: [cm3]}
    :return: [cm1, cm2, cm3]
    """
    first_stage_solution_list = []
    for k, v in first_stage_solution.items():
        for i in range(len(v)):
            first_stage_solution_list.append(v[i])
    return first_stage_solution_list


def get_second_stage_solution_dict(input_solution: {int: [[CarMove]]}, num_first_stage_tasks: int) -> {int: [CarMove]}:
    """
    :param input_solution:  dictionary with two scenarios {e1: [[cm1], [cm1, cm2]], e2: [[cm3], [cm3]]}
    :param num_first_stage_tasks: integer
    :return: if num_first_stage_tasks = 1, {e1: [[], [cm2]], e2: [[], []]}
    """
    second_stage_solution = {}
    for k, v in input_solution.items():
        second_stage_solution[k] = []
        for s in range(len(input_solution[k])):
            second_stage_solution[k].append([])
            for i in range(num_first_stage_tasks, len(input_solution[k][s])):
                second_stage_solution[k][s].append(input_solution[k][s][i])
    return second_stage_solution


def get_second_stage_solution_list_from_dict(second_stage_solution_dict: {int: [CarMove]}, num_scenarios: int):
    """
    :param second_stage_solution_dict: eg. {e1: [[], [cm2]], e2: [[], []]}
    :param num_scenarios:  integer
    :return: list of scenarios containing car_moves, [[],[cm2]]
    """
    second_stage_solution = [[] for _ in range(num_scenarios)]
    for k, v in second_stage_solution_dict.items():
        for s in range(num_scenarios):
            for i in range(len(v[s])):
                second_stage_solution[s].append(v[s][i])
    return second_stage_solution


def insert_car_move(current_solution: {int: [CarMove]}, car_move: CarMove, employee: int) -> {int: [CarMove]}:
    """
    :param current_solution: dictionary with employee as key, list of first stage moves as value
    :param car_move: car move object
    :param employee: employee object
    :return: solution with the inserted car move
    """
    solution = copy.deepcopy(current_solution)
    solution.get(employee).append(car_move)
    return solution


def remove_car_move(chosen_car_move: CarMove, car_moves: [CarMove]) -> [CarMove]:
    """
    Removes a car move from a list of car moves and returns the result
    :param chosen_car_move: the car move to remove
    :param car_moves: the list of car moves to remove a car move from
    :return: the list of car moves without the removed move
    """
    car = chosen_car_move.car.car_id
    # return list of car moves that are not associated with the car of the chosen car move
    return [cm for cm in car_moves if cm.car.car_id != car]


def get_assigned_car_moves(employees, scenario: int = None):
    car_moves = []
    if scenario is None:
        for employee in employees:
            for car_move in employee.car_moves:
                car_moves.append(car_move)
    else:
        for employee in employees:
            for car_move in employee.car_moves_second_stage[scenario]:
                car_moves.append(car_move)

    return car_moves


def get_travel_time_between_car_moves(employees, scenario: int = None):
    travel_time = 0
    if scenario is None:
        for employee in employees:
            travel_time += employee.current_time
            for car_move in employee.car_moves:
                travel_time -= car_move.handling_time
    else:
        for employee in employees:
            travel_time += employee.current_time_second_stage[scenario]
            for car_move in employee.car_moves_second_stage[scenario]:
                travel_time -= car_move.handling_time

    return travel_time


def get_best_car_move(parking_nodes, employees, car_moves, first_stage, prioritize_charging, num_scenarios):
    # FIRST STAGE
    if first_stage:
        best_car_move_first_stage = None
        assigned_car_moves_first_stage = get_assigned_car_moves(employees)
        best_obj_val_first_stage = -1000
        longest_travel_time_first_stage = -1000
        if not prioritize_charging:
            best_obj_val_first_stage = get_obj_val_of_car_moves(parking_nodes=parking_nodes, num_scenarios=num_scenarios,
                                                                first_stage_car_moves=assigned_car_moves_first_stage)
            for r in range(len(car_moves)):
                obj_val = get_obj_val_of_car_moves(parking_nodes, num_scenarios,
                                                   first_stage_car_moves=assigned_car_moves_first_stage + [
                                                       car_moves[r]])
                if obj_val > best_obj_val_first_stage:
                    best_obj_val_first_stage = obj_val
                    best_car_move_first_stage = car_moves[r]
        else:
            for r in range(len(car_moves)):
                travel_time = car_moves[r].handling_time
                if travel_time > longest_travel_time_first_stage:
                    longest_travel_time_first_stage = travel_time
                    best_car_move_first_stage = car_moves[r]

        # print("obj_val: ", obj_val)
        # print("best_obj_val: ", best_obj_val_first_stage)
        # print(f"best_car_move: {best_car_move_first_stage.car_move_id}, {best_car_move_first_stage.start_node.node_id} --> {best_car_move_first_stage.end_node.node_id}")
        return best_car_move_first_stage

    # SECOND STAGE
    # TODO: Handle assigning charging moves in the second stage
    else:
        best_car_move_second_stage = [None for _ in range(num_scenarios)]
        best_obj_val_second_stage = [-1000 for _ in range(num_scenarios)]
        assigned_first_stage_car_moves = get_assigned_car_moves(employees)
        longest_travel_time_second_stage = [0 for _ in range(num_scenarios)]

        if not prioritize_charging:
            for s in range(num_scenarios):
                assigned_second_stage_car_moves = get_assigned_car_moves(employees, scenario=s)
                best_obj_val_second_stage[s] = get_obj_val_of_car_moves(parking_nodes, num_scenarios,
                                                                        first_stage_car_moves=
                                                                        assigned_first_stage_car_moves,
                                                                        second_stage_car_moves=
                                                                        assigned_second_stage_car_moves,
                                                                        scenario=s)
        # print(f"best_obj_val_second_stage {best_obj_val_second_stage}")
        obj_val = [0 for _ in range(num_scenarios)]
        for s in range(num_scenarios):
            # zero indexed scenario
            assigned_second_stage_car_moves = get_assigned_car_moves(employees, scenario=s)
            # Parking moves second stage
            if not prioritize_charging:
                for r in range(len(car_moves[s])):
                    obj_val[s] = get_obj_val_of_car_moves(parking_nodes, num_scenarios,
                                                          first_stage_car_moves=assigned_first_stage_car_moves,
                                                          second_stage_car_moves=assigned_second_stage_car_moves +
                                                                                 [car_moves[s][r]], scenario=s)
                    # if car_moves[s][r].car_move_id == 7 and s == 0:
                    #    print(f"car_move {car_moves[s][r].car_move_id}, s {s + 1}")
                    #    print(f"obj_val {obj_val[s]} best_obj_val {best_obj_val_second_stage[s]}")

                    if obj_val[s] > best_obj_val_second_stage[s]:
                        best_obj_val_second_stage[s] = obj_val[s]
                        best_car_move_second_stage[s] = car_moves[s][r]

            # Charging moves second stage
            else:
                for r in range(len(car_moves[s])):
                    travel_time = car_moves[s][r].handling_time
                    if travel_time > longest_travel_time_second_stage[s]:
                        longest_travel_time_second_stage[s] = travel_time
                        best_car_move_second_stage[s] = car_moves[s][r]

        out_list = []
        for car_move in best_car_move_second_stage:
            if car_move is not None:
                out_list.append(car_move.car_move_id)
            else:
                out_list.append(car_move)

        # print(out_list)
        # print([round(o,2) for o in best_obj_val_second_stage])
        return best_car_move_second_stage


def get_best_employee(parking_moves, employees, best_car_move, first_stage, num_scenarios, world_instance,
                      prioritize_charging, charging_moves, charging_moves_second_stage, parking_moves_second_stage):
    feasibility_checker = FeasibilityChecker(world_instance)
    if first_stage:
        best_employee = None
        best_travel_time_to_car_move = 100
        end_node = best_car_move.start_node
    else:
        best_employee_second_stage = [None for _ in range(num_scenarios)]
        best_travel_time_to_car_move_second_stage = [100 for _ in range(num_scenarios)]
        end_node = [(cm.start_node if cm is not None else cm) for cm in best_car_move]

    best_move_not_legal = True

    for employee in employees:
        task_num = len(employee.car_moves)
        # if first stage and the number of completed task for employee is below the number of tasks in first stage,
        # or if second stage and the number of completed tasks are the same or larger than the number of tasks in first stage
        if first_stage == (task_num < world_instance.first_stage_tasks):
            if first_stage:
                legal_move = feasibility_checker.check_legal_move(car_move=best_car_move, employee=employee)
                print(f"legal_move {legal_move}")
                if legal_move:
                    best_move_not_legal = False
                    start_node = employee.current_node
                    travel_time_to_car_move = world_instance.get_employee_travel_time_to_node(start_node,
                                                                                              end_node)
                    if travel_time_to_car_move < best_travel_time_to_car_move:
                        best_travel_time_to_car_move = travel_time_to_car_move
                        best_employee = employee

            else:
                for s in range(num_scenarios):
                    if best_car_move[s] is not None:
                        legal_move = feasibility_checker.check_legal_move(
                            car_move=best_car_move[s], employee=employee, scenario=s)
                        if legal_move:
                            best_move_not_legal = False
                            start_node = employee.current_node_second_stage[s]
                            travel_time_to_car_move = world_instance.get_employee_travel_time_to_node(
                                start_node, end_node[s])
                            if travel_time_to_car_move < best_travel_time_to_car_move_second_stage[s]:
                                best_travel_time_to_car_move_second_stage[s] = travel_time_to_car_move
                                best_employee_second_stage[s] = employee

        # Remove best move if not legal. Else return best employee
    if first_stage:
        if best_move_not_legal:
            if prioritize_charging:
                charging_moves.remove(best_car_move)
            else:
                parking_moves.remove(best_car_move)
            return
        else:
            return best_employee
    else:
        if best_move_not_legal:
            if prioritize_charging:
                for s in range(num_scenarios):
                    charging_moves_second_stage[s] = [cm for cm in charging_moves_second_stage[s] if
                                                      cm != best_car_move[s]]
            else:
                for s in range(num_scenarios):
                    parking_moves_second_stage[s] = [cm for cm in parking_moves_second_stage[s] if
                                                     cm != best_car_move[s]]
            return
        else:
            return best_employee_second_stage


def check_all_charging_moves_completed(num_scenarios, employees, first_stage, parking_nodes) -> bool:
    num = [0 for _ in range(num_scenarios)]
    for employee in employees:
        for car_move in employee.car_moves:
            if isinstance(car_move.end_node, ChargingNode):
                num = [n + 1 for n in num]
        if not first_stage:
            for i in range(num_scenarios):
                for car_move in employee.car_moves_second_stage[i]:
                    if isinstance(car_move.end_node, ChargingNode):
                        num[i] += 1
    # returns whether the number of charging moves for the scenario with the lowest number of charging moves assigned
    # equals the sum of cars in need of charging
    # print(num)
    # print(sum(n.charging_state for n in self.parking_nodes))
    return min(num) == sum(n.charging_state for n in parking_nodes)
