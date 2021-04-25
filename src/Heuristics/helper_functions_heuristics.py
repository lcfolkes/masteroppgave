import os

from Heuristics.objective_function import get_obj_val_of_car_moves, get_objective_function_val
from path_manager import path_to_src
from src.InstanceGenerator.instance_components import CarMove, ChargingNode, Employee

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


def get_second_stage_solution_list_from_dict(second_stage_solution_dict: {int: [CarMove]}, num_scenarios: int) -> [[CarMove]]:
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

def get_first_and_second_stage_solution_list_from_dict(input_solution: {Employee: [CarMove]}, num_first_stage_tasks: int) -> ([CarMove], [[CarMove]]):
    """
    :param input_solution: e.g. dictionary with two scenarios {e1: [[cm1], [cm1, cm2]], e2: [[cm3], [cm3]]}
    :param num_first_stage_tasks: integer
    :return: e.g. [cm1, cm3], [[], [cm2]]
    """
    first_stage_list = []
    second_stage_list = []
    for k, car_moves in input_solution.items():
        first_stage_list += car_moves[0][:min(len(car_moves[0]), num_first_stage_tasks)]
        for s in range(len(car_moves)):
            try:
                second_stage_list[s] += car_moves[s][min(len(car_moves[0]), num_first_stage_tasks):]
            except:
                second_stage_list.append([])
                second_stage_list[s] += car_moves[s][min(len(car_moves[0]), num_first_stage_tasks):]
    return first_stage_list, second_stage_list


def insert_car_move(solution: {Employee: [CarMove]}, car_move: CarMove, employee, idx):
    """
    Updates state of solution. No deep copy, original object is mutated
    :param current_solution: dictionary with employee as key, list of first stage moves as value
    :param car_move: car move object
    :param employee: employee object
    :return: solution with the inserted car move
    """

    solution.get(employee).insert(idx, car_move)
    # Update charging state in end node if the chosen move is a charging move
    car_move.set_employee(employee)


def remove_car_move_from_employee_from_solution(solution: {Employee: [CarMove]}, car_move: CarMove, employee):
    """
    :param solution: input solution with move
    :param car_move: car move to be removed
    :param employee_id: id of employee of whom is assigned the car move
    :return: solution without car_move
    """
    # employee_obj = [e for e in solution.keys() if e.employee_id == employee_id][0]
    solution.get(employee).remove(car_move)
    car_move.reset()
    # return solution


'''
def remove_car_move(chosen_car_move: CarMove, car_moves: [CarMove]) -> [CarMove]:
    """
    Removes a car move from a list of car moves and returns the result
    :param chosen_car_move: the car move to remove
    :param car_moves: the list of car moves to remove a car move from
    :return: the list of car moves without the removed move
    """
    # return list of car moves that are not associated with the car of the chosen car move
    return car_moves.remove(chosen_car_move)'''


def remove_all_car_moves_of_car_in_car_move(chosen_car_move: CarMove, car_moves: [CarMove]) -> [CarMove]:
    """
    Removes a car move and all other moves of the same car from a list of car moves and returns the result
    :param chosen_car_move: the car move to remove
    :param car_moves: the list of car moves to remove a car move from
    :return: list of car moves that are not associated with the car of the chosen car move
    """
    car = chosen_car_move.car.car_id
    # return
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


def get_separate_assigned_car_moves(employees, num_scenarios):
    first_stage_car_moves = []
    second_stage_car_moves = [[] for _ in range(num_scenarios)]
    for employee in employees:
        for car_move in employee.car_moves:
            if car_move not in first_stage_car_moves:
                first_stage_car_moves.append(car_move)

        for scenario in range(len(employee.car_moves_second_stage)):
            for car_move in employee.car_moves_second_stage[scenario]:
                if car_move not in second_stage_car_moves[scenario]:
                    second_stage_car_moves[scenario].append(car_move)

    return first_stage_car_moves, second_stage_car_moves


def get_best_car_move(parking_nodes, employees, car_moves, first_stage, num_scenarios):
    # FIRST STAGE
    if first_stage:
        best_car_move_first_stage = None
        assigned_car_moves_first_stage = get_assigned_car_moves(employees)
        best_obj_val_first_stage = get_obj_val_of_car_moves(parking_nodes=parking_nodes, num_scenarios=num_scenarios,
                                                            first_stage_car_moves=assigned_car_moves_first_stage,
                                                            include_employee_check=False)
        # print("Iteration")
        for r in range(len(car_moves)):
            if car_moves[r].is_charging_move:
                # Checking if charging node has space for another car
                if car_moves[r].end_node.capacity == car_moves[r].end_node.num_charging[0]:
                    continue

            obj_val = get_obj_val_of_car_moves(parking_nodes, num_scenarios,
                                               first_stage_car_moves=assigned_car_moves_first_stage + [
                                                   car_moves[r]], include_employee_check=False)
            if obj_val > best_obj_val_first_stage:
                best_obj_val_first_stage = obj_val
                best_car_move_first_stage = car_moves[r]
                # print(f"{best_car_move_first_stage.start_node.node_id} -> {best_car_move_first_stage.end_node.node_id}, Obj val:{obj_val}")
            # elif best_car_move_first_stage:
            # print(f"{best_car_move_first_stage.start_node.node_id} -> {best_car_move_first_stage.end_node.node_id}, Not improving")

        return best_car_move_first_stage


    # SECOND STAGE
    else:
        best_car_move_second_stage = [None for _ in range(num_scenarios)]

        assigned_first_stage_car_moves, assigned_second_stage_car_moves = get_separate_assigned_car_moves(employees,
                                                                                                          num_scenarios)
        best_obj_val = get_obj_val_of_car_moves(parking_nodes, num_scenarios,
                                                first_stage_car_moves=assigned_first_stage_car_moves,
                                                second_stage_car_moves=assigned_second_stage_car_moves,
                                                include_employee_check=False)

        best_obj_val_second_stage = [best_obj_val for _ in range(num_scenarios)]

        # obj_val = [0 for _ in range(num_scenarios)]
        for s in range(num_scenarios):
            # Parking moves second stage
            for r in range(len(car_moves[s])):
                if car_moves[s][r].is_charging_move:
                    # Checking if charging node has space for another car

                    if car_moves[s][r].end_node.capacity == car_moves[s][r].end_node.num_charging[s]:
                        continue

                assigned_second_stage_car_moves[s].append(car_moves[s][r])
                obj_val = get_obj_val_of_car_moves(parking_nodes, num_scenarios,
                                                   first_stage_car_moves=assigned_first_stage_car_moves,
                                                   second_stage_car_moves=assigned_second_stage_car_moves,  # +
                                                   # [car_moves[s][r]],
                                                   include_employee_check=False)
                assigned_second_stage_car_moves[s].remove(car_moves[s][r])

                if obj_val > best_obj_val_second_stage[s]:
                    if car_moves[s][r].is_charging_move:

                        # Checking if charging node has space for another car
                        if car_moves[s][r].end_node.capacity == car_moves[s][r].end_node.num_charging[s]:
                            continue

                        else:
                            best_obj_val_second_stage[s] = obj_val
                            best_car_move_second_stage[s] = car_moves[s][r]
                            # car_moves[s][r].end_node.add_car(scenario=s)
                    else:
                        best_obj_val_second_stage[s] = obj_val
                        best_car_move_second_stage[s] = car_moves[s][r]

        out_list = []
        for car_move in best_car_move_second_stage:
            if car_move is not None:
                out_list.append(car_move.car_move_id)
            else:
                out_list.append(car_move)

        # print([round(o,2) for o in best_obj_val_second_stage])
        return best_car_move_second_stage


def safe_zero_division(a, b):
    return a / b if b else 0.0


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


def get_first_stage_solution(input_solution, num_first_stage_tasks):
    first_stage_solution = {}
    # print(self.input_solution)
    for k, v in input_solution.items():
        first_stage_solution[k] = input_solution[k][0][:min(num_first_stage_tasks, len(input_solution[k][0]))]
    return first_stage_solution

def get_first_and_second_stage_solution(input_solution, num_first_stage_tasks):
    first_stage_solution = {}
    second_stage_solution = {}
    # print(self.input_solution)
    for k, v in input_solution.items():
        # For solutions where number of assigned tasks are less than the number of first stage tasks
        # First stage
        first_stage_solution[k] = input_solution[k][0][:min(num_first_stage_tasks, len(input_solution[k][0]))]
        # Second stage
        second_stage_solution[k] = []
        for s in range(len(v)):
            second_stage_solution[k].append(input_solution[k][s][min(num_first_stage_tasks, len(v[s])):])

    return first_stage_solution, second_stage_solution

def reconstruct_solution_from_first_and_second_stage(first_stage, second_stage):
    solution = {}
    for k, scenarios in second_stage.items():
        solution[k] = []
        for s in range(len(scenarios)):
            scenario_moves = []
            for i in range(len(first_stage[k])):
                scenario_moves.append(first_stage[k][i])
            for i in range(len(scenarios[s])):
                scenario_moves.append(scenarios[s][i])
            solution[k].append(scenario_moves)
    return solution


def get_first_stage_solution_and_removed_moves(input_solution, num_first_stage_tasks):
    removed_second_stage_moves = set()
    first_stage_solution = {}

    # print(self.input_solution)
    for k, v in input_solution.items():
        first_stage_solution[k] = []
        for i in range(min(num_first_stage_tasks, len(input_solution[k][0]))):
            first_stage_solution[k].append(input_solution[k][0][i])

        for s in range(len(input_solution[k])):
            # For solutions where number of assigned tasks are less than the number of first stage tasks
            for i in range(min(num_first_stage_tasks, len(input_solution[k][s])), len(input_solution[k][s])):
                # input_solution[k][s][i]
                input_solution[k][s][i].reset(s)
                removed_second_stage_moves.add(input_solution[k][s][i])

    removed_moves = list(removed_second_stage_moves)

    return first_stage_solution, removed_moves


def reset_car_moves(removed_car_moves: [CarMove]):
    for cm in removed_car_moves:
        cm.reset()


def get_solution_list(input_solution, num_first_stage_tasks):
    first_stage_solution = {}
    second_stage_solution = []
    # print(self.input_solution)
    for k, v in input_solution.items():
        first_stage_solution[k] = set()
        for s in range(len(input_solution[k])):
            second_stage_solution.append([])
            # For solutions where number of assigned tasks are less than the number of first stage tasks
            for i in range(min(num_first_stage_tasks, len(input_solution[k][s]))):
                first_stage_solution[k].add(input_solution[k][s][i])
            for i in range(min(num_first_stage_tasks, len(input_solution[k][s]), len(input_solution[k][s]))):
                second_stage_solution[s].append(input_solution[k][s][i])
        first_stage_solution[k] = list(first_stage_solution[k])

    first_stage_solution_list = get_first_stage_solution_list_from_dict(first_stage_solution)
    return first_stage_solution_list, second_stage_solution
