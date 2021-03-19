import copy
import os
import numpy as np

from Heuristics.feasibility_checker import FeasibilityChecker
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


# ------------------------ #
#    OBJECTIVE FUNCTION    #
# ------------------------ #

def calculate_z(parking_nodes: [ParkingNode], first_stage_car_moves: [CarMove], second_stage_car_moves: [[CarMove]],
                verbose: bool = False) -> {int: np.array([int])}:
    # z is the number of customer requests served. It must be the lower of the two values
    # available cars and the number of customer requests D_is
    # number of available cars in the beginning of the second stage, y_i
    """
    :param parking_nodes: list of parking node objects, [pn1, pn2]
    :param first_stage_car_moves: list of car_move objects for first stage, [cm1, cm2]
    :param second_stage_car_moves: [[cm2],[cm2],[cm2, cm3]]
    :param verbose: if True print information
    :return: z, dictionary with node_id as keys and a numpy array for each scenario as value,
    {node_id: np.array([0, 2, 1])}
    """

    start_nodes_first_stage = [car_move.start_node for car_move in first_stage_car_moves if
                               isinstance(car_move.end_node, ParkingNode)]
    end_nodes_first_stage = [car_move.end_node for car_move in first_stage_car_moves if
                             isinstance(car_move.end_node, ParkingNode)]

    y = {parking_node.node_id: parking_node.parking_state for parking_node in parking_nodes}
    for n in start_nodes_first_stage:
        y[n.node_id] -= 1
    for n in end_nodes_first_stage:
        y[n.node_id] += 1

    node_demands = {parking_node.node_id: {'customer_requests': parking_node.customer_requests,
                                           'car_returns': parking_node.car_returns} for parking_node in parking_nodes}

    z = {}

    start_nodes_second_stage = [[car_move.start_node.node_id for car_move in scenarios
                                 if isinstance(car_move.end_node, ParkingNode)] for scenarios in
                                second_stage_car_moves]

    # car_move.start_node should be a list of car moves with len(list) = num_scenarios
    for n in parking_nodes:
        second_stage_moves_out = np.array([scenario.count(n.node_id) for scenario in start_nodes_second_stage])
        y[n.node_id] = np.maximum(y[n.node_id], 0)

        z_val = np.minimum(y[n.node_id] + node_demands[n.node_id]['car_returns'] - second_stage_moves_out,
                           node_demands[n.node_id]['customer_requests'])
        z_val = np.maximum(z_val, 0)
        z[n.node_id] = z_val
        if verbose:
            print(f"z[{n.node_id}] {z[n.node_id]}")
    return z


def calculate_profit_customer_requests(z: {int: np.array([int])}, scenario: int = None) -> float:
    """
    :param z: dictionary with node_id as keys and a numpy array for each scenario as value,
    {node_id: np.array([0, 2, 1])}
    :param scenario: integer describing scenario. If given, only calculate profit from request for that scenario
    :return: float with profit. if scenarios is given, then avg. over scenarios is returned, else value for given
    scenario is returned
    """
    # sum across scenarios for all nodes
    z_sum = sum(v for k, v in z.items())
    if scenario is None:
        # print(f"z_sum {z_sum}")
        z_sum_scenario_average = np.mean(z_sum)
        return World.PROFIT_RENTAL * z_sum_scenario_average
    else:
        # print(f"z_sum[{scenario+1}] {z_sum[scenario]}")
        return World.PROFIT_RENTAL * z_sum[scenario]


def calculate_costs_relocation(car_moves: [[CarMove]], num_scenarios: int = None,
                               individual_scenario: bool = False) -> float:
    """
    :param car_moves: list of scenarios containing car_moves, [[cm1],[cm1],[cm1, cm2]]
    :param num_scenarios: int
    :param individual_scenario: boolean. if individual scenario then you do not average over scenarios
    :return: float with costs for relocation
    """
    # Sum of all travel times across all car moves
    sum_travel_time = sum(car_move.handling_time for car_move in car_moves)
    if individual_scenario:
        # print(f"individual_scenario {sum_travel_time}")
        return World.COST_RELOCATION * sum_travel_time
    else:
        sum_travel_time_scenario_avg = sum_travel_time / num_scenarios
        # print(f"sum_scenarios {sum_travel_time}")
        # print(f"avg_scenarios {sum_travel_time_scenario_avg}")
        return World.COST_RELOCATION * sum_travel_time_scenario_avg


def calculate_cost_deviation_ideal_state(parking_nodes: [ParkingNode], z: {int: np.array([int])},
                                         first_stage_car_moves: [CarMove], second_stage_car_moves: [[CarMove]],
                                         scenario: int = None, verbose: bool = False) -> float:
    """
    :param parking_nodes: list of parking node objects
    :param z: dictionary with node_id as keys and a numpy array for each scenario as value, {node_id: np.array([0, 2, 1])}
    :param first_stage_car_moves: list of car_move objects for first stage, [cm1, cm2]
    :param second_stage_car_moves: [[cm2],[cm2],[cm2, cm3]]
    :param scenario: if None, then average for all scenario is calculated, else for a specific scenario
    :param verbose: True if you want to print information
    :return: return float with the cost associated with deviation from the ideal state.
    """

    start_nodes_first_stage = [car_move.start_node for car_move in first_stage_car_moves if
                               isinstance(car_move.end_node, ParkingNode)]
    end_nodes_first_stage = [car_move.end_node for car_move in first_stage_car_moves if
                             isinstance(car_move.end_node, ParkingNode)]

    w = {n.node_id: (n.ideal_state - n.parking_state + z[n.node_id] - n.car_returns) for n in parking_nodes}

    for n in start_nodes_first_stage:
        w[n.node_id] += 1
    for n in end_nodes_first_stage:
        w[n.node_id] -= 1

    start_nodes_second_stage = [
        [car_move.start_node.node_id for car_move in scenarios if isinstance(car_move.end_node, ParkingNode)] for
        scenarios in
        second_stage_car_moves]

    end_nodes_second_stage = [
        [car_move.end_node.node_id for car_move in scenarios if isinstance(car_move.end_node, ParkingNode)] for
        scenarios in second_stage_car_moves]

    for n in parking_nodes:
        second_stage_moves_out = np.array([cm.count(n.node_id) for cm in start_nodes_second_stage])
        second_stage_moves_in = np.array([cm.count(n.node_id) for cm in end_nodes_second_stage])

        w[n.node_id] += second_stage_moves_out - second_stage_moves_in
        # require w_is >= 0
        w[n.node_id] = np.maximum(w[n.node_id], 0)

        if verbose:
            print(f"\nw[{n.node_id}] {w[n.node_id]}")
            print(f"ideal state {n.ideal_state}")
            print(f"initial_state {n.parking_state}")
            print(f"car returns {n.car_returns}")
            print(f"customer requests {n.customer_requests}")

    w_sum = sum(v for k, v in w.items())

    if scenario is None:
        w_sum_scenario_average = np.mean(w_sum)
        # print(f"w_sum {w_sum}")
        return World.COST_DEVIATION * w_sum_scenario_average
    else:
        # print(f"w_sum[{scenario+1}] {w_sum[scenario]}")
        return World.COST_DEVIATION * w_sum[scenario]


def get_obj_val_of_car_moves(parking_nodes: [ParkingNode], num_scenarios: int, first_stage_car_moves: [CarMove] = None,
                             second_stage_car_moves: [[CarMove]] = None,
                             scenario: int = None, verbose: bool = False) -> float:
    """
    :param parking_nodes: list of parking node objects
    :param num_scenarios:
    :param first_stage_car_moves: list of car_move objects for first stage, [cm1, cm2]
    :param second_stage_car_moves: [[cm2],[cm2],[cm2, cm3]]
    :param scenario: if None, then average for all scenario is calculated, else for a specific scenario
    :param verbose:  True if you want to print information
    :return: float of objective value of car_moves
    """

    # first stage
    if scenario is None:
        z = calculate_z(parking_nodes=parking_nodes, first_stage_car_moves=first_stage_car_moves,
                        second_stage_car_moves=[[]])
        profit_customer_requests = calculate_profit_customer_requests(z)
        cost_deviation_ideal_state = calculate_cost_deviation_ideal_state(parking_nodes=parking_nodes, z=z,
                                                                          first_stage_car_moves=first_stage_car_moves,
                                                                          second_stage_car_moves=[[]])

        first_stage_duplicate_for_scenarios = list(np.repeat(first_stage_car_moves, num_scenarios))
        cost_relocation = calculate_costs_relocation(first_stage_duplicate_for_scenarios, num_scenarios)

    else:
        car_moves_second_stage = [[] for _ in range(num_scenarios)]
        car_moves_second_stage[scenario] = second_stage_car_moves
        z = calculate_z(parking_nodes=parking_nodes, first_stage_car_moves=first_stage_car_moves,
                        second_stage_car_moves=car_moves_second_stage)  # , verbose=True)
        profit_customer_requests = calculate_profit_customer_requests(z, scenario=scenario)
        cost_deviation_ideal_state = calculate_cost_deviation_ideal_state(parking_nodes=parking_nodes, z=z,
                                                                          first_stage_car_moves=first_stage_car_moves,
                                                                          second_stage_car_moves=car_moves_second_stage,
                                                                          scenario=scenario)

        # first_stage_duplicate_for_scenarios = list(np.repeat(first_stage_car_moves, self.num_scenarios))
        cost_relocation = calculate_costs_relocation(first_stage_car_moves + second_stage_car_moves, num_scenarios,
                                                     individual_scenario=True)

    return profit_customer_requests - cost_relocation - cost_deviation_ideal_state


def get_objective_function_val(parking_nodes: [ParkingNode], employees: [Employee], num_scenarios: int) -> float:
    """
    :param parking_nodes: list of parking node objects, [pn1, pn2]
    :param employees: list of employees, [em1, em2]
    :param num_scenarios: int describing number of scenarios
    :return: returns float of objective value for all carmoves assigned to employees
    """

    first_stage_car_moves = []
    second_stage_car_moves = [[] for _ in range(num_scenarios)]
    for employee in employees:

        for car_move in employee.car_moves:
            first_stage_car_moves.append(car_move)

        for s in range(num_scenarios):
            for car_move in employee.car_moves_second_stage[s]:
                second_stage_car_moves[s].append(car_move)

    all_second_stage_car_moves = [cm for s in second_stage_car_moves for cm in s]
    first_stage_duplicate_for_scenarios = list(np.repeat(first_stage_car_moves, num_scenarios))
    all_car_moves = first_stage_duplicate_for_scenarios + all_second_stage_car_moves
    z = calculate_z(parking_nodes, first_stage_car_moves, second_stage_car_moves, True)
    profit_customer_requests = calculate_profit_customer_requests(z)
    cost_relocation = calculate_costs_relocation(car_moves=all_car_moves, num_scenarios=num_scenarios)
    cost_deviation_ideal_state = calculate_cost_deviation_ideal_state(parking_nodes=parking_nodes, z=z,
                                                                      first_stage_car_moves=first_stage_car_moves,
                                                                      second_stage_car_moves=second_stage_car_moves,
                                                                      scenario=None, verbose=True)

    obj_val = profit_customer_requests - cost_relocation - cost_deviation_ideal_state
    print(f"Objective function value: {round(obj_val, 2)}")

    return obj_val


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


def get_best_car_move(parking_nodes, employees, car_moves, first_stage, prioritize_charging, num_scenarios):
    # FIRST STAGE
    if first_stage:
        best_car_move_first_stage = None
        assigned_car_moves_first_stage = get_assigned_car_moves(employees)
        best_obj_val_first_stage = -1000
        longest_travel_time_first_stage = -1000
        if not prioritize_charging:
            best_obj_val_first_stage = get_obj_val_of_car_moves(parking_nodes, num_scenarios,
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
    #print(num)
    #print(sum(n.charging_state for n in self.parking_nodes))
    return min(num) == sum(n.charging_state for n in parking_nodes)