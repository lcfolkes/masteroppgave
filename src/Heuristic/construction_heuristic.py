import os
from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance

os.chdir('../InstanceGenerator')
from src.InstanceGenerator.instance_components import ParkingNode, Employee, ChargingNode, CarMove
from InstanceGenerator.world import World
from src.HelperFiles.helper_functions import load_object_from_file
from src.Gurobi.Model.run_model import run_model
import numpy as np
from itertools import product


def remove_car_move(chosen_car_move, car_moves):
    car = chosen_car_move.car.car_id
    # return list of car moves that are not associated with the car of the chosen car move
    return [cm for cm in car_moves if cm.car.car_id != car]


class ConstructionHeuristic:
    # instance_file = "InstanceFiles/6nodes/6-3-1-1_d.pkl"
    # filename = "InstanceFiles/6nodes/6-3-1-1_b.yaml"

    def __init__(self, instance_file):

        self.world_instance = load_object_from_file(instance_file)
        self.num_scenarios = self.world_instance.num_scenarios
        self.employees = self.world_instance.employees
        self.parking_nodes = self.world_instance.parking_nodes
        self.cars = self.world_instance.cars
        self.unused_car_moves = [[] for _ in range(
            self.num_scenarios)]  # [beta_s] list of unused car_moves for scenaro s (zero index)
        self.assigned_car_moves = {ks: [] for ks in product([k.employee_id for k in self.employees], [s + 1 for s in
                                                                                                      range(
                                                                                                          self.num_scenarios)])}  # [gamma_ks] dictionary containing ordered list of car_moves assigned to employee k in scenario s
        self.car_moves = []  # self.world_instance.car_moves
        self.charging_moves = []
        self.parking_moves = []
        self._initialize_car_moves()

        self.available_employees = True
        self.prioritize_charging = True
        self.first_stage = True
        self.charging_moves_second_stage = []
        self.parking_moves_second_stage = []

    def _initialize_car_moves(self):
        for car in self.world_instance.cars:
            for car_move in car.car_moves:
                self.car_moves.append(car_move)
                for s in range(self.num_scenarios):
                    self.unused_car_moves[s].append(car_move)
                if not car.needs_charging:
                    self.parking_moves.append(car_move)
            if car.needs_charging:
                fastest_time = 1000
                fastest_move = None
                for car_move in car.car_moves:
                    if car_move.handling_time < fastest_time:
                        fastest_time = car_move.handling_time
                        fastest_move = car_move
                self.charging_moves.append(fastest_move)

    def _calculate_z(self, first_stage_car_moves, second_stage_car_moves, verbose=False):
        # z is the number of customer requests served. It must be the lower of the two values
        # available cars and the number of customer requests D_is
        # number of available cars in the beginning of the second stage, y_i

        start_nodes_first_stage = [car_move.start_node for car_move in first_stage_car_moves if
                                   isinstance(car_move.end_node, ParkingNode)]
        end_nodes_first_stage = [car_move.end_node for car_move in first_stage_car_moves if
                                 isinstance(car_move.end_node, ParkingNode)]

        y = {parking_node.node_id: parking_node.parking_state for parking_node in self.parking_nodes}
        for n in start_nodes_first_stage:
            y[n.node_id] -= 1
        for n in end_nodes_first_stage:
            y[n.node_id] += 1

        node_demands = {parking_node.node_id: {'customer_requests': parking_node.customer_requests,
                                               'car_returns': parking_node.car_returns} for parking_node in
                        self.parking_nodes}

        z = {}

        start_nodes_second_stage = [[car_move.start_node.node_id for car_move in scenarios
                                     if isinstance(car_move.end_node, ParkingNode)] for scenarios in
                                    second_stage_car_moves]

        # car_move.start_node should be a list of car moves with len(list) = num_scenarios
        for n in self.parking_nodes:
            second_stage_moves_out = np.array([scenario.count(n.node_id) for scenario in start_nodes_second_stage])
            y[n.node_id] = np.maximum(y[n.node_id], 0)

            z_val = np.minimum(y[n.node_id] + node_demands[n.node_id]['car_returns'] - second_stage_moves_out,
                               node_demands[n.node_id]['customer_requests'])
            z_val = np.maximum(z_val, 0)
            z[n.node_id] = z_val
            if verbose:
                print(f"z[{n.node_id}] {z[n.node_id]}")
        return z

    def _calculate_profit_customer_requests(self, z, scenario=None):
        # sum across scenarios for all nodes
        z_sum = sum(v for k, v in z.items())
        if scenario is None:
            # print(f"z_sum {z_sum}")
            z_sum_scenario_average = np.mean(z_sum)
            return World.PROFIT_RENTAL * z_sum_scenario_average
        else:
            # print(f"z_sum[{scenario+1}] {z_sum[scenario]}")
            return World.PROFIT_RENTAL * z_sum[scenario]

    def _calculate_costs_relocation(self, car_moves, individual_scenario=False):
        # Sum of all travel times across all car moves
        sum_travel_time = sum(car_move.handling_time for car_move in car_moves)
        if individual_scenario:
            # print(f"individual_scenario {sum_travel_time}")
            return World.COST_RELOCATION * sum_travel_time
        else:
            sum_travel_time_scenario_avg = sum_travel_time / self.num_scenarios
            # print(f"sum_scenarios {sum_travel_time}")
            # print(f"avg_scenarios {sum_travel_time_scenario_avg}")
            return World.COST_RELOCATION * sum_travel_time_scenario_avg

    def _calculate_cost_deviation_ideal_state(self, z, first_stage_car_moves, second_stage_car_moves, scenario=None,
                                              verbose=False):
        # TODO: Some car moves moving into a node (maybe when only happens in one/some scenarios) are not counted in calculation of w variable.

        start_nodes_first_stage = [car_move.start_node for car_move in first_stage_car_moves if
                                   isinstance(car_move.end_node, ParkingNode)]
        end_nodes_first_stage = [car_move.end_node for car_move in first_stage_car_moves if
                                 isinstance(car_move.end_node, ParkingNode)]

        w = {n.node_id: (n.ideal_state - n.parking_state + z[n.node_id] - n.car_returns) for n in self.parking_nodes}

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

        for n in self.parking_nodes:
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
        # only return first scenario for now

    def _get_obj_val_of_car_move(self, first_stage_car_moves: [CarMove] = None,
                                 second_stage_car_moves: [CarMove] = None,
                                 scenario=None, verbose=False):
        # first stage
        if scenario is None:
            z = self._calculate_z(first_stage_car_moves=first_stage_car_moves, second_stage_car_moves=[[]])
            profit_customer_requests = self._calculate_profit_customer_requests(z)
            cost_deviation_ideal_state = self._calculate_cost_deviation_ideal_state(z,
                                                                                    first_stage_car_moves=first_stage_car_moves,
                                                                                    second_stage_car_moves=[[]])
            first_stage_duplicate_for_scenarios = list(np.repeat(first_stage_car_moves, self.num_scenarios))
            cost_relocation = self._calculate_costs_relocation(first_stage_duplicate_for_scenarios)


        else:
            car_moves_second_stage = [[] for _ in range(self.num_scenarios)]
            car_moves_second_stage[scenario] = second_stage_car_moves
            z = self._calculate_z(first_stage_car_moves=first_stage_car_moves,
                                  second_stage_car_moves=car_moves_second_stage)  # , verbose=True)
            profit_customer_requests = self._calculate_profit_customer_requests(z, scenario=scenario)
            cost_deviation_ideal_state = self._calculate_cost_deviation_ideal_state(z,
                                                                                    first_stage_car_moves=first_stage_car_moves,
                                                                                    second_stage_car_moves=car_moves_second_stage,
                                                                                    scenario=scenario)

            # first_stage_duplicate_for_scenarios = list(np.repeat(first_stage_car_moves, self.num_scenarios))
            cost_relocation = self._calculate_costs_relocation(first_stage_car_moves + second_stage_car_moves,
                                                               individual_scenario=True)

        return profit_customer_requests - cost_relocation - cost_deviation_ideal_state

    def get_objective_function_val(self):
        first_stage_car_moves = []
        second_stage_car_moves = [[] for _ in range(self.num_scenarios)]
        for employee in self.employees:

            for car_move in employee.car_moves:
                first_stage_car_moves.append(car_move)

            for s in range(self.num_scenarios):
                for car_move in employee.car_moves_second_stage[s]:
                    second_stage_car_moves[s].append(car_move)

        all_second_stage_car_moves = [cm for s in second_stage_car_moves for cm in s]
        first_stage_duplicate_for_scenarios = list(np.repeat(first_stage_car_moves, self.num_scenarios))
        all_car_moves = first_stage_duplicate_for_scenarios + all_second_stage_car_moves
        z = self._calculate_z(first_stage_car_moves, second_stage_car_moves, True)
        profit_customer_requests = self._calculate_profit_customer_requests(z)
        cost_relocation = self._calculate_costs_relocation(all_car_moves)
        cost_deviation_ideal_state = self._calculate_cost_deviation_ideal_state(z=z,
                                                                                first_stage_car_moves=first_stage_car_moves,
                                                                                second_stage_car_moves=second_stage_car_moves,
                                                                                scenario=None, verbose=True)

        obj_val = profit_customer_requests - cost_relocation - cost_deviation_ideal_state
        print("Objective function value: ", round(obj_val, 2))

        #### OBJECTIVE FUNCTION VALUE FOR SCENARIOS ####
        obj_val_list = []
        assigned_first_stage_car_moves = self._get_assigned_car_moves()
        second_stage_car_moves = []
        for s in range(self.num_scenarios):
            assigned_second_stage_car_moves = self._get_assigned_car_moves(scenario=s)
            second_stage_car_moves.append(assigned_second_stage_car_moves)
            obj_val_list.append(self._get_obj_val_of_car_move(
                first_stage_car_moves=assigned_first_stage_car_moves,
                second_stage_car_moves=assigned_second_stage_car_moves, scenario=s))
        print(f"Obj. function value of scenarios: {[round(o, 2) for o in obj_val_list]}, mean: {np.mean(obj_val_list)}")
        assert (round(np.mean(obj_val_list), 2) == round(obj_val, 2)), "Different objective function values"
        return obj_val

    def add_car_moves_to_employees(self):
        improving_car_move_exists = True
        while self.available_employees and improving_car_move_exists:
            # check if charging_moves_list is not empty
            if self.charging_moves:
                self.prioritize_charging = True
                if self.first_stage:
                    car_moves = self.charging_moves
                else:
                    car_moves = self.charging_moves_second_stage

            else:
                self.prioritize_charging = False
                if not self._check_all_charging_moves_completed():
                    print("Instance not solvable. Cannot charge all cars.")
                    break
                if self.first_stage:
                    car_moves = self.parking_moves
                else:
                    car_moves = self.parking_moves_second_stage

            if self.first_stage:
                #### GET BEST CAR MOVE ###
                best_car_move_first_stage = self._get_best_car_move(car_moves=car_moves)
                # print(best_car_move_first_stage.to_string())
                #### GET BEST EMPLOYEE ###
                best_employee_first_stage = self._get_best_employee(best_car_move=best_car_move_first_stage)
                # print(f"employee {best_employee_first_stage}")
                if best_employee_first_stage is not None:
                    #### ADD CAR MOVE TO EMPLOYEE ###
                    self._add_car_move_to_employee(car_moves=car_moves, best_car_move=best_car_move_first_stage,
                                                   best_employee=best_employee_first_stage)

            else:
                #### GET BEST CAR MOVE ###
                best_car_move_second_stage = self._get_best_car_move(car_moves=car_moves)
                if all(cm is None for cm in best_car_move_second_stage):
                    improving_car_move_exists = False
                #### GET BEST EMPLOYEE ###
                best_employee_second_stage = self._get_best_employee(best_car_move=best_car_move_second_stage)
                #### ADD CAR MOVE TO EMPLOYEE ###
                if best_employee_second_stage is not None:
                    self._add_car_move_to_employee(car_moves=car_moves, best_car_move=best_car_move_second_stage,
                                                   best_employee=best_employee_second_stage)

    def _get_assigned_car_moves(self, scenario: int = None):
        car_moves = []
        if scenario is None:
            for employee in self.employees:
                for car_move in employee.car_moves:
                    car_moves.append(car_move)
        else:
            for employee in self.employees:
                for car_move in employee.car_moves_second_stage[scenario]:
                    car_moves.append(car_move)

        return car_moves

    def _get_best_car_move(self, car_moves):
        # FIRST STAGE
        if self.first_stage:
            best_car_move_first_stage = None
            assigned_car_moves_first_stage = self._get_assigned_car_moves()
            best_obj_val_first_stage = -1000
            longest_travel_time_first_stage = -1000
            if not self.prioritize_charging:
                best_obj_val_first_stage = self._get_obj_val_of_car_move(
                    first_stage_car_moves=assigned_car_moves_first_stage)
            else:
                for r in range(len(car_moves)):
                    travel_time = car_moves[r].handling_time
                    if travel_time > longest_travel_time_first_stage:
                        longest_travel_time_first_stage = travel_time
                        best_car_move_first_stage = car_moves[r]
            if not self.prioritize_charging:
                for r in range(len(car_moves)):
                    obj_val = self._get_obj_val_of_car_move(
                        first_stage_car_moves=assigned_car_moves_first_stage + [car_moves[r]])
                    if obj_val > best_obj_val_first_stage:
                        best_obj_val_first_stage = obj_val
                        best_car_move_first_stage = car_moves[r]

            # print("obj_val: ", obj_val)
            # print("best_obj_val: ", best_obj_val_first_stage)
            # print(f"best_car_move: {best_car_move_first_stage.car_move_id}, {best_car_move_first_stage.start_node.node_id} --> {best_car_move_first_stage.end_node.node_id}")
            return best_car_move_first_stage

        # SECOND STAGE
        #TODO: Handle assigning charging moves in the second stage
        else:
            best_car_move_second_stage = [None for _ in range(self.num_scenarios)]
            best_obj_val_second_stage = [-1000 for _ in range(self.num_scenarios)]
            assigned_first_stage_car_moves = self._get_assigned_car_moves()

            if not self.prioritize_charging:
                for s in range(self.num_scenarios):
                    assigned_second_stage_car_moves = self._get_assigned_car_moves(scenario=s)
                    best_obj_val_second_stage[s] = self._get_obj_val_of_car_move(
                        first_stage_car_moves=assigned_first_stage_car_moves,
                        second_stage_car_moves=assigned_second_stage_car_moves, scenario=s)
            # print(f"best_obj_val_second_stage {best_obj_val_second_stage}")
            obj_val = [0 for _ in range(self.num_scenarios)]
            for s in range(self.num_scenarios):
                # zero indexed scenario
                assigned_second_stage_car_moves = self._get_assigned_car_moves(scenario=s)
                for r in range(len(car_moves[s])):
                    obj_val[s] = self._get_obj_val_of_car_move(first_stage_car_moves=assigned_first_stage_car_moves,
                                                               second_stage_car_moves=assigned_second_stage_car_moves +
                                                                                      [car_moves[s][r]], scenario=s)
                    #if car_moves[s][r].car_move_id == 7 and s == 0:
                    #    print(f"car_move {car_moves[s][r].car_move_id}, s {s + 1}")
                    #    print(f"obj_val {obj_val[s]} best_obj_val {best_obj_val_second_stage[s]}")

                    if obj_val[s] > best_obj_val_second_stage[s]:
                        best_obj_val_second_stage[s] = obj_val[s]
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

    def _get_best_employee(self, best_car_move):
        if self.first_stage:
            best_employee = None
            best_travel_time_to_car_move = 100
            end_node = best_car_move.start_node
        else:
            best_employee_second_stage = [None for _ in range(self.num_scenarios)]
            best_travel_time_to_car_move_second_stage = [100 for _ in range(self.num_scenarios)]
            end_node = [(cm.start_node if cm is not None else cm) for cm in best_car_move]

        best_move_not_legal = True

        for employee in self.employees:
            task_num = len(employee.car_moves)
            # if first stage and the number of completed task for employee is below the number of tasks in first stage,
            # or if second stage and the number of completed tasks are the same or larger than the number of tasks in first stage
            if self.first_stage == (task_num < self.world_instance.first_stage_tasks):
                if self.first_stage:
                    legal_move = self.world_instance.check_legal_move(car_move=best_car_move, employee=employee)
                    print(f"legal_move {legal_move}")
                    if legal_move:
                        best_move_not_legal = False
                        start_node = employee.current_node
                        travel_time_to_car_move = self.world_instance.get_employee_travel_time_to_node(start_node,
                                                                                                       end_node)
                        if travel_time_to_car_move < best_travel_time_to_car_move:
                            best_travel_time_to_car_move = travel_time_to_car_move
                            best_employee = employee

                else:
                    for s in range(self.num_scenarios):
                        if best_car_move[s] is not None:
                            legal_move = self.world_instance.check_legal_move(
                                car_move=best_car_move[s], employee=employee, scenario=s)
                            if legal_move:
                                best_move_not_legal = False
                                start_node = employee.current_node_second_stage[s]
                                travel_time_to_car_move = self.world_instance.get_employee_travel_time_to_node(
                                    start_node, end_node[s])
                                if travel_time_to_car_move < best_travel_time_to_car_move_second_stage[s]:
                                    best_travel_time_to_car_move_second_stage[s] = travel_time_to_car_move
                                    best_employee_second_stage[s] = employee

        # Remove best move if not legal. Else return best employee
        if self.first_stage:
            if best_move_not_legal:
                if self.prioritize_charging:
                    self.charging_moves.remove(best_car_move)
                else:
                    self.parking_moves.remove(best_car_move)
                return
            else:
                return best_employee
        else:
            if best_move_not_legal:
                if self.prioritize_charging:
                    for s in range(self.num_scenarios):
                        self.charging_moves_second_stage[s] = [cm for cm in self.charging_moves_second_stage[s] if
                                                               cm != best_car_move[s]]
                else:
                    for s in range(self.num_scenarios):
                        self.parking_moves_second_stage[s] = [cm for cm in self.parking_moves_second_stage[s] if
                                                              cm != best_car_move[s]]
                return
            else:
                return best_employee_second_stage

    # print(best_travel_time_to_car_move_second_stage)

    def _add_car_move_to_employee(self, car_moves, best_car_move, best_employee):
        if self.first_stage:
            if best_employee is not None:
                print('\nEmployee id', best_employee.employee_id)
                print('Employee node before', best_employee.current_node.node_id)
                print('Employee time before', best_employee.current_time)
                # print('Travel time to start node', best_travel_time_to_car_move)
                print(best_car_move.to_string())
                self.world_instance.add_car_move_to_employee(best_car_move, best_employee)
                for s in range(self.num_scenarios):
                    self.assigned_car_moves[(best_employee.employee_id, s + 1)].append(best_car_move)
                    self.unused_car_moves[s].remove(best_car_move)
                print('Employee node after', best_employee.current_node.node_id)
                print('Employee time after', best_employee.current_time)
                if self.prioritize_charging:
                    self.charging_moves = remove_car_move(best_car_move,
                                                          car_moves)  # should remove car move and other car-moves with the same car
                else:
                    self.parking_moves = remove_car_move(best_car_move,
                                                         car_moves)  # should remove car move and other car-moves with the same car

                self.first_stage = False
                for employee in self.employees:
                    task_num = len(employee.car_moves)
                    if task_num < self.world_instance.first_stage_tasks:
                        self.first_stage = True
                if not self.first_stage:
                    # initialize charging and parking moves for second stage
                    self.charging_moves_second_stage = [self.charging_moves for s in range(self.num_scenarios)]
                    self.parking_moves_second_stage = [self.parking_moves for s in range(self.num_scenarios)]
            else:
                self.available_employees = False
        # Second stage
        else:
            # print(best_car_move_second_stage)
            # If any employee is not note, continue
            if not all(e is None for e in best_employee):
                for s in range(self.num_scenarios):
                    # print(best_employee_second_stage[s].to_string())
                    if best_employee[s] is not None:
                        print('\nEmployee id', best_employee[s].employee_id)
                        print('Scenario', s + 1)
                        print('Employee node before', best_employee[s].current_node_second_stage[s].node_id)
                        print('Employee time before', best_employee[s].current_time_second_stage[s])
                        # print('Travel time to start node', best_travel_time_to_car_move_second_stage[s])
                        print(best_car_move[s].to_string())
                        self.world_instance.add_car_move_to_employee(best_car_move[s], best_employee[s], s)
                        if best_car_move[s] is not None:
                            self.assigned_car_moves[(best_employee[s].employee_id, s + 1)].append(best_car_move[s])
                            self.unused_car_moves[s].remove(best_car_move[s])
                        print('Employee node after', best_employee[s].current_node_second_stage[s].node_id)
                        print('Employee time after', best_employee[s].current_time_second_stage[s])
                        # When first stage is finished, initialize car_moves to be list of copies of car_moves (number of copies = num_scenarios)
                        if self.prioritize_charging:
                            self.charging_moves_second_stage[s] = remove_car_move(best_car_move[s],
                                                                                  car_moves[
                                                                                      s])  # should remove car move and other car-moves with the same car
                        else:
                            self.parking_moves_second_stage[s] = remove_car_move(best_car_move[s],
                                                                                 car_moves[
                                                                                     s])  # should remove car move and other car-moves wit
                # print(f"car_moves: {len(car_moves[s])}")
                if not any(self.parking_moves_second_stage):
                    self.available_employees = False
            else:
                self.available_employees = False

    def _check_all_charging_moves_completed(self) -> bool:
        num = [0 for _ in range(self.num_scenarios)]
        for employee in self.employees:
            for car_move in employee.car_moves:
                if isinstance(car_move.end_node, ChargingNode):
                    num = [n + 1 for n in num]
            if not self.first_stage:
                for i in range(self.num_scenarios):
                    for car_move in employee.car_moves_second_stage[i]:
                        if isinstance(car_move.end_node, ChargingNode):
                            num[i] += 1
        # returns whether the number of charging moves for the scenario with the lowest number of charging moves assigned
        # equals the sum of cars in need of charging
        print(num)
        print(sum(n.charging_state for n in self.parking_nodes))
        return min(num) == sum(n.charging_state for n in self.parking_nodes)

    # print(f"obj_val: {obj_val}")

    def print_solution(self):

        print("-------------- First stage routes --------------")
        for employee in self.employees:
            for car_move in employee.car_moves:
                print(f"employee: {employee.employee_id}, " + car_move.to_string())

        print("-------------- Second stage routes --------------")
        for employee in self.employees:
            if any(employee.car_moves_second_stage):
                for s in range(self.num_scenarios):
                    for car_move in employee.car_moves_second_stage[s]:
                        print(f"employee: {employee.employee_id}, scenario: {s + 1} " + car_move.to_string())


filename = "InstanceFiles/6nodes/6-1-1-1_b"

print("\n---- HEURISTIC ----")
ch = ConstructionHeuristic(filename + ".pkl")
# try:
ch.add_car_moves_to_employees()
ch.print_solution()
#ch.get_objective_function_val()
print(ch.assigned_car_moves)
print(ch.unused_car_moves)
print("\n---- GUROBI ----")
#gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=False)
gi = GurobiInstance(filename + ".yaml")
run_model(gi)
# except:
#    print("Instance not solvable")
