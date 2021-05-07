import itertools

import numpy as np

from HelperFiles.helper_functions import read_config
import os
from path_manager import path_to_src

os.chdir(path_to_src)

TIME_CONSTANTS = read_config('InstanceGenerator/world_constants_config.yaml')['time_constants']

# NODE OBJECT DOES NOT NEED TO BE DEEP COPIED EVERY TIME
'''
class Node:
    id_iter = itertools.count(start=1)

    def __init__(self, x_coordinate: int, y_coordinate: int):
        self.node_id = next(self.id_iter)
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
'''


class Node:
    id_iter = itertools.count(start=1)

    def __init__(self):
        self.node_id = next(self.id_iter)

    def get_id(self):
        return self.node_id


# class ParkingNode(Node):
class ParkingNode(Node):
    # Should it be possible to update parking state?
    # def __init__(self, x_coordinate: int, y_coordinate: int, parking_state: int, charging_state: int, ideal_state: int):
    def __init__(self, node_nr: int, parking_state: int, charging_state: int, ideal_state: int):
        # super().__init__(x_coordinate, y_coordinate)
        super().__init__()
        self.node_nr = node_nr
        self.parking_state = parking_state
        self.charging_state = charging_state
        self.ideal_state = ideal_state
        self.customer_requests = None
        self.car_returns = None

    def set_customer_requests(self, value: np.array([int])):
        self.customer_requests = value

    def set_car_returns(self, value: np.array([int])):
        self.car_returns = value

    def get_nr(self):
        return self.node_nr


# class ChargingNode(Node):
class ChargingNode(Node):
    def __init__(self, parking_node: ParkingNode, capacity):
        super().__init__()
        # super().__init__(parking_node.x_coordinate, parking_node.y_coordinate)
        self.capacity = capacity
        # self.capacities = None
        self.num_charging = None
        self.parking_node = parking_node

    def initialize_charging_node_state(self, num_scenarios: [int]):
        self.num_charging = [0 for _ in range(num_scenarios)]

    # self.capacities = [self.capacity for _ in range(num_scenarios)]

    def add_car(self, scenario: int = None):
        if scenario is not None:
            if self.num_charging[scenario] == self.capacity:  # self.capacities[scenario]:

                raise Exception("No cars can be added to charging node as the capacity is reached")
            else:
                self.num_charging[scenario] += 1
        else:
            for s in range(len(self.num_charging)):
                if self.num_charging[s] == self.capacity:  # self.capacities[s]:
                    raise Exception("No cars can be added to charging node as the capacity is reached")
                else:
                    self.num_charging[s] += 1


    def remove_car(self, scenario: int = None):
        if scenario is not None:
            if self.num_charging[scenario] == 0:
                #print(f"Car move {} can not be removed from node {} as there are no cars there")
                raise Exception("No cars can be removed from charging node as there are no cars there")
            else:
                self.num_charging[scenario] -= 1
        else:
            for s in range(len(self.num_charging)):
                if self.num_charging[s] == 0:
                    raise Exception("No cars can be removed from charging node as there are no cars there")
                else:
                    self.num_charging[s] -= 1


    def reset(self, scenario: int = None):
        if scenario is not None:
            self.num_charging[scenario] = 0
        else:
            self.num_charging = [0 for _ in self.num_charging]


class Car:
    id_iter = itertools.count(start=1)

    def __init__(self, parking_node: ParkingNode, start_time: float, needs_charging: bool):
        self.car_id = next(self.id_iter)
        self.parking_node = parking_node
        self.start_time = start_time
        self.needs_charging = needs_charging
        self.destinations = []
        self.car_moves = []

    def set_destinations(self, destinations: [Node]):
        self.destinations = destinations
        self._set_car_moves(destinations)

    def _set_car_moves(self, destinations):
        car_moves = []
        for destination in destinations:
            cm = CarMove(self, self.parking_node, destination)
            car_moves.append(cm)
        self.car_moves = car_moves


# earliest start time should be added
class CarMove:
    id_iter = itertools.count(start=1)

    def __init__(self, car: Car, start_node: ParkingNode, end_node: Node):
        self.car_move_id = next(self.id_iter)
        self.car = car
        self.start_node = start_node  # origin node. this could be derived from car object
        self.end_node = end_node  # destination node
        self.handling_time = None
        self.employee = None
        self.employee_second_stage = []
        self.is_charging_move = (True if isinstance(end_node, ChargingNode) else False)
        self.start_time = None
        self.start_times_second_stage = []

    def set_travel_time(self, time: int):
        if isinstance(self.end_node, ParkingNode):
            self.handling_time = time + TIME_CONSTANTS['handling_parking']
        elif isinstance(self.end_node, ChargingNode):
            self.handling_time = time + TIME_CONSTANTS['handling_charging']

    def set_start_time(self, time: int, scenario=None):
        if scenario is None:
            self.start_time = time
        else:
            self.start_times_second_stage[scenario] = time

    def reset_start_time(self, scenario=None):
        if scenario is None:
            self.start_time = None
        else:
            self.start_times_second_stage[scenario] = None

    def set_employee(self, employee, scenario=None):
        if scenario is None:
            self.employee = employee
            if self.is_charging_move:
                self.end_node.add_car()

        else:
            try:
                self.employee_second_stage[scenario] = employee

            except:
                self.initialize_second_stage(num_scenarios=len(employee.car_moves_second_stage))
                self.employee_second_stage[scenario] = employee

            # Update charging state in end node if the chosen move is a charging move
            if self.is_charging_move:
                self.end_node.add_car(scenario)

    def reset(self, scenario=None):
        if scenario is None:
            if self.is_charging_move and all(i > 0 for i in self.end_node.num_charging):
                self.end_node.remove_car()
            self.employee = None
            self.start_time = None

        else:
            if self.is_charging_move and self.end_node.num_charging[scenario] > 0:
                self.end_node.remove_car(scenario)
            self.employee_second_stage[scenario] = []
            self.start_times_second_stage[scenario] = []



    def initialize_second_stage(self, num_scenarios: int):
        for _ in range(num_scenarios):
            self.employee_second_stage.append(None)
            self.start_times_second_stage.append(None)

    def to_string(self):
        '''
        if self.is_charging_move:
            s = "C: "
        else:
            s = "P: "
        '''
        return f"Car move: {self.car_move_id}, Car: {self.car.car_id}, Route: ({self.start_node.node_id} -> {self.end_node.node_id})"
                   #f"" \
                   #f"Handling time: {self.handling_time}"


class Employee:
    id_iter = itertools.count(start=1)

    def __init__(self, start_node: Node, start_time: int, handling: bool):

        self.employee_id = next(self.id_iter)
        self.start_node: Node = start_node
        self.current_node: Node = start_node
        self.current_node_second_stage: [Node] = []
        self.start_time = start_time
        self.current_time = start_time
        self.current_time_second_stage = []
        self.handling = handling
        self.car_moves = []
        self.start_times_car_moves = []
        self.travel_times_car_moves = []
        self.car_moves_second_stage = []
        self.start_times_car_moves_second_stage = []
        self.travel_times_car_moves_second_stage = []

    def add_car_move(self, total_travel_time: float, car_move: CarMove, scenario: int = None):
        if scenario is None:
            car_move.set_employee(self)
            self.current_time += total_travel_time
            self.current_node = car_move.end_node
            self.car_moves.append(car_move)
            self.start_times_car_moves.append(car_move.start_time)
            self.travel_times_car_moves.append(total_travel_time - car_move.handling_time)

        else:
            # zero-indexed scenario
            self.current_time_second_stage[scenario] += total_travel_time
            # print(f"e_id: {self.employee_id}, second_current_time: {self.current_time_second_stage}")
            self.current_node_second_stage[scenario] = car_move.end_node
            self.car_moves_second_stage[scenario].append(car_move)

            # first move in second stage in the scenario
            if len(self.start_times_car_moves_second_stage[scenario]) == 0:
                self.start_times_car_moves_second_stage[scenario].append(self.current_time + total_travel_time
                                                                         - car_move.handling_time)

            # not the first move in the second stage in the scenario
            else:
                self.start_times_car_moves_second_stage[scenario].append(
                    self.start_times_car_moves_second_stage[scenario][-1]
                    + self.car_moves_second_stage[scenario][-2].handling_time
                    + total_travel_time - car_move.handling_time)
            self.travel_times_car_moves_second_stage[scenario].append(total_travel_time - car_move.handling_time)
            car_move.set_employee(self, scenario)

    def initialize_second_stage(self, num_scenarios: int):
        for s in range(num_scenarios):
            self.current_node_second_stage.append(self.current_node)
            self.current_time_second_stage.append(self.current_time)
            self.car_moves_second_stage.append([])
            self.start_times_car_moves_second_stage.append([])
            self.travel_times_car_moves_second_stage.append([])


    def remove_last_car_move(self, total_travel_time: float):
        self.current_time -= total_travel_time
        cm = self.car_moves.pop()
        cm.employee = None
        try:
            self.current_node = self.car_moves[-1].end_node
        except:
            self.current_node = self.start_node

    def reset(self):
        for cm in self.car_moves:
            cm.reset()
        for s, scenario in enumerate(self.car_moves_second_stage):
            for cm in scenario:
                cm.reset(scenario=s)
        '''        
        for i in range(len(self.start_times_car_moves)):
            self.start_times_car_moves[i] = []
            self.travel_times_car_moves[i] = []
        for s in range(len(self.start_times_car_moves_second_stage)):
            for i in range(len(self.start_times_car_moves_second_stage[s])):
                self.start_times_car_moves_second_stage[s][i] = []
                self.travel_times_car_moves_second_stage[s][i] = []
        '''
        self.current_node = self.start_node
        self.current_node_second_stage = []
        self.current_time = self.start_time
        self.current_time_second_stage = []
        self.car_moves = []
        self.start_times_car_moves = []
        self.travel_times_car_moves = []
        self.car_moves_second_stage = []
        self.start_times_car_moves_second_stage = []
        self.travel_times_car_moves_second_stage = []

    def to_string(self):
        return f"employee_id: {self.employee_id}\t start_node: {self.start_node.node_id}\t current_node: {self.current_node.node_id}" \
               f"\tcurrent_node_second_stage: {[n.node_id for n in self.current_node_second_stage]} \n current_time: {self.current_time}" \
               f"\tcurrent_time_second_stage: {self.current_time_second_stage}\n car_moves: {[cm.car_move_id for cm in self.car_moves]}" \
               f"\tcar_moves_second_stage: {[[cm.car_move_id for cm in car_moves] for car_moves in self.car_moves_second_stage]}"
