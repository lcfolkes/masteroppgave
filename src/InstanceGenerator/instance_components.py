import itertools
from src.HelperFiles.helper_functions import read_config
import os
from path_manager import path_to_src
os.chdir(path_to_src)


TIME_CONSTANTS = read_config('InstanceGenerator/world_constants_config.yaml')['time_constants']


class Node:
    id_iter = itertools.count(start=1)

    def __init__(self, x_coordinate: int, y_coordinate: int):
        self.node_id = next(self.id_iter)
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate


class ParkingNode(Node):
    # Should it be possible to update parking state?
    def __init__(self, x_coordinate: int, y_coordinate: int, parking_state: int, charging_state: int, ideal_state: int):
        super().__init__(x_coordinate, y_coordinate)
        self.parking_state = parking_state
        self.charging_state = charging_state
        self.ideal_state = ideal_state
        self.customer_requests = None
        self.car_returns = None

    def set_customer_requests(self, value: [int]):
        self.customer_requests = value

    def set_car_returns(self, value: [int]):
        self.car_returns = value


class ChargingNode(Node):
    def __init__(self, parking_node: ParkingNode, capacity, max_capacity: int):
        super().__init__(parking_node.x_coordinate, parking_node.y_coordinate)
        self.capacity = capacity
        self.max_capacity = max_capacity
        self.parking_node = parking_node


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

#earliest start time should be added
class CarMove:
    id_iter = itertools.count(start=1)

    def __init__(self, car: Car, start_node: ParkingNode, end_node: Node):
        self.car_move_id = next(self.id_iter)
        self.car = car
        self.start_node = start_node  # origin node. this could be derived from car object
        self.end_node = end_node  # destination node
        self.handling_time = None


    # self.employee = None

    def set_travel_time(self, time: int):
        if isinstance(self.end_node, ParkingNode):
            self.handling_time = time + TIME_CONSTANTS['handling_parking']
        elif isinstance(self.end_node, ChargingNode):
            self.handling_time = time + TIME_CONSTANTS['handling_charging']

    def to_string(self):
        return f"car_move_id: {self.car_move_id}, car: {self.car.car_id}, start_node: {self.start_node.node_id}, end_node: {self.end_node.node_id}, " \
               f"handling_time: {self.handling_time}"


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
        self.car_moves_second_stage = []


    def add_car_move(self, total_travel_time: float, car_move: CarMove, scenario: int = None):
        if scenario is None:
            self.current_time += total_travel_time
            self.current_node = car_move.end_node
            self.car_moves.append(car_move)
        else:
            # zero-indexed scenario
            self.current_time_second_stage[scenario] += total_travel_time
            #print(f"e_id: {self.employee_id}, second_current_time: {self.current_time_second_stage}")
            self.current_node_second_stage[scenario] = car_move.end_node
            self.car_moves_second_stage[scenario].append(car_move)

    def initialize_second_stage(self, num_scenarios: int):
        for s in range(num_scenarios):
            self.current_node_second_stage.append(self.current_node)
            self.current_time_second_stage.append(self.current_time)
            self.car_moves_second_stage.append([])

    def remove_last_car_move(self, total_travel_time: float):
        self.current_time -= total_travel_time
        self.car_moves.pop()
        try:
            self.current_node = self.car_moves[-1].end_node
        except:
            self.current_node = self.start_node

    def reset(self):
        self.__init__(start_node=self.start_node, start_time=self.start_time, handling=self.handling)

    def to_string(self):
         return  f"employee_id: {self.employee_id}\t start_node: {self.start_node.node_id}\t current_node: {self.current_node.node_id}" \
              f"\tcurrent_node_second_stage: {[n.node_id for n in self.current_node_second_stage]} \n current_time: {self.current_time}" \
              f"\tcurrent_time_second_stage: {self.current_time_second_stage}\n car_moves: {[cm.car_move_id for cm in self.car_moves]}" \
              f"\tcar_moves_second_stage: {[[cm.car_move_id for cm in car_moves] for car_moves in self.car_moves_second_stage]}"
