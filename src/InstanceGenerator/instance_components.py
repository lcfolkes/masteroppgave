import itertools
from src.HelperFiles.helper_functions import read_config
import os
#os.chdir('../')
print(os.getcwd())


TIME_CONSTANTS = read_config('./world_constants_config.yaml')['time_constants']


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
        self.start_node = start_node
        self.current_node = start_node
        self.start_time = start_time
        self.current_time = start_time
        self.handling = handling
        self.car_moves = []

    def add_car_move(self, total_travel_time: float, car_move: CarMove):
        self.current_time += total_travel_time
        self.current_node = car_move.end_node
        self.car_moves.append(car_move)
