class ParkingNode:
    # change x_coordinate and y_coordinate to location dict?
    def __init__(self, x_coordinate: int, y_coordinate: int, parking_state: int, charging_state: int, ideal_state: int):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.parking_state = parking_state
        self.charging_state = charging_state
        self.ideal_state = ideal_state
        self.customer_requests = None
        self.car_returns = None

    def set_customer_requests(self, value: [int]):
        self.customer_requests = value

    def set_car_returns(self, value: [int]):
        self.car_returns = value


class ChargingNode:
    # should parking node instead be the parking node object?
    def __init__(self, x_coordinate: float, y_coordinate: float, capacity: int, max_capacity: int, parking_node: int):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.capacity = capacity
        self.max_capacity = max_capacity
        self.parking_node = parking_node

class Employee:
    # find better word for "handling". perhaps occupied or busy is better?
    def __init__(self, start_node: int, start_time: int, handling: bool):
        self.start_node = start_node
        self.start_time = start_time
        self.handling = handling


class Car:
    def __init__(self, parking_node: int, start_time: float, is_charging: bool):
        self.parking_node = parking_node
        self.start_time = start_time
        self.is_charging = is_charging
        self.destinations = None

    def set_destinations(self, value: [int]):
        self.destinations = value
