class ParkingNode:

    # change x_coordinate and y_coordinate to location dict?
    def __init__(self, x_coordinate, y_coordinate, parking_state,
                 charging_state, ideal_state, customer_reqests, car_returns):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.parking_state = parking_state
        self.charging_state = charging_state
        self.ideal_state = ideal_state
        self.customer_reqests = customer_reqests
        self.car_returns = car_returns

class ChargingNode:

    def __init__(self,  x_coordinate: float, y_coordinate: float, capacity: int, max_capacity: int, parking_node: ParkingNode):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.capacity = capacity
        self.max_capacity = max_capacity
        self.parking_node = parking_node

class Employee:
    # find better word for "handling". perhaps occupied or busy is better?
    def __init__(self, start_node: ParkingNode, start_time: float, handling: bool):
        self.start_node = start_node
        self.start_time = start_time
        self.handling = handling

class Car:
    def __init__(self, start_time, parking_node, is_charging: bool):
        self.start_time = start_time
        self.parking_node = parking_node
        self.destinations = []
        self.is_charging = is_charging