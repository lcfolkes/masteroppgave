class Node:
    def __init__(self, node_id: int, x_coordinate: int, y_coordinate: int):
        self.node_id = node_id
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate

class ParkingNode(Node):
    # change x_coordinate and y_coordinate to location dict?
    def __init__(self, node_id: int, x_coordinate: int, y_coordinate: int, parking_state: int, charging_state: int, ideal_state: int):
        super().__init__(node_id, x_coordinate, y_coordinate)
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
    def __init__(self, node_id: int, parking_node: ParkingNode, capacity, max_capacity: int):
        super().__init__(node_id, parking_node.x_coordinate, parking_node.y_coordinate)
        self.capacity = capacity
        self.max_capacity = max_capacity
        self.parking_node = parking_node

class Employee:
    # find better word for "handling". perhaps occupied or busy is better?
    def __init__(self, start_node: Node, start_time: int, handling: bool):
        self.start_node = start_node
        self.start_time = start_time
        self.handling = handling


class Car:
    def __init__(self, parking_node: ParkingNode, start_time: float, is_charging: bool):
        self.parking_node = parking_node
        self.start_time = start_time
        self.is_charging = is_charging
        self.destinations = None

    # TODO: make sure destinations works correctly
    def set_destinations(self, destination: [Node]):
        self.destinations = destination

