from src.HelperFiles.helper_functions import read_config
from src.InstanceGenerator.instance_components import ParkingNode, ChargingNode, Employee, Car, CarMove
import numpy as np
import pandas as pd
import math
import random
import time
import copy
import os
from path_manager import path_to_src

os.chdir(path_to_src)

DISTANCESCALE = 3


class World:
    cf = read_config('InstanceGenerator/world_constants_config_new.yaml')
    # COST CONSTANTS #
    PROFIT_RENTAL = cf['objective_function']['profit_rental']
    COST_RELOCATION = cf['objective_function']['cost_relocation']
    COST_DEVIATION = cf['objective_function']['cost_deviation']

    # TIME CONSTANTS #
    HANDLING_TIME_PARKING = cf['time_constants']['handling_parking']
    HANDLING_TIME_CHARGING = cf['time_constants']['handling_charging']
    PLANNING_PERIOD = cf['time_constants']['planning_period']

    # NODE STATES
    CHARGING_STATE = cf['node_states']['charging']

    def __init__(self):

        # TASKS
        self.tasks = 0
        self.first_stage_tasks = 0

        # SCENARIOS
        self.num_scenarios = 0

        # ENTITIES #
        self.employees = []
        self.nodes = []
        self.parking_nodes = []
        self.charging_nodes = []
        self.distances_public_bike = []
        self.distances_car = []
        self.cars = []
        self.car_moves = []
        self.bigM = []

    def _set_num_scenarios(self, n: int):
        self.num_scenarios = n

    def _set_num_tasks(self, n: int):
        self.tasks = n

    def _set_num_first_stage_tasks(self, n: int):
        self.first_stage_tasks = n

    def _add_node(self, node):
        self.nodes.append(node)

    def _add_parking_node(self, parking_node: ParkingNode):
        self.parking_nodes.append(parking_node)

    def _add_charging_node(self, charging_node: ChargingNode):
        self.charging_nodes.append(charging_node)

    def _add_employee(self, employee: Employee):
        self.employees.append(employee)

    def _add_car(self, car: Car):
        self.cars.append(car)

    def _add_car_move(self, car_move: CarMove):
        self.car_moves.append(car_move)

    # customer requests in the second stage and
    # customer deliveries/car returns in the second stage
    def _set_demands(self):
        customer_requests = [k for k in World.CUSTOMER_REQUESTS]
        customer_requests_prob = [World.CUSTOMER_REQUESTS[k] for k in World.CUSTOMER_REQUESTS]
        car_returns = [k for k in World.CAR_RETURNS]
        car_returns_prob = [World.CAR_RETURNS[k] for k in World.CAR_RETURNS]
        for i in range(len(self.parking_nodes)):
            # np.random.seed(i)
            self.parking_nodes[i].set_customer_requests(
                np.random.choice(customer_requests, size=self.num_scenarios, p=customer_requests_prob))
            self.parking_nodes[i].set_car_returns(
                np.random.choice(car_returns, size=self.num_scenarios, p=car_returns_prob))

    def add_car_move_to_employee(self, car_move: CarMove, employee: Employee, scenario: int = None):

        if scenario is None:
            travel_time_to_car_move = self.get_employee_travel_time_to_node(start_node=employee.current_node,
                                                                            end_node=car_move.start_node)
            car_move_start_time = employee.current_time + travel_time_to_car_move
            total_travel_time = travel_time_to_car_move + car_move.handling_time
            car_move.set_start_time(car_move_start_time)
            employee.add_car_move(total_travel_time, car_move)
            if len(employee.car_moves) == self.first_stage_tasks:
                employee.initialize_second_stage(num_scenarios=self.num_scenarios)
        else:
            # zero-indexed scenario
            travel_time_to_car_move = self.get_employee_travel_time_to_node(
                start_node=employee.current_node_second_stage[scenario], end_node=car_move.start_node)
            car_move_start_time = employee.current_time_second_stage[scenario] + travel_time_to_car_move
            total_travel_time = travel_time_to_car_move + car_move.handling_time
            car_move.set_start_time(car_move_start_time)
            employee.add_car_move(total_travel_time=total_travel_time, car_move=car_move, scenario=scenario)

    def remove_car_move_from_employee(self, car_move: CarMove, employee: Employee):
        # start_node is the end node of the car move performed before the one we want to remove.
        # what if the list is empty? then take start_node
        try:
            total_travel_time = self.get_employee_travel_time_to_node(start_node=employee.car_moves[-2].end_node,
                                                                      end_node=car_move.start_node) + car_move.handling_time
        except:
            total_travel_time = self.get_employee_travel_time_to_node(start_node=employee.start_node,
                                                                      end_node=car_move.start_node) + car_move.handling_time

        employee.remove_last_car_move(total_travel_time)

    def get_employee_travel_time_to_node(self, start_node: ParkingNode, end_node: ParkingNode):
        employee_start_node = start_node.node_id - 1
        employee_end_node = end_node.node_id - 1
        return self.distances_public_bike[employee_start_node * len(self.nodes) + employee_end_node]

    ## CALCULATE DISTANCE ##

    def calculate_distances(self):

        # TODO:Distance matrix
        for x in range(len(self.nodes)):
            for y in range(len(self.nodes)):
                # Creating some distance between charging and parking nodes
                if (int(distance) == 0 and x != y):
                    distanceSq = 1.0
                    distance_public_bike = 1.0

                self.distances_car.append(distanceSq)
                self.distances_public_bike.append(distance_public_bike)

    ## CALCULATE BIGM
    def calculate_bigM(self):
        for i in range(len(self.cars)):
            for j in range(len(self.cars[i].destinations)):
                max_diff = 0
                for l in range(len(self.cars)):
                    for k in range(len(self.cars[l].destinations)):
                        for x in range(len(self.parking_nodes)):
                            distances1 = self.distances_public_bike[
                                (len(self.nodes) * (self.cars[i].destinations[j].node_id - 1)) + x]
                            distances2 = self.distances_public_bike[
                                (len(self.nodes) * (self.cars[l].destinations[k].node_id - 1)) + x]
                            handling_time = self.distances_car[
                                len(self.nodes) * (self.cars[l].parking_node.node_id - 1) + self.cars[l].destinations[
                                    k].node_id - 1]
                            diff = distances1 - (distances2 + handling_time)
                            if (diff > max_diff):
                                max_diff = diff
                bigM_diff = max_diff
                bigM = float(format(bigM_diff, '.1f'))
                self.bigM.append(bigM)

    ## CALCULATE VISITS ##

    def calculate_initial_add(self):
        # Initial theta is the initial number of employees at each node (employees on its way to a node included)
        initial_theta = [0 for i in range(len(self.nodes))]
        # Initial handling is a list of employees coming in to each node after handling car
        # if finishing up a task, an employee and a car will arrive a this node
        initial_handling = [0 for i in range(len(self.nodes))]
        for j in range(len(self.employees)):
            initial_theta[self.employees[j].start_node.node_id - 1] += 1
            if (self.employees[j].handling):
                initial_handling[self.employees[j].start_node.node_id - 1] += 1
        return initial_theta, initial_handling

    ## SCALE IDEAL STATE ##

    #
    def create_real_ideal_state(self):
        initial_add = [0 for i in range(len(self.nodes))]
        for j in range(len(self.employees)):
            if (self.employees[j].handling):
                initial_add[self.employees[j].start_node.node_id - 1] += 1

        sum_ideal_state = 0
        sum_parking_state = 0
        # net sum of Requests-Deliveries for each scenario
        for i in range(len(self.parking_nodes)):
            sum_ideal_state += self.parking_nodes[i].ideal_state
            sum_parking_state += self.parking_nodes[i].parking_state + initial_add[i]

        # ideal state should be scaled to sum of available cars in the worst case

        # Setter max outflow i hver node til 1.1, for å få litt mer spennende instanser
        max_flow = math.ceil(0.6 * len(self.parking_nodes))
        # max_flow = (max(DEMAND)-min(DELIVERIES))*len(self.parking_nodes)

        sum_ideal_state += max_flow
        # sum_ideal_state += int(max(netFlowScenarios))
        sum_parking_state_after = 0

        for i in range(len(self.parking_nodes)):
            # Scale parking_state with rounded share of sum_parking_state
            # Share of parking_state = iptate/sum_parking_state
            self.parking_nodes[i].parking_state = int(
                round(float(sum_ideal_state) * (float(self.parking_nodes[i].parking_state) / sum_parking_state)))
            sum_parking_state_after += self.parking_nodes[i].parking_state

        # Correct for errors due to rounding
        while (sum_parking_state_after != sum_ideal_state):
            if (sum_parking_state_after < sum_ideal_state):
                r = random.randint(0, len(self.parking_nodes) - 1)
                self.parking_nodes[r].parking_state += 1
                sum_parking_state_after += 1
            else:
                r = random.randint(0, len(self.parking_nodes) - 1)
                if (self.parking_nodes[r].parking_state - initial_add[r] > 0):
                    self.parking_nodes[r].parking_state -= 1
                    sum_parking_state_after -= 1


## - CREATORS - ##
# PNODES

def create_parking_nodes(world: World, num_parking_nodes: int, time_of_day: int):
    charging_states = [s for s in World.CHARGING_STATE]
    charging_state_probs = [World.CHARGING_STATE[s] for s in World.CHARGING_STATE]

    distributions_df = pd.read_csv('./data/pickup_delivery_distributions_every_hour')
    distributions_current_time = distributions_df.loc[distributions_df.Period == time_of_day]
    distributions_next_time_step = distributions_df.loc[distributions_df.Period == time_of_day + 1]
    all_node_ids = [i for i in range(1, 255)]
    # deliveries_prob is the probability of having more than zero deliveries in the former time period of each chosen
    # parking node. It is used to distribute cars for parking nodes after all nodes have been chosen. High pickup
    # probability means high probability of having a parking state > 0.
    deliveries_prob = []
    chosen_ids = []
    parking_states = []
    charging_states = []
    ideal_states = []

    for i in range(num_parking_nodes):
        node_id = np.choice(all_node_ids)
        all_node_ids.remove(node_id)
        charging_state = int(np.random.choice(charging_states, size=1, p=charging_state_probs))

        pickup_distribution = distributions_current_time.loc[
            distributions_current_time.Zone == node_id, 'Pickup distribution']
        pickup_distribution_as_list = float(pickup_distribution[1:-1].split(', '))


def create_parking_nodes(world: World, num_parking_nodes: int, num_cars):
    charging_states = [s for s in World.CHARGING_STATE]
    charging_state_probs = [World.CHARGING_STATE[s] for s in World.CHARGING_STATE]
    parking_states = [s for s in World.PARKING_STATE]
    parking_state_probs = [World.PARKING_STATE[s] for s in World.PARKING_STATE]
    ideal_states = [s for s in World.IDEAL_STATE]
    ideal_state_probs = [World.IDEAL_STATE[s] for s in World.IDEAL_STATE]
    x_dim = parking_dim['x']
    y_dim = parking_dim['y']
    # node_id = 0
    for i in range(y_dim):
        x_coordinate = i
        for j in range(x_dim):
            charging_state = int(np.random.choice(charging_states, size=1, p=charging_state_probs))
            parking_state = int(np.random.choice(parking_states, size=1, p=parking_state_probs))
            ideal_state = int(np.random.choice(ideal_states, size=1, p=ideal_state_probs))
            node = ParkingNode(parking_state=parking_state, charging_state=charging_state, ideal_state=ideal_state)
            # , demand, deliveries)
            world.add_nodes(node)
            world.add_parking_nodes(node)
    world.add_parking_dim(x_dim, y_dim)


# CNODES
def create_charging_nodes(world: World, num_charging_nodes: int, parking_nodes: [int], capacities: [int],
                          max_capacities: [int]):
    # node_id = len(parking_nodes)
    for i in range(num_charging_nodes):
        # node_id += 1
        print("Creating charging node: ", i + 1)
        parking_node_num = parking_nodes[i]
        parking_node = world.parking_nodes[parking_node_num - 1]
        capacity = capacities[i]
        max_capacity = max_capacities[i]
        # charging_node = ChargingNode(node_id=node_id, parking_node=parking_node, capacity=capacity, max_capacity=max_capacity)
        charging_node = ChargingNode(parking_node=parking_node, capacity=capacity, max_capacity=max_capacity)
        world.add_charging_nodes(charging_node)
        world.add_nodes(charging_node)


# Employees
def create_employees(world: World, num_employees: int, start_time_employees: [int], handling_employees: [int]):
    for i in range(num_employees):
        print("Creating employee", i + 1)
        start_time = start_time_employees[i]
        # 1 if employee is underway with car, 0 if not currently doing task
        handling = handling_employees[i]

        # %Vi plasserer en on its way-employee i en hvilken som helst node,
        # det eneste kravet er at det må stå minst en bil der. Dette er sufficient,
        # fordi vi antar at dersom han kommer med en bil, så er den definert som tilgjengelig i systemet fra tid 0.
        if handling == 1:
            parking_states = []
            for node in world.parking_nodes:
                parking_states.append(node.parking_state)
            # Nodes where pstate is positive
            positive_start_nodes = [i + 1 for i, parking_state in enumerate(parking_states) if parking_state > 0]
            start_node_id = random.choice(positive_start_nodes)
            for node in world.parking_nodes:
                if node.node_id == start_node_id:
                    employee = Employee(start_node=node, start_time=start_time, handling=True)
                    world.add_employee(employee)
        else:
            start_node_id = random.randint(1, len(world.parking_nodes))
            for node in world.parking_nodes:
                if node.node_id == start_node_id:
                    employee = Employee(start_node=node, start_time=start_time, handling=False)
                    world.add_employee(employee)


# CARS
def create_cars(world: World):
    initial_theta, initial_handling = world.calculate_initial_add()
    car_count = 0
    for i in range(len(world.parking_nodes)):
        # Add cars not in need of charging
        for j in range(world.parking_nodes[i].parking_state):  # -world.pNodes[i].cState):
            new_car = Car(parking_node=world.parking_nodes[i], start_time=0, needs_charging=False)
            destinations = []
            for x in range(len(world.parking_nodes)):
                if i != x:
                    destinations.append(world.nodes[x])
            new_car.set_destinations(destinations)
            world.add_car(new_car)
            car_count += 1
            print("Car {} Node {} - car not in need of charging".format(car_count, i + 1))
        # Add cars on its way to parking node
        if initial_handling[i] > 0:
            for j in range(len(world.employees)):
                if (world.employees[j].handling == 1) and (world.employees[j].start_node.node_id - 1 == i):
                    new_car = Car(parking_node=world.parking_nodes[i], start_time=world.employees[j].start_time,
                                  needs_charging=False)
                    destinations = []
                    for x in range(len(world.parking_nodes)):
                        if i != x:
                            destinations.append(world.nodes[x])
                    new_car.set_destinations(destinations)
                    world.add_car(new_car)
                    world.parking_nodes[i].parking_state += 1

                    car_count += 1
                    print("Car {} Node {} - car on its way to parking node ".format(car_count, i + 1))
        # Add cars in need of charging
        for j in range(world.parking_nodes[i].charging_state):
            destinations = []
            new_car = Car(parking_node=world.parking_nodes[i], start_time=0, needs_charging=True)
            for x in range(len(world.charging_nodes)):
                destinations.append(world.nodes[len(world.parking_nodes) + x])
            new_car.set_destinations(destinations)
            world.add_car(new_car)
            car_count += 1
            print("Car {} Node {} - car in need of charging".format(car_count, i + 1))


def create_car_moves(world: World):
    num_nodes = len(world.nodes)
    for car in world.cars:
        for car_move in car.car_moves:
            index = (car_move.start_node.node_id - 1) * num_nodes + car_move.end_node.node_id - 1
            travel_time = world.distances_car[index]
            car_move.set_travel_time(travel_time)
            world.add_car_move(car_move)


def get_index_of_percentile(probs: [float], percentile: float):
    cumm_prob = 0
    for i in range(len(probs)):
        if (probs(i) + cumm_prob) > percentile:
            return i
        else:
            cumm_prob += probs(i)


def main():
    a = get_index_of_percentile([0.3, 0.2, 0.5], 0.8)
    print(a)


main()
