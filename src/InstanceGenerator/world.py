from matplotlib import pyplot as plt

from HelperFiles.helper_functions import read_config
from Heuristics.helper_functions_heuristics import frequencies, probabilities
from InstanceGenerator.instance_components import Node, ParkingNode, ChargingNode, Employee, Car, CarMove
import numpy as np
import pandas as pd
import random
import os
import seaborn as sns
from path_manager import path_to_src

os.chdir(path_to_src)


class World:
    # TODO: make class constants

    cf = read_config('InstanceGenerator/world_constants_config.yaml')
    # COST CONSTANTS #
    PROFIT_RENTAL = cf['objective_function']['profit_rental']
    COST_RELOCATION = cf['objective_function']['cost_relocation']
    COST_DEVIATION = cf['objective_function']['cost_deviation']

    # TIME CONSTANTS #
    HANDLING_TIME_PARKING = cf['time_constants']['handling_parking']
    HANDLING_TIME_CHARGING = cf['time_constants']['handling_charging']
    # PLANNING_PERIOD = cf['time_constants']['planning_period']

    # NODE STATES
    #CHARGING_STATE = cf['charging_probs']

    '''
    # COST CONSTANTS #
    cf = read_config('InstanceGenerator/world_constants_config.yaml')
    profit_rental = cf['objective_function']['profit_rental']
    cost_relocation = cf['objective_function']['cost_relocation']
    cost_deviation = cf['objective_function']['cost_deviation']

    # TIME CONSTANTS #
    handling_time_parking = cf['time_constants']['handling_parking']
    handling_time_charging = cf['time_constants']['handling_charging']
    # NODE STATES
    charging_state = cf['charging_probs']
    '''

    def __init__(self):

        self.planning_period = None

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
        self.relevant_car_moves = []
        self.bigM = []

    def set_num_scenarios(self, n: int):
        self.num_scenarios = n
        for charging_node in self.charging_nodes:
            charging_node.initialize_charging_node_state(num_scenarios=n)

    def set_num_tasks(self, n: int):
        self.tasks = n

    def set_num_first_stage_tasks(self, n: int):
        self.first_stage_tasks = n

    def set_planning_period(self, planning_period: int):
        self.planning_period = planning_period

    def add_node(self, node):
        self.nodes.append(node)

    def add_parking_node(self, parking_node: ParkingNode):
        self.parking_nodes.append(parking_node)

    def add_charging_node(self, charging_node: ChargingNode):
        self.charging_nodes.append(charging_node)

    def add_employee(self, employee: Employee):
        self.employees.append(employee)

    def add_car(self, car: Car):
        self.cars.append(car)

    def add_car_move(self, car_move: CarMove):
        self.car_moves.append(car_move)

    def _irrelevant_end_nodes(self, parking_nodes, acceptance_percentage=1):
        out_list = []
        for node in parking_nodes:
            net_flow = node.ideal_state + node.customer_requests - node.car_returns - node.parking_state
            count = sum(1 for s in net_flow if s <= 0)
            if count / len(net_flow) >= acceptance_percentage:
                out_list.append(node)
        return out_list

    def _irrelevant_start_nodes(self, parking_nodes, acceptance_percentage=1):
        out_list = []
        for node in parking_nodes:
            net_flow = node.ideal_state + node.customer_requests - node.car_returns - node.parking_state
            count = sum(1 for s in net_flow if s > 0)
            if count / len(net_flow) >= acceptance_percentage:
                out_list.append(node)
        return out_list

    def initialize_relevant_car_moves(self, acceptance_percentage):
        relevant_car_moves = []
        if acceptance_percentage == 2:
            relevant_car_moves = [cm for cm in self.car_moves]
        else:
            irrelevant_end_nodes = self._irrelevant_end_nodes(self.parking_nodes, acceptance_percentage)
            irrelevant_start_nodes = self._irrelevant_start_nodes(self.parking_nodes, acceptance_percentage)
            for cm in self.car_moves:
                if (cm.end_node not in irrelevant_end_nodes and cm.start_node not in irrelevant_start_nodes) or cm.is_charging_move:
                    relevant_car_moves.append(cm)
        self.relevant_car_moves = relevant_car_moves

    def initialize_relevant_car_moves_distance(self, distance_inclusion_fraction: float):
        '''
        :param distance_inclusion_fraction:
                float (0,1). 1 if include all moves and 0 to remove all moves. if e.g. distance_fraction is 0.4
                then only moves that are shorter than or equal to 0.4*max_travel_distance, where max_travel_distance is the
                distance of the longest car-move
        :return:
        '''

        if distance_inclusion_fraction == 1:
            return
        else:
            max_handling_time = max(cm.handling_time for cm in self.car_moves)
            relevant_car_moves = []
            for cm in self.relevant_car_moves:
                cm_handling_time_frac = cm.handling_time/max_handling_time
                if cm_handling_time_frac <= distance_inclusion_fraction or cm.is_charging_move:
                    relevant_car_moves.append(cm)

            self.relevant_car_moves = relevant_car_moves


    def add_car_move_to_employee(self, car_move: CarMove, employee: Employee, scenario: int = None):

        if scenario is None:
            travel_time_to_car_move = self.get_employee_travel_time_to_node(start_node=employee.current_node,
                                                                            end_node=car_move.start_node)
            employee.add_car_move(travel_time_to_car_move=travel_time_to_car_move, car_move=car_move)
            if len(employee.car_moves) == self.first_stage_tasks:
                employee.initialize_second_stage(num_scenarios=self.num_scenarios)
        else:
            # zero-indexed scenario
            travel_time_to_car_move = self.get_employee_travel_time_to_node(
                start_node=employee.current_node_second_stage[scenario], end_node=car_move.start_node)

            employee.add_car_move(travel_time_to_car_move=travel_time_to_car_move, car_move=car_move, scenario=scenario)

    '''
    def remove_car_move_from_employee(self, car_move: CarMove, employee: Employee):
        # start_node is the end node of the car move performed before the one we want to remove.
        # what if the list is empty? then take start_node
        try:
            idx = employee.car_moves.index(car_move)
            travel_time_to_car_move = self.get_employee_travel_time_to_node(start_node=employee.car_moves[idx-1].end_node,
                                                                      end_node=car_move.start_node)
        except:
            travel_time_to_car_move = self.get_employee_travel_time_to_node(start_node=employee.start_node,
                                                                      end_node=car_move.start_node)
        idx = employee.car_moves.index(car_move)
        if idx == 0 and len(employee.car_moves) > 1:
            new_travel_time_to_car_move = self.get_employee_travel_time_to_node(
                start_node=employee.start_node,
                end_node=employee.car_moves[1].start_node)

        elif idx > 0 and len(employee.car_moves) > idx:
            new_travel_time_to_car_move = self.get_employee_travel_time_to_node(
                start_node=employee.car_moves[idx-1].start_node,
                end_node=employee.car_moves[idx+1].start_node
            )

        else:
            new_travel_time_to_car_move = 0

        employee.remove_car_move(idx=idx, new_travel_time_to_car_move=new_travel_time_to_car_move)
        '''

    def get_employee_travel_time_to_node(self, start_node: Node, end_node: Node):
        employee_start_node = start_node.node_id - 1
        employee_end_node = end_node.node_id - 1
        return self.distances_public_bike[employee_start_node][employee_end_node]

    ## CALCULATE BIGM
    def calculate_bigM(self):
        for i in range(len(self.cars)):
            for j in range(len(self.cars[i].destinations)):
                max_diff = 0
                for l in range(len(self.cars)):
                    for k in range(len(self.cars[l].destinations)):
                        for x in range(len(self.parking_nodes)):
                            distances1 = self.distances_public_bike[self.cars[i].destinations[j].node_id - 1][x]
                            # distances1 = self.distances_public_bike[
                            #    (len(self.nodes) * (self.cars[i].destinations[j].node_id - 1)) + x]
                            distances2 = self.distances_public_bike[self.cars[l].destinations[k].node_id - 1][x]
                            # distances2 = self.distances_public_bike[
                            #    (len(self.nodes) * (self.cars[l].destinations[k].node_id - 1)) + x]
                            handling_time = self.distances_car[self.cars[l].parking_node.node_id - 1][
                                self.cars[l].destinations[k].node_id - 1]
                            # handling_time = self.distances_car[
                            #    len(self.nodes) * (self.cars[l].parking_node.node_id - 1) + self.cars[l].destinations[
                            #        k].node_id - 1]
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


## CALCULATE DISTANCE ##

def set_distances(world: World, parking_node_nums: [int]):
    distances_car = pd.read_csv('../data/travel_times_car_all_zones.csv', index_col=0)
    distances_transit_bike = pd.read_csv('../data/travel_times_non_car_all_zones.csv', index_col=0)
    parking_node_nrs = np.array([node.get_nr() for node in world.parking_nodes])
    indices_parking_nodes = parking_node_nrs - 1
    distance_matrix_parking_nodes_car = distances_car.iloc[indices_parking_nodes, indices_parking_nodes]
    distance_matrix_parking_nodes_transit_bike = distances_transit_bike.iloc[
        indices_parking_nodes, indices_parking_nodes]

    charging_nodes_pnodes = [node.parking_node.get_nr() for node in world.charging_nodes]

    counter = 300  # random number > 254
    index_counter = 0
    for node in charging_nodes_pnodes:
        distance_matrix_parking_nodes_car[counter] = distance_matrix_parking_nodes_car[str(node)]
        distance_matrix_parking_nodes_transit_bike[counter] = distance_matrix_parking_nodes_transit_bike[str(node)]
        distance_matrix_parking_nodes_car = distance_matrix_parking_nodes_car.append(
            distance_matrix_parking_nodes_car.iloc[parking_node_nums[index_counter] - 1])
        distance_matrix_parking_nodes_transit_bike = distance_matrix_parking_nodes_transit_bike.append(
            distance_matrix_parking_nodes_transit_bike.iloc[parking_node_nums[index_counter] - 1])

        index_counter += 1
        counter += 1

    distance_matrix_parking_nodes_car = np.array(distance_matrix_parking_nodes_car)
    distance_matrix_parking_nodes_transit_bike = np.array(distance_matrix_parking_nodes_transit_bike)

    for x in range(len(distance_matrix_parking_nodes_car)):
        for y in range(len(distance_matrix_parking_nodes_car)):
            # Creating some distance between charging and parking nodes
            if distance_matrix_parking_nodes_car[x][y] == 0 and x != y:
                distance_matrix_parking_nodes_car[x][y] = 60
                distance_matrix_parking_nodes_transit_bike[x][y] = 60

    distance_matrix_parking_nodes_car = np.round(distance_matrix_parking_nodes_car / 60, 1)
    distance_matrix_parking_nodes_transit_bike = np.round(distance_matrix_parking_nodes_transit_bike / 60, 1)

    world.distances_car = distance_matrix_parking_nodes_car.tolist()
    world.distances_public_bike = distance_matrix_parking_nodes_transit_bike.tolist()


# customer requests in the second stage and
# customer deliveries/car returns in the second stage
def set_demands(world: World, time_of_day: int):
    distributions_df = pd.read_csv('../data/pickup_delivery_distributions_every_hour.csv', index_col=0)
    distributions_current_time = distributions_df.loc[distributions_df.Period == time_of_day]

    for i in range(len(world.parking_nodes)):
        node_id = world.parking_nodes[i].get_id()

        # Requests
        pickup_distribution = distributions_current_time.loc[
            distributions_current_time.Zone == node_id, 'Pickup distribution']
        pickup_distribution_as_list = [np.round(float(i), 2) for i in
                                       pickup_distribution.item()[1:-1].split(', ')]
        pickup_distribution_as_list_scaled = scale_up_distribution(pickup_distribution_as_list, 0.2)
        customer_requests = [i for i in range(len(pickup_distribution_as_list_scaled))]

        world.parking_nodes[i].set_customer_requests(
            np.random.choice(customer_requests, size=world.num_scenarios, p=pickup_distribution_as_list_scaled))

        # Returns
        delivery_distribution = distributions_current_time.loc[
            distributions_current_time.Zone == node_id, 'Delivery distribution']
        delivery_distribution_as_list = [np.round(float(i), 2) for i in
                                         delivery_distribution.item()[1:-1].split(', ')]
        delivery_distribution_as_list_scaled = scale_up_distribution(delivery_distribution_as_list, 0.2)

        customer_returns = [i for i in range(len(delivery_distribution_as_list_scaled))]

        world.parking_nodes[i].set_car_returns(

            np.random.choice(customer_returns, size=world.num_scenarios, p=delivery_distribution_as_list_scaled))


def create_parking_nodes(world: World, num_parking_nodes: int, time_of_day: int, num_cars: int):
    #possible_charging_states = [s for s in world.CHARGING_STATE]
    #charging_state_probs = [world.CHARGING_STATE[s] for s in world.CHARGING_STATE]

    distributions_df = pd.read_csv('../data/pickup_delivery_distributions_every_hour.csv', index_col=0)
    distributions_next_time_step = distributions_df.loc[distributions_df.Period == time_of_day + 1]
    distributions_former_time_step = distributions_df.loc[distributions_df.Period == time_of_day - 1]
    all_node_nrs = [i for i in range(1, 255)]

    # deliveries_prob is the probability of having more than zero deliveries in the former time period of each chosen
    # parking node. It is used to distribute cars for parking nodes after all nodes have been chosen. High delivery
    # probability means high probability of having a parking state > 0.
    deliveries_prob = []
    chosen_nrs = []
    parking_states = {}

    ideal_states = []
    expected_num_deliveries_former = []

    for i in range(num_parking_nodes):
        node_nr = np.random.choice(all_node_nrs)
        all_node_nrs.remove(node_nr)
        chosen_nrs.append(node_nr)

        # CHARGING STATE
        #charging_states.append(int(np.random.choice(possible_charging_states, p=charging_state_probs)))

        # IDEAL STATE
        # Ideal state should be the index of the element which crosses a given percentile, such as 0.9. This means that
        # 90 percent of the time, this number of cars will be able to serve all demand in next time period, not
        # considering customer deliveries.

        pickup_distribution_next_time_step = distributions_next_time_step.loc[
            distributions_next_time_step.Zone == node_nr, 'Pickup distribution']

        pickup_distribution_next_as_list = [float(i) for i in
                                            pickup_distribution_next_time_step.item()[1:-1].split(', ')]

        ideal_states.append(get_index_of_percentile(pickup_distribution_next_as_list, 0.9))

        # PROBABILITIES OF FORMER PARKING STATE TO BE USED TO SET PARKING STATE
        delivery_distribution_former_time_step = distributions_former_time_step.loc[
            distributions_former_time_step.Zone == node_nr, 'Delivery distribution']

        delivery_distribution_former_as_list = \
            [float(i) for i in delivery_distribution_former_time_step.item()[1:-1].split(', ')]

        deliveries_prob.append(1 - delivery_distribution_former_as_list[0])
        possible_nrs_of_deliveries = [i for i in range(len(delivery_distribution_former_as_list))]
        expected_deliveries_former = [possible_nrs_of_deliveries[i] * delivery_distribution_former_as_list[i]
                                      for i in range(len(delivery_distribution_former_as_list))]

        expected_num_deliveries_former.append(sum(expected_deliveries_former) / len(expected_deliveries_former))

    normalized_delivery_probs = normalize_list(deliveries_prob)

    for i in range(num_parking_nodes):
        parking_states[chosen_nrs[i]] = 0

    for i in range(num_cars):
        node = np.random.choice(chosen_nrs, p=normalized_delivery_probs)
        parking_states[node] += 1

    # SET CHARGING STATE

    charging_states = [0 for i in range(len(chosen_nrs))]
    # ratio of cars in need of charging to number of charged cars, a random number between 0.1 and 0.2
    ratio_charging = random.uniform(0.125, 0.1875)

    # Fill the parking nodes with cars in need of charging until the desired number of cars in need of charging have
    # been placed. Cars with a high expected number of deliveries in the former time period are more likely to have
    # cars in need of charging.
    for i in range(round(num_cars * ratio_charging)):
        dist = normalize_list(expected_num_deliveries_former)
        node = np.random.choice([i for i in range(len(chosen_nrs))], p=dist)
        charging_states[node] += 1

    for i in range(len(chosen_nrs)):
        node = ParkingNode(node_nr=chosen_nrs[i], parking_state=parking_states[chosen_nrs[i]],
                           charging_state=charging_states[i], ideal_state=ideal_states[i])
        world.add_node(node)
        world.add_parking_node(node)


# CNODES
def create_charging_nodes(world: World, num_charging_nodes: int, parking_nodes: [int], capacities: [int]):
    for i in range(num_charging_nodes):
        print("Creating charging node: ", i + 1)
        parking_node_num = parking_nodes[i]
        parking_node = world.parking_nodes[parking_node_num - 1]
        capacity = capacities[i]
        charging_node = ChargingNode(parking_node=parking_node, capacity=capacity)
        # charging_node.initialize_charging_node_state(world.num_scenarios)
        world.add_charging_node(charging_node)
        world.add_node(charging_node)


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
            start_node_index = car_move.start_node.node_id - 1
            end_node_index = car_move.end_node.node_id - 1
            travel_time = world.distances_car[start_node_index][end_node_index]
            car_move.set_travel_time(travel_time)
            world.add_car_move(car_move)


# HELPER FUNCTIONS

def get_index_of_percentile(probs: [float], percentile: float):
    cummulative_prob = 0
    for i in range(len(probs)):
        if (probs[i] + cummulative_prob) >= percentile:
            return i
        else:
            cummulative_prob += probs[i]


def normalize_list(probs: [int]):
    probs_np = np.array(probs)
    normalizer = np.array([np.sum(probs_np) for _ in range(len(probs_np))])
    return probs_np / normalizer


# Moving a certain percentage of the probabilities upwards in order to simulate unmet demand that is not
# seen through distributions
def scale_up_distribution(dist: [float], percentage: float):
    dist_new = dist.copy()
    for i in range(len(dist_new)):
        frac = dist_new[i] * percentage
        dist_new[i] -= frac
        if i == len(dist_new) - 1:
            dist_new.append(frac)
        else:
            dist_new[i + 1] += frac

    dist_new = np.round(np.array(dist_new), 2)
    # Ensuring that the array sums to one
    dist_new[0] += 1.0 - np.sum(dist_new)

    return dist_new
