from InstanceGenerator.helper_functions import *
import numpy as np
import math
import random
import time
import copy
from DataRetrieval import google_traffic_information_retriever as gI


DISTANCESCALE = 3

class World:
    cf = read_config('world_constants_config.yaml')

    # COST CONSTANTS #
    PROFIT_RENTAL = cf['objective_function']['profit_rental']
    COST_RELOCATION = cf['objective_function']['cost_relocation']
    COST_DEVIATION = cf['objective_function']['cost_deviation']

    # TIME CONSTANTS #
    HANDLING_TIME_PARKING = cf['time_constants']['handling_parking']
    HANDLING_TIME_CHARGING = cf['time_constants']['handling_charging']
    PLANING_PERIOD = cf['time_constants']['planning_period']

    # BOARD SIZE #
    OPERATING_AREA_SIZE = [cf['operating_area_grid']['y'], cf['operating_area_grid']['x']]

    # DEMAND
    CUSTOMER_REQUESTS = cf['node_demands']['customer_requests']
    CAR_RETURNS = cf['node_demands']['car_returns']

    # COORDINATE CONSTANTS
    LOWERLEFT = (cf['board']['coordinates']['lower_left']['lat'], cf['board']['coordinates']['lower_left']['long'])
    UPPERRIGHT = (cf['board']['coordinates']['upper_right']['lat'], cf['board']['coordinates']['upper_right']['long'])

    def __init__(self):

        # PARKING GRID
        self.x_dim_parking = 0
        self.y_dim_parking = 0

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
        self.demands = {}
        self.deliveries = {}
        self.coordinates = []
        self.remove_coordinates = []
        self.bigM = []

    def add_parking_dim(self, x_dim_parking, y_dim_parking):
        self.x_dim_parking = x_dim_parking
        self.y_dim_parking = y_dim_parking

    def add_nodes(self, node):
        self.nodes.append(node)

    def add_parking_nodes(self, parking_node):
        self.parking_nodes.append(parking_node)

    def add_charging_nodes(self, charging_node):
        self.charging_nodes.append(charging_node)

    def add_employee(self, employee):
        self.employees.append(employee)

    def add_car(self, car):
        self.cars.append(car)

    def set_scenarios(self, n):
        self.numScenarios = n

    def set_num_first_stage_tasks(self, n):
        self.firstStageTasks = n

    # customer requests in the second stage and
    # customer deliveries/car returns in the second stage
    def set_demands(self):
        customer_requests = [k for k in World.CUSTOMER_REQUESTS]
        customer_requests_prob = [World.CUSTOMER_REQUESTS[k] for k in World.CUSTOMER_REQUESTS]
        car_returns = [k for k in World.CAR_RETURNS]
        car_returns_prob = [World.CAR_RETURNS[k] for k in World.CAR_RETURNS]
        for i in range(len(self.parking_nodes)):
            # np.random.seed(i)
            self.parking_nodes[i].demand = np.random.choice(customer_requests, size=self.numScenarios, p=customer_requests_prob)
            self.parking_nodes[i].deliveries = np.random.choice(car_returns, size=self.numScenarios, p=car_returns_prob)

    def add_cords(self, coordinates: [(float, float)]):
        # cord (y,x)
        self.coordinates = coordinates

    ## CALCULATE DISTANCE ##

    def calculate_distances(self):

        maxDistance = math.sqrt(math.pow(self.parking_nodes[0].xCord - self.parking_nodes[len(self.parking_nodes) - 1].xCord, 2) + math.pow(
            self.parking_nodes[0].yCord - self.parking_nodes[len(self.parking_nodes) - 1].yCord, 2))
        scale = float(format((self.PLANING_PERIOD - 1) / (maxDistance * DISTANCESCALE), '.1f'))
        for x in range(len(self.nodes)):
            for y in range(len(self.nodes)):
                distance = math.pow(self.nodes[x].xCord - self.nodes[y].xCord, 2) + math.pow(
                    self.nodes[x].yCord - self.nodes[y].yCord, 2)
                distanceSq = float(format(math.sqrt(distance) * scale, '.1f'))
                distanceB = float(format(distanceSq * 2, '.1f'))

                # Creating some distance between charging and parking nodes
                if (int(distance) == 0 and x != y):
                    distanceSq = 1.0
                    distanceB = 1.0

                self.distancesC.append(distanceSq)
                self.distancesB.append(distanceB)

    def giveRealCoordinatesSpread(self):
        stepX = (self.UPPERRIGHT[1] - self.LOWERLEFT[1]) / World.OPERATING_AREA_SIZE[1]
        stepY = (self.UPPERRIGHT[0] - self.LOWERLEFT[0]) / World.OPERATING_AREA_SIZE[0]
        startX = self.LOWERLEFT[1] + 0.5 * stepX
        startY = self.UPPERRIGHT[0] - 0.5 * stepY

        grid_cords = []
        # Generate coordinate for each of the 8x12 (y,x) nodes in operating area
        for i in range(World.OPERATING_AREA_SIZE[0]):
            for j in range(World.OPERATING_AREA_SIZE[1]):
                cordX = startX + stepX * j
                cordY = startY - stepY * i
                cord = (cordY, cordX)
                grid_cords.append(cord)

        self.remove_coordinates = [(59.90816431249999, 10.647886300000001), (59.90816431249999, 10.662206900000001),
                                   (59.915344937499995, 10.662206900000001), (59.958428687499996, 10.8054129),
                                   (59.9512480625, 10.8054129), (59.958428687499996, 10.8197335),
                                   (59.958428687499996, 10.834054100000001), (59.90816431249999, 10.705168700000002),
                                   (59.9512480625, 10.834054100000001), (59.958428687499996, 10.8483747)]

        grid_cords = [x for x in grid_cords if x not in self.remove_coordinates]

        # Draw sample from operating area to create subset of nodes
        sample_cords = []
        max_nodes = self.x_dim_parking * self.y_dim_parking
        for i in range(max_nodes):
            r = random.randint(0, len(grid_cords) - 1)
            cord = grid_cords.pop(r)
            sample_cords.append(cord)

        self.add_coordinates(sample_cords)
        return sample_cords

    def giveRealCoordinatesCluster(self):
        pass

    def calculateRealDistances(self, cords):
        if (len(cords) == 0):
            step_x = (World.UPPERRIGHT[1] - World.LOWERLEFT[1]) / self.x_dim_parking
            step_y = (World.UPPERRIGHT[0] - World.LOWERLEFT[0]) / self.y_dim_parking
            start_x = World.LOWERLEFT[1] + 0.5 * step_x
            start_y = World.UPPERRIGHT[0] - 0.5 * step_y
            cords = []
            for i in range(self.y_dim_parking):
                for j in range(self.x_dim_parking):
                    cordX = start_x + j * step_x
                    cordY = start_y - i * step_y
                    cord = (cordY, cordX)
                    cords.append(cord)

        travelMatrixCar = gI.run(cords, "driving", False)
        time.sleep(2)
        travelMatrixBicycle = gI.run(cords, "bicycling", False)
        time.sleep(2)
        travelMatrixTransit = gI.run(cords, "transit", False)

        for i in range(len(travelMatrixBicycle)):
            for j in range(len(self.charging_nodes)):
                if (self.charging_nodes[j].parking_node - 1 == i):
                    travelMatrixBicycle[i].append(60)
                    travelMatrixTransit[i].append(60)
                    travelMatrixCar[i].append(60)
                else:
                    travelMatrixBicycle[i].append(travelMatrixBicycle[i][self.charging_nodes[j].parking_node - 1])
                    travelMatrixTransit[i].append(travelMatrixTransit[i][self.charging_nodes[j].parking_node - 1])
                    travelMatrixCar[i].append(travelMatrixCar[i][self.charging_nodes[j].parking_node - 1])
        for i in range(len(self.charging_nodes)):
            travelMatrixBicycle.append(copy.deepcopy(travelMatrixBicycle[self.charging_nodes[i].parking_node - 1]))
            travelMatrixTransit.append(copy.deepcopy(travelMatrixTransit[self.charging_nodes[i].parking_node - 1]))
            travelMatrixCar.append(copy.deepcopy(travelMatrixCar[self.charging_nodes[i].parking_node - 1]))

            travelMatrixBicycle[len(travelMatrixBicycle) - 1][self.charging_nodes[i].parking_node - 1] = 60
            travelMatrixTransit[len(travelMatrixTransit) - 1][self.charging_nodes[i].parking_node - 1] = 60
            travelMatrixCar[len(travelMatrixCar) - 1][self.charging_nodes[i].parking_node - 1] = 60

            # Set distance to charging station to 1 for corresponding node
            travelMatrixBicycle[len(travelMatrixBicycle) - 1][len(self.parking_nodes) + i] = 1.0
            travelMatrixTransit[len(travelMatrixTransit) - 1][len(self.parking_nodes) + i] = 1.0
            travelMatrixCar[len(travelMatrixCar) - 1][len(self.parking_nodes) + i] = 1.0

        travelMatrixNotHandling = []
        travelMatrixHandling = []
        for i in range(len(travelMatrixBicycle)):
            for j in range(len(travelMatrixBicycle[i])):
                travelMatrixNotHandling.append(
                    float(format(min(travelMatrixBicycle[i][j], travelMatrixTransit[i][j]) / 60, '.1f')))
                travelMatrixHandling.append(float(format(travelMatrixCar[i][j] / 60, '.1f')))
        self.distancesC = travelMatrixHandling
        self.distancesB = travelMatrixNotHandling

    ## CALCULATE BIGM
    def calculateBigM(self):
        for i in range(len(self.cars)):
            for j in range(len(self.cars[i].destinations)):
                maxDiff = 0
                for l in range(len(self.cars)):
                    for k in range(len(self.cars[l].destinations)):
                        for x in range(len(self.parking_nodes)):
                            distances1 = self.distancesB[(len(self.nodes) * (self.cars[i].destinations[j] - 1)) + x]
                            distances2 = self.distancesB[(len(self.nodes) * (self.cars[l].destinations[k] - 1)) + x]
                            handlingTime2 = self.distancesC[
                                len(self.nodes) * (self.cars[l].parkingNode - 1) + self.cars[l].destinations[k] - 1]
                            diff = distances1 - (distances2 + handlingTime2)
                            if (diff > maxDiff):
                                maxDiff = diff
                bigMdiff = maxDiff
                bigM = float(format(bigMdiff, '.1f'))
                self.bigM.append(bigM)

    ## CALCULATE VISITS ##

    def calculateInitialAdd(self):
        # Initial theta is the initial number of employees at each node (employees on its way to a node included)
        initial_theta = [0 for i in range(len(self.nodes))]
        # Initial handling is a list of employees coming in to each node after handling car
        # if finishing up a task, an employee and a car will arrive a this node
        initial_handling = [0 for i in range(len(self.nodes))]
        for j in range(len(self.employees)):
            initial_theta[self.employees[j].startNode - 1] += 1
            if (self.employees[j].handling):
                initial_handling[self.employees[j].startNode - 1] += 1
        return initial_theta, initial_handling

    ## SCALE IDEAL STATE ##

    #
    def create_real_ideal_state(self):
        initialAdd = [0 for i in range(len(self.nodes))]
        for j in range(len(self.employees)):
            if (self.employees[j].handling):
                initialAdd[self.employees[j].startNode - 1] += 1

        sum_ideal_state = 0
        sum_parking_state = 0
        # net sum of Requests-Deliveries for each scenario
        for i in range(len(self.parking_nodes)):
            sum_ideal_state += self.parking_nodes[i].iState
            sum_parking_state += self.parking_nodes[i].pState + initialAdd[i]

        # ideal state should be scaled to sum of available cars in the worst case

        # Setter max outflow i hver node til 1.1, for å få litt mer spennende instanser
        max_flow = math.ceil(0.6 * len(self.parking_nodes))
        # max_flow = (max(DEMAND)-min(DELIVERIES))*len(self.parking_nodes)

        sum_ideal_state += max_flow
        # sum_ideal_state += int(max(netFlowScenarios))
        sum_parking_state_after = 0

        for i in range(len(self.parking_nodes)):
            # Scale pstate with rounded share of sum_parking_state
            # Share of pstate = iptate/sum_parking_state
            self.parking_nodes[i].pState = int(round(float(sum_ideal_state) * (float(self.parking_nodes[i].pState) / sum_parking_state)))
            sum_parking_state_after += self.parking_nodes[i].pState

        # Correct for errors due to rounding
        while (sum_parking_state_after != sum_ideal_state):
            if (sum_parking_state_after < sum_ideal_state):
                r = random.randint(0, len(self.parking_nodes) - 1)
                self.parking_nodes[r].pState += 1
                sum_parking_state_after += 1
            else:
                r = random.randint(0, len(self.parking_nodes) - 1)
                if (self.parking_nodes[r].pState - initialAdd[r] > 0):
                    self.parking_nodes[r].pState -= 1
                    sum_parking_state_after -= 1


