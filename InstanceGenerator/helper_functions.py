import yaml
from InstanceGenerator.world import World
from InstanceGenerator.instance_components import *
import numpy as np
import random


def read_config(config_name: str):
    with open(config_name, 'r') as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
# read_config('world_constants_config.yaml')


## - CREATORS - ##
# PNODES
def create_parking_nodes(world: World, parking_dim: {str: int}):
    charging_states = [s for s in World.CHARGING_STATE]
    charging_state_probs = [World.CHARGING_STATE[s] for s in World.CHARGING_STATE]
    parking_states = [s for s in World.CHARGING_STATE]
    parking_state_probs = [World.PARKING_STATE[s] for s in World.PARKING_STATE]
    ideal_states = [s for s in World.CHARGING_STATE]
    ideal_state_probs = [World.IDEAL_STATE[s] for s in World.IDEAL_STATE]
    x_dim = parking_dim['x']
    y_dim = parking_dim['y']

    for i in range(y_dim):
        x_coordinate = i
        for j in range(x_dim):
            # np.random.seed(i+j)
            # cState = int(np.random.choice([0,1], size=1, p=[0.75, 0.25]))
            charging_state = int(np.random.choice(charging_states, size=1, p=charging_state_probs))
            # np.random.seed(i + j)
            parking_state = int(np.random.choice(parking_states, size=1, p=parking_state_probs))
            # np.random.seed(i + j)
            ideal_state = int(np.random.choice(ideal_states, size=1, p=ideal_state_probs))

            # np.random.seed(i + j)
            # demand = np.random.choice(DEMAND, size=NUMSCENARIOS, p=DEMANDPROB) #customer requests in the second stage
            # deliveries = np.random.choice(DELIVERIES, NUMSCENARIOS, p=DELIVERIESPROB) #customer deliveries in the second stage
            y_coordinate = j
            node = ParkingNode(x_coordinate, y_coordinate, parking_state, charging_state,
                               ideal_state)  # , demand, deliveries)
            world.add_nodes(node)
            world.add_parking_nodes(node)
    world.add_parking_dim(x_dim, y_dim)


# CNODES
def create_charging_nodes(world: World, num_charging_nodes: int, parking_nodes: [int], capacities: [int],
                          max_capacities: [int]):
    for i in range(num_charging_nodes):
        print("Creating charging node: ", i + 1)
        parking_node_num = parking_nodes[i]
        parking_node = world.parking_nodes[parking_node_num - 1]
        capacity = capacities[i]
        max_capacity = max_capacities[i]
        charging_node = ChargingNode(parking_node.x_coordinate, parking_node.y_coordinate, capacity, max_capacity,
                                     parking_node_num)
        world.add_charging_nodes(charging_node)
        world.add_nodes(charging_node)


# Employees
def create_employees(world: World, num_employees: int, start_time_employees: [float], handling_employees: [int]):
    for i in range(num_employees):
        print("Creating employee", i + 1)
        time = start_time_employees[i]
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
            start_node = random.choice(positive_start_nodes)
            employee = Employee(start_node, time, True)
            world.add_employee(employee)
        else:
            start_node = random.randint(1, len(world.parking_nodes))
            employee = Employee(start_node, time, True)
            world.add_employee(employee)


# CARS
def create_cars(world):
    initial_theta, initial_handling = world.calculateInitialAdd()
    car_count = 0
    for i in range(len(world.parking_nodes)):
        # Add cars not in need of charging
        for j in range(world.parking_nodes[i].parking_state):  # -world.pNodes[i].cState):
            new_car = Car(0, i + 1, False)
            destinations = []
            for x in range(len(world.parking_nodes)):
                if i != x:
                    destinations.append(x + 1)
            new_car.set_destinations(destinations)
            world.add_car(new_car)
            car_count += 1
            print("Car {} Node {} - car not in need of charging".format(car_count, i + 1))
        # Add cars on its way to parking node
        if initial_handling[i] > 0:
            for j in range(len(world.employees)):
                if (world.employees[j].handling == 1) and (world.employees[j].start_node - 1 == i):
                    new_car = Car(world.employees[j].start_time, i + 1, False)
                    for x in range(len(world.parking_nodes)):
                        if i != x:
                            new_car.destinations.append(x + 1)
                    world.addCar(new_car)
                    world.parking_nodes[i].parking_state += 1

                    # if (world.pNodes[i].pState == 0):
                    #     pstates = []
                    #     for node in world.pNodes:
                    #         pstates.append(node.pState)
                    #     positiveStartNodes = [i + 1 for i, pstate in enumerate(pstates) if pstate > 0]
                    #     startNode = random.choice(positiveStartNodes)
                    #     world.pNodes[i].pState += 1
                    #     startNode.pState -= 1
                    #     world.addCar(newCar)

                    car_count += 1
                    print("Car {} Node {} - car on its way to parking node ".format(car_count, i + 1))
        # Add cars in need of charging
        for j in range(world.parking_nodes[i].charging_state):
            new_car = Car(0, i + 1, True)
            for x in range(len(world.cNodes)):
                new_car.destinations.append(len(world.parking_nodes) + x + 1)
            world.addCar(new_car)
            car_count += 1
            print("Car {} Node {} - car in need of charging".format(car_count, i + 1))
