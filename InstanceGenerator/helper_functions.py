import yaml
import os

def read_config(config_name: str):
    with open(config_name, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
# read_config('world_constants_config.yaml')

 ## FILE HANDLER ##

def write_to_file(world, instance_name: str):
    test_dir = "../Gurobi/tests/{}nodes/".format(len(world.parking_nodes))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    file_name = test_dir + str(instance_name) + "_a.txt" #+ "_a.txt"
    if (os.path.exists(file_name)):
        file_name = test_dir + str(instance_name) + "_b.txt"
        if (os.path.exists(file_name)):
            file_name = test_dir + str(instance_name) + "_c.txt"
    print(file_name)
    f = open(file_name, 'w')
    string = "[INITIALIZE]\n"

    # Number of scenarios
    string += "num_scenarios: " + str(world.num_scenarios) + "\n"
    # Number of parking nodes
    string += "num_parking_nodes: " + str(len(world.parking_nodes)) + "\n"
    # Number of charging nodes
    string += "num_charging_nodes: " + str(len(world.charging_nodes)) + "\n"
    # Number of employees
    string += "num_employees: " + str(len(world.employees)) + "\n"
    string += "\n"
    # Grid
    #string += "hNodes : " + str(world.YCORD) + "\n"
    #string += "wNodes : " + str(world.XCORD) + "\n"

    # start node of employee
    string += "start_node_employee: [ "
    for i in range(len(world.employees)):
        string += str(world.employees[i].start_node)
        if (i < len(world.employees) - 1):
            string += " "
    string += " ] \n"
    string += "\n"
    # Available charging slots
    string += "charging_slots_available: [ "
    for i in range(len(world.charging_nodes)):
        string += str(world.charging_nodes[i].capacity)
        if (i < len(world.charging_nodes) - 1):
            string += " "
    string += " ] \n"
    # Total number of charging slots (total capacity of charging node)
    string += "total_number_of_charging_slots: [ "
    for i in range(len(world.charging_nodes)):
        string += str(world.charging_nodes[i].max_capacity)
        if (i < len(world.charging_nodes) - 1):
            string += " "
    string += " ] \n"
    string += "\n"
    # reward for renting out car in second stage
    string += "profit_rental : " + str(world.PROFIT_RENTAL) + "\n"
    # cost of relocating vehicle
    string += "cost_relocation : " + str(world.COST_RELOCATION) + "\n"
    # cost of deviating from ideal state
    string += "cost_deviation : " + str(world.COST_DEVIATION) + "\n"
    string += "\n"
    # car travel times between nodes
    string += "travel_time_vehicle: [ "
    for i in range(len(world.nodes)):
        for j in range(len(world.nodes)):
            string += str(world.distances_car[i*len(world.nodes) + j]) + " "
        if(i < len(world.nodes) -1):
            string += "\n"
            string+= "\t" + "\t"
    string+="]" + "\n"
    string += "\n"
    # bike travel times between nodes
    string += "travel_time_bike: [ "
    for i in range(len(world.nodes)):
        for j in range(len(world.nodes)):
            string += str(world.distances_public_bike[i * len(world.nodes) + j]) + " "
        if (i < len(world.nodes) - 1):
            string += "\n"
            string += "\t" + "\t"
    string += "]" + "\n"
    string += "\n"
    # handling time for p- and c-nodes. Time used to initiate charging and find vacant parking spot.
    # used in CarMoveHandlingTime
    string += "handling_time_parking: " + str(world.HANDLING_TIME_PARKING) + "\n"
    string += "handling_time_charging: " + str(world.HANDLING_TIME_CHARGING) + "\n"

    # Travel
    string += "travel_time_to_origin: [ "
    for i in range(len(world.employees)):
        string += str(world.employees[i].start_time)
        if (i < len(world.employees) - 1):
            string += " "
        else:
            string += " ] \n"

    string += "start_time_car: [ "
    for i in range(len(world.cars)):
        string += str(world.cars[i].start_time)
        if (i < len(world.cars) - 1):
            string += " "
        else:
            string += " ] \n"

    string += "planning_period: " + str(world.PLANNING_PERIOD) + "\n"
    string += "\n"

    # Cars in
    string += "parking_state: [ "
    for i in range(len(world.parking_nodes)):
        string += str(world.parking_nodes[i].parking_state)
        if (i < len(world.parking_nodes) - 1):
            string += " "
        else:
            string += " ] \n"
    # Cars in need of charging in parking node
    string += "charging_state: [ "
    for i in range(len(world.parking_nodes)):
        string += str(world.parking_nodes[i].charging_state)
        if (i < len(world.parking_nodes) - 1):
            string += " "
        else:
            string += " ] \n"

    string += "ideal_state: [ "
    for i in range(len(world.parking_nodes)):
        string += str(world.parking_nodes[i].ideal_state)
        if (i < len(world.parking_nodes) - 1):
            string += " "
        else:
            string += " ] \n"
    string += "\n"
    string += "customer_requests: [ "
    for i in range(len(world.parking_nodes)):
        #print(world.parking_nodes[i].customer_requests)
        for j in range(world.num_scenarios):
            string += str(world.parking_nodes[i].customer_requests[j]) + " "
        if (i < len(world.parking_nodes) - 1):
            string += "\n"
            string += "\t" + "\t"
        else:
            string += "] \n"
    string += "\n"
    string += "car_returns: [ "
    for i in range(len(world.parking_nodes)):
        for j in range(world.num_scenarios):
            string += str(world.parking_nodes[i].car_returns[j]) + " "
        if (i < len(world.parking_nodes) - 1):
            string += "\n"
            string += "\t" + "\t"
        else:
            string += "] \n"
    string += "\n"

    count = 0
    for i in range(len(world.cars)):
        if not (world.cars[i].is_charging):
            for j in range(len(world.cars[i].destinations)):
                count += 1
    string += "num_car_moves_parking : " + str(count) + "\n"
    count = 0
    for i in range(len(world.cars)):
        if (world.cars[i].is_charging):
            for j in range(len(world.cars[i].destinations)):
                count += 1
    string += "num_car_moves_charging : " + str(count) + "\n"
    string += "num_cars : " + str(len(world.cars)) + "\n"
    print("numcars: ", len(world.cars))
    string += "num_tasks : " + str(world.tasks) + "\n"
    string += "num_first_stage_tasks : " + str(world.first_stage_tasks) + "\n"
    string += "\n"

    string += "car_move_cars : [ "
    for i in range(len(world.cars)):
        for j in range(len(world.cars[i].destinations)):
            string += str(i + 1)
            if (i < len(world.cars) - 1):
                string += " "
            else:
                if (j < len(world.cars[i].destinations) - 1):
                    string += " "
    string += " ] \n"

    string += "car_move_start_time : [ "
    for i in range(len(world.cars)):
        for j in range(len(world.cars[i].destinations)):
            string += str(world.cars[i].start_time)
            if (i < len(world.cars) - 1):
                string += " "
            else:
                if (j < len(world.cars[i].destinations) - 1):
                    string += " "
    string += " ] \n"

    string += "car_move_origin : [ "
    for i in range(len(world.cars)):
        for j in range(len(world.cars[i].destinations)):
            string += str(world.cars[i].parking_node)
            if (i < len(world.cars) - 1):
                string += " "
            else:
                if (j < len(world.cars[i].destinations) - 1):
                    string += " "
    string += " ] \n"
    string += "car_move_destination : [ "
    for i in range(len(world.cars)):
        for j in range(len(world.cars[i].destinations)):
            string += str(world.cars[i].destinations[j])
            if (i < len(world.cars) - 1):
                string += " "
            else:
                if (j < len(world.cars[i].destinations) - 1):
                    string += " "
    string += " ] \n"
    string += "car_move_handling_time : [ "
    for i in range(len(world.cars)):
        for j in range(len(world.cars[i].destinations)):
            if (world.cars[i].destinations[j] > len(world.parking_nodes)):
                string += str(world.distances_car[
                                  (world.cars[i].parking_node - 1) * len(world.nodes) + world.cars[i].destinations[
                                      j] - 1] + world.HANDLING_TIME_CHARGING)
            else:
                string += str(world.distances_car[
                                  (world.cars[i].parking_node - 1) * len(world.nodes) + world.cars[i].destinations[
                                      j] - 1] + world.HANDLING_TIME_PARKING)
            if (i < len(world.cars) - 1):
                string += " "
            else:
                if (j < len(world.cars[i].destinations) - 1):
                    string += " "
    string += " ] \n"
    count = 0
    for i in range(len(world.parking_nodes)):
        if (world.parking_nodes[i].charging_state > 0):
            count += 1
    string += "num_cars_in_need_of_charging : " + str(count)
    string += "\n"
    car_in_need_of_charging_nodes = []
    for i in range(len(world.parking_nodes)):
        if (world.parking_nodes[i].charging_state > 0):
            car_in_need_of_charging_nodes.append(i)
    string += "nodes_with_cars_in_need_of_charging : [ "
    for i in range(len(car_in_need_of_charging_nodes)):
        string += str(car_in_need_of_charging_nodes[i] + 1)
        if (i < len(car_in_need_of_charging_nodes) - 1):
            string += " "
    string += " ]\n"

    string += "cars_in_need_of_charging_at_nodes : [ "
    for i in range(len(world.parking_nodes)):
        print("(node: {0}, pstate: {1}, cstate: {2}, istate: {3}, requests: {4}, deliveries: {5})".format(
            i+1, world.parking_nodes[i].parking_state, world.parking_nodes[i].charging_state, world.parking_nodes[i].ideal_state,
            world.parking_nodes[i].customer_requests, world.parking_nodes[i].car_returns))
    for i in range(len(world.cars)):
        print("(car: {0}, parking_node: {1}, destinations: {2})".format(i+1, world.cars[i].parking_node, world.cars[i].destinations))

    for i in range(len(car_in_need_of_charging_nodes)):
        string += str(world.parking_nodes[car_in_need_of_charging_nodes[i]].charging_state)
        string += " "
    string += "]\n"

    string += "bigM : [ "
    for i in range(len(world.bigM)):
        string += str(world.bigM[i])
        string += " "
    string += "]\n"
    string += "\n"

    string += "cars_available_in_node : [ "
    for i in range(len(world.parking_nodes)):
        string += str(world.parking_nodes[i].parking_state)
        string += " "
    string += "]\n"
    string += "\n"

    f.write(string)