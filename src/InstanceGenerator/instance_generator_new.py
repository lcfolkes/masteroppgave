from src.InstanceGenerator.file_writer import write_to_file_yaml
from src.HelperFiles.helper_functions import read_config
from src.InstanceGenerator.world import World, create_parking_nodes, \
    create_charging_nodes, create_employees, create_cars, create_car_moves
import copy


def build_world(instance_config: str) -> World:
    SPREAD = True
    world = World()
    initializeWorld(world, instance_config)
    coordinates = []
    if SPREAD:
        coordinates = world.give_real_coordinates_spread()
    # writeToFile(cords)
    world.create_real_ideal_state()
    # if (len(world.pNodes) > 0):
    # world.calculateDistances()
    # else:
    world.calculate_real_distances(coordinates)
    print("World Initialized")
    create_cars(world=world)
    create_car_moves(world=world)
    print("World Created")
    world.calculate_bigM()
    return world


def initialize_world(world: World, instance_config: str):
    cf = instance_config
    create_parking_nodes(world=world, parking_dim=cf['board']['parking_node_dim'])
    create_charging_nodes(world=world, num_charging_nodes=cf['charging_nodes']['num_charging'],
                          parking_nodes=cf['charging_nodes']['parking_nodes'],
                          capacities=cf['charging_nodes']['capacities'],
                          max_capacities=cf['charging_nodes']['max_capacities'])
    create_employees(world=world, num_employees=cf['employees']['num_employees'],
                     start_time_employees=cf['employees']['start_time'],
                     handling_employees=cf['employees']['handling'])


def create_instance_from_world(world: World, num_scenarios: int, num_tasks: int, num_first_stage_tasks: int,
                               version: int) -> World:
    new_world = copy.deepcopy(world)
    new_world.set_scenarios(n=num_scenarios)
    new_world.set_tasks(n=num_tasks)
    new_world.set_num_first_stage_tasks(n=num_first_stage_tasks)
    new_world.set_demands()
    total_moves = 0
    for j in range(len(new_world.cars)):
        total_moves += len(new_world.cars[j].destinations)
    available_cars = 0
    cars_to_charging = 0
    for n in new_world.parking_nodes:
        available_cars += n.parking_state
        cars_to_charging += n.charging_state
    instance_name = str(len(new_world.parking_nodes)) + "-" + str(new_world.num_scenarios) + "-" + str(
        new_world.first_stage_tasks) + "-" + str(version)
    write_to_file_yaml(new_world, instance_name)
    print("Finished")
    return new_world


def main():
    cf = read_config('./InstanceGenerator/InstanceConfigs/instance_config.yaml')
    print("\nWELCOME TO THE EXAMPLE CREATOR \n")
    worlds = []
    for i in range(cf['examples']):
        print("Creating instance: ", i)
        world = buildWorld(instance_config=cf)
        create_instance_from_world(world, num_scenarios=cf['num_scenarios'], num_tasks=cf['tasks']['num_all'],
                                   num_first_stage_tasks=cf['tasks']['num_first_stage'], version=i + 1)
        # create_instance_from_world(world, num_scenarios=1, num_tasks=cf['tasks']['num_all'],
        #                       num_first_stage_tasks=cf['tasks']['num_first_stage'], version=i+1)
        worlds.append(world)


main()
