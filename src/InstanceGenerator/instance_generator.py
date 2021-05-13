from InstanceGenerator.file_writer import write_to_file_yaml
from HelperFiles.helper_functions import read_config
from InstanceGenerator.world import World, create_parking_nodes, \
    create_charging_nodes, create_employees, create_cars, create_car_moves, set_distances, set_demands
import copy


def build_world(instance_config: str) -> World:
    SPREAD = True
    world = World()
    initialize_world(world, instance_config)
    print("World Initialized")
    create_cars(world=world)
    create_car_moves(world=world)
    print("World Created")
    world.calculate_bigM()
    return world


def initialize_world(world: World, instance_config: str):
    cf = instance_config
    create_parking_nodes(world=world, num_parking_nodes=int(cf['num_parking_nodes']), time_of_day=cf['time_of_day'], num_cars=cf['num_charged_cars'])
    create_charging_nodes(world=world, num_charging_nodes=cf['charging_nodes']['num_charging'],
                          parking_nodes=cf['charging_nodes']['parking_nodes'],
                          capacities=cf['charging_nodes']['capacities'])
    create_employees(world=world, num_employees=cf['employees']['num_employees'],
                     start_time_employees=cf['employees']['start_time'],
                     handling_employees=cf['employees']['handling'])
    set_distances(world=world, parking_node_nums=cf['charging_nodes']['parking_nodes'])


def create_instance_from_world(world: World, num_scenarios: int, num_tasks: int, num_first_stage_tasks: int,
                               version: int, time_of_day: int, planning_period: int) -> World:
    new_world = copy.deepcopy(world)
    print("Setting number of scenarios...")
    new_world.set_num_scenarios(n=num_scenarios)
    print("Number of scenarios set")
    print("Setting number of tasks...")
    new_world.set_num_tasks(n=num_tasks)
    print("Number of tasks set")
    print("Setting number of first stage tasks...")
    new_world.set_num_first_stage_tasks(n=num_first_stage_tasks)
    print("Number of first stage tasks set")
    print("Setting planning period...")
    new_world.set_planning_period(planning_period=planning_period)
    print("planning period set")
    print("setting demands...")
    set_demands(new_world, time_of_day)
    print("demands set")
    print("Setting total moves, available cars and cars to charge")
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
    cf = read_config('./InstanceGenerator/InstanceConfigs/instance_config20.yaml')
    print("\nWELCOME TO THE EXAMPLE CREATOR \n")
    worlds = []
    for i in range(cf['examples']):
        print("Creating instance: ", i)
        world = build_world(instance_config=cf)
        create_instance_from_world(world, num_scenarios=cf['num_scenarios'], num_tasks=cf['tasks']['num_all'],
                                   num_first_stage_tasks=cf['tasks']['num_first_stage'], version=i + 1,
                                   time_of_day=cf['time_of_day'], planning_period=cf['planning_period'])
        # create_instance_from_world(world, num_scenarios=1, num_tasks=cf['tasks']['num_all'],
        #                       num_first_stage_tasks=cf['tasks']['num_first_stage'], version=i+1)
        worlds.append(world)


main()
