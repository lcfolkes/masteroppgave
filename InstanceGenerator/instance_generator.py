from InstanceGenerator.file_writer import write_to_file, write_to_file_yaml
from HelperFiles.helper_functions import read_config
from InstanceGenerator.world import World, create_parking_nodes, create_charging_nodes, create_employees, create_cars
import copy


# WORLD AND ENTITIES
def initializeWorld(world: World):
    cf = read_config('instance_config.yaml')
    create_parking_nodes(world=world, parking_dim=cf['board']['parking_node_dim'])
    create_charging_nodes(world=world, num_charging_nodes=cf['charging_nodes']['num_charging'],
                          parking_nodes=cf['charging_nodes']['parking_nodes'], capacities=cf['charging_nodes']['capacities'],
                          max_capacities=cf['charging_nodes']['max_capacities'])
    create_employees(world=world, num_employees=cf['employees']['num_employees'], start_time_employees=cf['employees']['start_time'],
                     handling_employees=cf['employees']['handling'])

def buildWorld() -> World:
    SPREAD = True
    world = World()
    initializeWorld(world)
    coordinates = []
    if (SPREAD):
        coordinates = world.give_real_coordinates_spread()
    #writeToFile(cords)
    world.create_real_ideal_state()
    #if (len(world.pNodes) > 0):
    #world.calculateDistances()
    #else:
    world.calculate_real_distances(coordinates)
    print("World Initialized")
    create_cars(world=world)
    print("World Created")
    world.calculate_bigM()
    return world

def create_instance_from_world(world: World, num_scenarios: int, num_tasks: int, num_first_stage_tasks: int, version: int) -> World:
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
    filepath = str(len(new_world.parking_nodes)) + "-" + str(new_world.num_scenarios) + "-" + str(new_world.first_stage_tasks) + "-" + str(version)
    #write_to_file(new_world, filepath)
    write_to_file_yaml(new_world, filepath)
    print("Finished")
    return new_world


def main():
    cf = read_config('instance_config.yaml')
    print("\nWELCOME TO THE EXAMPLE CREATOR \n")
    worlds = []
    for i in range(cf['examples']):
        print("Creating instance: ", i)
        world = buildWorld()
        create_instance_from_world(world, num_scenarios=cf['num_scenarios'], num_tasks=cf['tasks']['num_all'],
                                num_first_stage_tasks=cf['tasks']['num_first_stage'], version=i+1)
        worlds.append(world)

main()