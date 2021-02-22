from InstanceGenerator.world import *
import copy


# WORLD AND ENTITIES
def initializeWorld(world: World):
    createNodes(world)
    createCNodes(world)
    createEmployees(world)

def buildWorld():
    world = World()
    initializeWorld(world)
    cords = []
    if (SPREAD):
        cords = world.giveRealCoordinatesSpread()
    writeToFile(cords)
    world.createRealIdeal()
    #if (len(world.pNodes) > 0):
    #world.calculateDistances()
    #else:
    world.calculateRealDistances(cords)
    print("World Initialized")
    createCars(world)
    print("World Created")
    world.calculateBigM()
    print(world.bigM)
    return world

def createInstanceFromWorld(world: World, num_scenarios: int, num_first_stage_tasks: int, version: int) -> World:
    new_world = copy.deepcopy(world)
    new_world.set_scenarios(n=num_scenarios)
    new_world.set_num_first_stage_tasks(n=num_first_stage_tasks)
    new_world.set_demands()
    totalMoves = 0
    for j in range(len(world.cars)):
        totalMoves += len(world.cars[j].destinations)
    available_cars = 0
    cars_to_charging = 0
    for n in new_world.pNodes:
        available_cars += n.pState
        cars_to_charging += n.cState
    filepath = str(len(new_world.pNodes)) + "-" + str(new_world.numScenarios) + "-" + str(new_world.firstStageTasks) + "-" + str(version)
    new_world.writeToFile(filepath)
    print("Finished")
    return new_world


def main():
    cf = read_config('instance_config.yaml')
    print("\nWELCOME TO THE EXAMPLE CREATOR \n")
    worlds = []
    for i in range(cf['examples']):
        print("Creating instance: ", i)
        world = buildWorld()
        createInstanceFromWorld(world, num_scenarios=cf['num_scenarios'],
                                num_first_stage_tasks=['tasks']['num_first_stage'], version=i+1)
        worlds.append(world)

main()