from InstanceGenerator.helper_functions import *
import numpy as np

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


    # COORDINATE CONSTANTS
    LOWERLEFT = (cf['board']['coordinates']['lower_left']['lat'],
                      cf['board']['coordinates']['lower_left']['long'])
    UPPERRIGHT = (cf['board']['coordinates']['upper_right']['lat'],
                       cf['board']['coordinates']['upper_right']['long'])

    def __init__(self):
        # PARKING GRID
        self.XCORD = 0
        self.YCORD = 0

        # TASKS
        self.tasks = 0
        self.firstStageTasks = 0

        # SCENARIOS
        self.numScenarios = 0

        # ENTITIES #
        self.employees = []
        self.nodes = []
        self.pNodes = []
        self.cNodes = []
        self.distancesB = []
        self.distancesC = []
        self.cars = []
        self.demands = {}
        self.deliveries = {}
        self.cords = []
        self.remove_coordinates = []
        self.bigM = []

    def addDim(self, xCord, yCord):
        self.XCORD = xCord
        self.YCORD = yCord

    def addNodes(self, node):
        self.nodes.append(node)

    def addPNodes(self, pNode):
        self.pNodes.append(pNode)

    def addcNodes(self, cNode):
        self.cNodes.append(cNode)

    def addEmployee(self, employee):
        self.employees.append(employee)

    def addCar(self, car):
        self.cars.append(car)

    def setScenarios(self, n):
        self.numScenarios = n

    def setNumFirstStageTasks(self, n):
        self.firstStageTasks = n

    # customer requests in the second stage and
    # customer deliveries in the second stage
    def setDemands(self):
        for i in range(len(self.pNodes)):
            # np.random.seed(i)
            self.pNodes[i].demand = np.random.choice(DEMAND, size=self.numScenarios, p=DEMANDPROB)
            self.pNodes[i].deliveries = np.random.choice(DELIVERIES, size=self.numScenarios, p=DELIVERIESPROB)

    def addCords(self, cords):
        # cord (y,x)
        self.cords = cords

    ## CALCULATE DISTANCE ##

    def calculateDistances(self):
        self.distancesC = []
        self.distancesB = []
        maxDistance = math.sqrt(math.pow(self.pNodes[0].xCord - self.pNodes[len(self.pNodes) - 1].xCord, 2) + math.pow(
            self.pNodes[0].yCord - self.pNodes[len(self.pNodes) - 1].yCord, 2))
        scale = float(format((self.TIMELIMIT - 1) / (maxDistance * DISTANCESCALE), '.1f'))
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
        stepX = (self.UPPERRIGHT[1] - self.LOWERLEFT[1]) / OPERATING_AREA_SIZE[1]
        stepY = (self.UPPERRIGHT[0] - self.LOWERLEFT[0]) / OPERATING_AREA_SIZE[0]
        startX = self.LOWERLEFT[1] + 0.5 * stepX
        startY = self.UPPERRIGHT[0] - 0.5 * stepY

        grid_cords = []
        # Generate coordinate for each of the 8x12 (y,x) nodes in operating area
        for i in range(OPERATING_AREA_SIZE[0]):
            for j in range(OPERATING_AREA_SIZE[1]):
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
        for i in range(MAXNODES):
            r = random.randint(0, len(grid_cords) - 1)
            cord = grid_cords.pop(r)
            sample_cords.append(cord)

        self.addCords(sample_cords)
        return sample_cords

    def giveRealCoordinatesCluster(self):
        pass

    def calculateRealDistances(self, cords):
        if (len(cords) == 0):
            stepX = (self.UPPERRIGHT[1] - self.LOWERLEFT[1]) / self.XCORD
            stepY = (self.UPPERRIGHT[0] - self.LOWERLEFT[0]) / self.YCORD
            startX = self.LOWERLEFT[1] + 0.5 * stepX
            startY = self.UPPERRIGHT[0] - 0.5 * stepY
            cords = []
            for i in range(self.YCORD):
                for j in range(self.XCORD):
                    cordX = startX + j * stepX
                    cordY = startY - i * stepY
                    cord = (cordY, cordX)
                    cords.append(cord)

        travelMatrixCar = gI.run(cords, "driving", False)
        time.sleep(2)
        travelMatrixBicycle = gI.run(cords, "bicycling", False)
        time.sleep(2)
        travelMatrixTransit = gI.run(cords, "transit", False)

        for i in range(len(travelMatrixBicycle)):
            for j in range(len(self.cNodes)):
                if (self.cNodes[j].pNode - 1 == i):
                    travelMatrixBicycle[i].append(60)
                    travelMatrixTransit[i].append(60)
                    travelMatrixCar[i].append(60)
                else:
                    travelMatrixBicycle[i].append(travelMatrixBicycle[i][self.cNodes[j].pNode - 1])
                    travelMatrixTransit[i].append(travelMatrixTransit[i][self.cNodes[j].pNode - 1])
                    travelMatrixCar[i].append(travelMatrixCar[i][self.cNodes[j].pNode - 1])
        for i in range(len(self.cNodes)):
            travelMatrixBicycle.append(copy.deepcopy(travelMatrixBicycle[self.cNodes[i].pNode - 1]))
            travelMatrixTransit.append(copy.deepcopy(travelMatrixTransit[self.cNodes[i].pNode - 1]))
            travelMatrixCar.append(copy.deepcopy(travelMatrixCar[self.cNodes[i].pNode - 1]))

            travelMatrixBicycle[len(travelMatrixBicycle) - 1][self.cNodes[i].pNode - 1] = 60
            travelMatrixTransit[len(travelMatrixTransit) - 1][self.cNodes[i].pNode - 1] = 60
            travelMatrixCar[len(travelMatrixCar) - 1][self.cNodes[i].pNode - 1] = 60

            # Set distance to charging station to 1 for corresponding node
            travelMatrixBicycle[len(travelMatrixBicycle) - 1][len(self.pNodes) + i] = 1.0
            travelMatrixTransit[len(travelMatrixTransit) - 1][len(self.pNodes) + i] = 1.0
            travelMatrixCar[len(travelMatrixCar) - 1][len(self.pNodes) + i] = 1.0

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
                        for x in range(len(self.pNodes)):
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
    def createRealIdeal(self):
        initialAdd = [0 for i in range(len(self.nodes))]
        for j in range(len(self.employees)):
            if (self.employees[j].handling):
                initialAdd[self.employees[j].startNode - 1] += 1

        sumIState = 0
        sumPState = 0
        # net sum of Requests-Deliveries for each scenario
        for i in range(len(self.pNodes)):
            sumIState += self.pNodes[i].iState
            sumPState += self.pNodes[i].pState + initialAdd[i]

        # ideal state should be scaled to sum of available cars in the worst case

        # Setter max outflow i hver node til 1.1, for å få litt mer spennende instanser
        max_flow = math.ceil(0.6 * len(self.pNodes))
        # max_flow = (max(DEMAND)-min(DELIVERIES))*len(self.pNodes)

        sumIState += max_flow
        # sumIState += int(max(netFlowScenarios))
        sumPStateAfter = 0

        for i in range(len(self.pNodes)):
            # Scale pstate with rounded share of sumpstate
            # Share of pstate = iptate/sumPstate
            self.pNodes[i].pState = int(round(float(sumIState) * (float(self.pNodes[i].pState) / sumPState)))
            sumPStateAfter += self.pNodes[i].pState

        # Correct for errors due to rounding
        while (sumPStateAfter != sumIState):
            if (sumPStateAfter < sumIState):
                r = random.randint(0, len(self.pNodes) - 1)
                self.pNodes[r].pState += 1
                sumPStateAfter += 1
            else:
                r = random.randint(0, len(self.pNodes) - 1)
                if (self.pNodes[r].pState - initialAdd[r] > 0):
                    self.pNodes[r].pState -= 1
                    sumPStateAfter -= 1


