
class World:
    def __init__(self):
        def __init__(self):
            # COST CONSTANTS #
            self.PROFIT_RENTAL = 0
            self.COST_RELOCATION = 0
            self.COST_DEVIATION = 0

            # TIME CONSTANTS #
            self.HANDLING_TIME_PARKING = 0
            self.HANDLING_TIME_CHARGING = 0
            self.PLANING_PERIOD = 0

            # WORLD CONSTANTS #
            self.YCORD = 0
            self.XCORD = 0

            # COORDINATE CONSTANTS
            self.UPPERRIGHT = (0, 0)
            self.LOWERLEFT = (0, 0)

            # ENTITIES #
            self.employees = []
            self.nodes = []
            self.pNodes = []
            self.cNodes = []
            self.distancesB = []
            self.distancesC = []
            self.cars = []
            self.numScenarios = 0
            self.demands = {}
            self.deliveries = {}
            self.cords = []
            self.remove_coordinates = []
            self.bigM = []
            self.firstStageTasks = 1
