# imports
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import product
from src.HelperFiles.helper_functions import read_config, read_2d_array_to_dict, create_dict_of_indices, create_car_moves_origin_destination


class GurobiInstance:
    def __init__(self, filepath: str, model_type: int, input_model=None):
        self.cf = read_config(filepath)
        # print("SCENARIOS: ", SCENARIOS)
        self.NODES = np.arange(1, self.cf['num_parking_nodes'] + self.cf['num_charging_nodes'] + 1)  # N, set of nodes
        # print("NODES: ", NODES)
        self.CHARGING_NODES = np.arange(self.cf['num_parking_nodes'] + 1, self.cf['num_parking_nodes'] + self.cf[
            'num_charging_nodes'] + 1)  # N^C, set of charging nodes
        # print("CHARGING NODES: ", CHARGING_NODES)
        self.PARKING_NODES = np.arange(1, self.cf['num_parking_nodes'] + 1)  # N^P, set of parking nodes
        # print("PARKING NODES: ", PARKING_NODES)
        # NPS = list(np.arange(NUM_SURPLUS_NODES)) # N^(P+), set of surplus nodes
        self.PARKING_NEED_CHARGING_NODES = self.cf[
            'nodes_with_cars_in_need_of_charging']  # N^(PC), set of parking nodes with cars in need of charging
        # print("PARKING_NEED_CHARGING_NODES: ", PARKING_NEED_CHARGING_NODES)
        self.CARS = np.arange(1, self.cf['num_cars'] + 1)  # C, set of cars potentially subject to relocation
        # print("CARS: ", CARS)
        self.CARMOVES = np.arange(1, self.cf['num_car_moves_parking'] + self.cf['num_car_moves_charging'] + 1)  # R, set of car-moves
        self.CARMOVES_CAR = create_dict_of_indices(self.CARS, self.cf['car_move_cars'])  # R_c, set of car-moves for car c
        self.PARKING_MOVES = np.arange(1, self.cf['num_car_moves_parking'] + 1)  # set of PARKING-moves, R_i^PD and R_i^PO
        # print(PARKING_MOVES)
        self.CHARGING_MOVES = np.arange(self.cf['num_car_moves_parking'] + 1,
                                        self.cf['num_car_moves_parking'] + self.cf['num_car_moves_charging'] + 1)  # set of charging-moves, trenger  R_i^CD and R_i^CO
        self.EMPLOYEES = np.arange(1, self.cf['num_employees'] + 1)  # K, set of service employees
        self.TASKS = np.arange(1, self.cf['num_tasks'] + 1)  # M, set of possible abstract tasks

        # DECLARE PARAMETERS
        # SCENARIO_PROBABILITY = dict(zip(SCENARIOS, self.cf.scenarioProb)) # P_s, Probability of scenario s
        # print("SCENARIO_PROBABILITY: ", SCENARIO_PROBABILITY)
        self.PROFIT_RENTAL = self.cf['profit_rental']  # C^(RC), Profit per unit of time of renting out cars
        self.COST_RELOCATION = self.cf['cost_relocation']  # C^R, Cost per unit of time of relocation activities
        self.COST_DEVIATION = self.cf['cost_deviation']  # C^D, Cost per car of not reaching ideal state
        self.INITIAL_AVAILABLE_CARS = dict(zip(self.PARKING_NODES, self.cf[
            'cars_available_in_node']))  # S^(In)_i, Initial number of available cars (cars not in need of charging)
        self.IDEAL_STATE = dict(
            zip(self.PARKING_NODES, self.cf['ideal_state']))  # S_i^(P) , Ideal state in parking node i
        self.INITIAL_NEED_CHARGING = dict(zip(self.PARKING_NEED_CHARGING_NODES, self.cf[
            'cars_in_need_of_charging_at_nodes']))  # S_i^C, Initial number of cars that require charging at parking node i
        self.INITIAL_AVAILABLE_CAPACITY = dict(zip(self.CHARGING_NODES, self.cf[
            'charging_slots_available']))  # N_i^(CS), Initial available capacity of charging node i
        self.TASKS_FIRST_STAGE = np.arange(1, self.cf[
            'num_first_stage_tasks'] + 1)  # F, number of tasks in first stage, check that F is smaller or equal to len(M)
        self.TASKS_SECOND_STAGE = np.arange(self.cf['num_first_stage_tasks'] + 1, self.cf['num_tasks'] + 1)
        self.PARKING_MOVES_ORIGINATING_IN_NODE, self.PARKING_MOVES_ENDING_IN_NODE, self.CHARGING_MOVES_ORIGINATING_IN_NODE, self.CHARGING_MOVES_ENDING_IN_NODE \
            = create_car_moves_origin_destination(self.PARKING_NODES, self.CHARGING_NODES, self.cf['car_move_origin'],
                                                  self.cf['car_move_destination'])
        # R_i^(PO), R_i^(PD), R_i^(CO), R_i^(CD)
        self.CARMOVE_ORIGIN = dict(zip(self.CARMOVES, self.cf['car_move_origin']))  # o(r), origin node of car move r
        self.CARMOVE_DESTINATION = dict(
            zip(self.CARMOVES, self.cf['car_move_destination']))  # d(r), destination node of car move r
        self.CARMOVE_START_TIME = dict(zip(self.CARMOVES, self.cf['car_move_start_time']))
        self.RELOCATION_TIME = dict(
            zip(self.CARMOVES, self.cf['car_move_handling_time']))  # T_r^H, time needed to perform car-move r
        self.START_TIME_EMPLOYEE = dict(zip(self.EMPLOYEES, self.cf[
            'travel_time_to_origin']))  # T_k^(SO), earliest start time of service employee k
        self.EMPLOYEE_START_LOCATION = dict(
            zip(self.EMPLOYEES, self.cf['start_node_employee']))  # o(k), location of employee k at time T^(SO)_k
        self.TRAVEL_TIME = self.cf['travel_time_bike']  # T_(ij), travel times between node i and j
        self.PLANNING_PERIOD = self.cf['planning_period']  # T^bar, planning period
        self.BIGM = dict(zip(self.CARMOVES, self.cf['bigM']))
        self.scenarios = np.arange(1, self.cf['num_scenarios'] + 1)

        self.CUSTOMER_REQUESTS = {}
        self.CUSTOMER_DELIVERIES = {}

        self.m = self.create_model(model_type, input_model)

    # Model type
    # 0: stochastic
    # 1: deterministic
    # 3: Expectation of the expected value problem
    def create_model(self, model_type: int, input_model=None):
        if model_type == 0:
            print("\nCreating stochastic Model")
        elif model_type == 1:
            print("\nCreating deterministic Model")
        elif model_type == 2:
            if input_model is not None:
                print("\nCreating EEV Model")
            else:
                print("\nNeed input_model argument")
                return

        # INITIALIZE NODE SETS
        if model_type == 0 or model_type == 3:
            self.SCENARIOS = np.arange(1, self.cf['num_scenarios'] + 1)  # S, set of scenarios
            self.SCENARIO_PROBABILITY = 1 / self.cf['num_scenarios']

        else:
            self.SCENARIOS = np.arange(1, 2)
            self.SCENARIO_PROBABILITY = 1

        if model_type == 0 or model_type == 3:
            self.CUSTOMER_REQUESTS = read_2d_array_to_dict(self.cf['customer_requests'])  # R_(is), number of customer requests in second stage in parking node i in scenario s in second stage
            self.CUSTOMER_DELIVERIES = read_2d_array_to_dict(self.cf['car_returns'])  # D_(is), number of vehicles delivered by customers in node i and scenario s in second stage
        else:
            deterministic_customer_request = {}
            deterministic_customer_deliveries = {}
            for i in range(len(self.PARKING_NODES)):
                deterministic_customer_request[
                    (i + 1, 1)] = 1  # round(sum(self.cf.demandP[i])/len(self.cf.demandP[i]),0)
                deterministic_customer_deliveries[
                    (i + 1, 1)] = 1  # round(sum(self.cf.deliveriesP[i])/len(self.cf.deliveriesP[i]),0)
            self.CUSTOMER_REQUESTS = deterministic_customer_request
            self.CUSTOMER_DELIVERIES = deterministic_customer_deliveries

        # Create a new Model
        m: gp.Model = gp.Model("mip1")

        # Create variables
        x = m.addVars(product(self.EMPLOYEES, self.CARMOVES, self.TASKS, self.SCENARIOS), vtype=GRB.BINARY,
                      name="x")  # x_krms, 1 if service employee k performs car-move r as task number m in scenario s, 0 otherwise
        if model_type == 3:
            for v in input_model.m.getVars():
                # x[krms], m < tasksfirsstage
                # t[kms], m < tasksfirststage
                if v.varName[0] == 'x':
                    var_indices = list(map(int, (v.varName[2:-1].split(','))))
                    if var_indices[2] < len(self.TASKS_FIRST_STAGE) + 1:
                        for s in self.SCENARIOS:
                            var_indices[-1] = s
                            x[tuple(var_indices)].lb = v.x
                            x[tuple(var_indices)].ub = v.x

        y = m.addVars(self.PARKING_NODES, lb=0, vtype=GRB.INTEGER,
                      name="y")  # y_i, Number of cars in node i by the beginning of the second stage
        z = m.addVars(product(self.PARKING_NODES, self.SCENARIOS), lb=0, vtype=GRB.INTEGER,
                      name="z")  # z_is, number of customer requests served in second stage in node i in scenario s
        w = m.addVars(product(self.PARKING_NODES, self.SCENARIOS), lb=0, vtype=GRB.INTEGER,
                      name="w")  # w_is, number of cars short of ideal state in node i in scenario s
        t = m.addVars(product(self.EMPLOYEES, self.TASKS, self.SCENARIOS), vtype=GRB.CONTINUOUS, lb=0, name="t")
        # print(x)
        # print(y)
        # print(z)
        # print(w)

        ### OBJECTIVE FUNCTION ###
        # profit from customer requests
        profit_customer_requests = gp.quicksum(
            (self.PROFIT_RENTAL) * z[(i, s)] for i in self.PARKING_NODES for s in self.SCENARIOS)

        # costs from relocation activities
        costs_relocation = gp.quicksum(self.COST_RELOCATION * self.RELOCATION_TIME[r] * x[(k, r, m, s)]
                                       for m in self.TASKS
                                       for r in self.CARMOVES
                                       for k in self.EMPLOYEES
                                       for s in self.SCENARIOS)

        # ideal state deviation cost
        cost_deviation_ideal_state = gp.quicksum(
            self.COST_DEVIATION * w[(i, s)] for i in self.PARKING_NODES for s in self.SCENARIOS)

        total_profit = self.SCENARIO_PROBABILITY * (
                profit_customer_requests - costs_relocation - cost_deviation_ideal_state)

        m.setObjective(total_profit, GRB.MAXIMIZE)

        ### CONSTRAINTS ###

        ## Relocation of rental vehicles

        # (2) Ensure each car is moved at most once
        m.addConstrs(
            (gp.quicksum(x[(k, r, m, s)] for k in self.EMPLOYEES for r in self.CARMOVES_CAR[c] for m in self.TASKS) <= 1
             for s in self.SCENARIOS for c in self.CARS), name="c2"
        )
        # print("c2")

        # (3) Make sure each task of each employee in every scenario consists of at most one car-move
        m.addConstrs(
            (gp.quicksum(x[(k, r, m, s)] for r in self.CARMOVES) <= 1 for k in self.EMPLOYEES for m in self.TASKS for s
             in self.SCENARIOS), name="c3"
        )
        # print("c3")

        # (4) Ensure tasks are performed in ascending order by task number
        m.addConstrs(
            (gp.quicksum(x[(k, r, m + 1, s)] for r in self.CARMOVES) <= gp.quicksum(
                x[(k, r, m, s)] for r in self.CARMOVES)
             for k in self.EMPLOYEES for m in self.TASKS[:-1] for s in self.SCENARIOS), name="c4"
        )
        # print("c4")

        # (5) All cars in need of charging must be moved to a charging station within the planning period
        m.addConstrs(
            ((gp.quicksum(
                x[(k, r, m, s)] for k in self.EMPLOYEES for r in self.CHARGING_MOVES_ORIGINATING_IN_NODE[i] for m in
                self.TASKS) == self.INITIAL_NEED_CHARGING[i])
             for i in self.PARKING_NEED_CHARGING_NODES for s in self.SCENARIOS), name="c5"
        )
        # print("c5")

        ## Node states

        # (6) Ensure capacity of charging stations is not exceeded in any of the scenarios
        m.addConstrs(
            ((gp.quicksum(
                x[(k, r, m, s)] for k in self.EMPLOYEES for r in self.CHARGING_MOVES_ENDING_IN_NODE[i] for m in
                self.TASKS) <= self.INITIAL_AVAILABLE_CAPACITY[i])
             for i in self.CHARGING_NODES for s in self.SCENARIOS), name="c6"
        )
        # print("c6")

        # (7) Calculate number of available cars in each parking node by the beginning of the second stage
        m.addConstrs((y[i] == self.INITIAL_AVAILABLE_CARS[i]
                      + gp.quicksum(
            x[(k, r, m, s)] for k in self.EMPLOYEES for r in self.PARKING_MOVES_ENDING_IN_NODE[i] for m in
            self.TASKS_FIRST_STAGE)
                      - gp.quicksum(
            x[(k, r, m, s)] for k in self.EMPLOYEES for r in self.PARKING_MOVES_ORIGINATING_IN_NODE[i] for m in
            self.TASKS_FIRST_STAGE)
                      for i in self.PARKING_NODES for s in self.SCENARIOS), name="c7"
                     )
        # print("c7")

        ## Demand during planning period

        # (8) number of customer requests served in each parking node in the second stage is less than available cars
        # in the beginning of the second stage
        m.addConstrs((z[(i, s)] <= y[i] + self.CUSTOMER_DELIVERIES[(i, s)] - gp.quicksum(x[(k, r, m, s)]
                                                                                         for k in self.EMPLOYEES
                                                                                         for r in
                                                                                         self.PARKING_MOVES_ORIGINATING_IN_NODE[
                                                                                             i]
                                                                                         for m in
                                                                                         self.TASKS_SECOND_STAGE)
                      for i in self.PARKING_NODES for s in self.SCENARIOS), name="c8")
        # print("c8")

        # (9) number of customer requests served in each parking node in the second stage is less than the number of
        # customer requests in the beginning of the second stage

        m.addConstrs((z[(i, s)] <= self.CUSTOMER_REQUESTS[(i, s)] for i in self.PARKING_NODES for s in self.SCENARIOS),
                     name="c9")
        # print("c9")

        # (10) calculate the number of cars short of the ideal state in every node in each scenario at the end
        # of the planning period
        m.addConstrs((w[(i, s)] >=
                      self.IDEAL_STATE[i]
                      - self.INITIAL_AVAILABLE_CARS[i]
                      - gp.quicksum(
            x[(k, r, m, s)] for k in self.EMPLOYEES for r in self.PARKING_MOVES_ENDING_IN_NODE[i] for m in self.TASKS)
                      + gp.quicksum(
            x[(k, r, m, s)] for k in self.EMPLOYEES for r in self.PARKING_MOVES_ORIGINATING_IN_NODE[i] for m in
            self.TASKS)
                      + z[(i, s)]
                      - self.CUSTOMER_DELIVERIES[(i, s)] for i in self.PARKING_NODES for s in self.SCENARIOS),
                     name="c10"
                     )
        # print("c10")

        ## Time tracking of node visits

        # (11) ensure that no task can be finished after the end of the planning period

        m.addConstrs(
            (t[(k, m, s)]
             + self.RELOCATION_TIME[r] * x[(k, r, m, s)]
             + gp.quicksum(
                self.TRAVEL_TIME[self.CARMOVE_DESTINATION[r] - 1][self.CARMOVE_ORIGIN[v] - 1] * x[(k, v, m + 1, s)]
                for v in self.CARMOVES)
             - self.BIGM[r] * (1 - x[(k, r, m, s)])
             <= t[(k, m + 1, s)]
             for k in self.EMPLOYEES for r in self.CARMOVES for m in self.TASKS[:-1] for s in self.SCENARIOS),
            name="c11")

        m.addConstrs(
            (t[(k, m, s)] <= t[(k, m + 1, s)]
             for k in self.EMPLOYEES for m in self.TASKS[:-1] for s in self.SCENARIOS), name="c12")

        m.addConstrs(((self.START_TIME_EMPLOYEE[k] + self.TRAVEL_TIME[self.EMPLOYEE_START_LOCATION[k] - 1][
            self.CARMOVE_ORIGIN[r] - 1]) * x[(k, r, 1, s)]
                      <= t[(k, 1, s)] for k in self.EMPLOYEES for r in self.CARMOVES for s in self.SCENARIOS),
                     name="c13")

        m.addConstrs((self.CARMOVE_START_TIME[r] * x[(k, r, m, s)] <= t[(k, m, s)]
                      for k in self.EMPLOYEES for r in self.CARMOVES for m in self.TASKS for s in self.SCENARIOS),
                     name="c14")

        m.addConstrs((t[(k, self.TASKS[-1], s)] + gp.quicksum(
            self.RELOCATION_TIME[r] * x[(k, r, self.TASKS[-1], s)] for r in self.CARMOVES) <=
                      self.PLANNING_PERIOD for k in self.EMPLOYEES for s in self.SCENARIOS), name="c15")

        # (15) ensure that all first-stage decisions are equal regardless of scenario
        m.addConstrs((x[(k, r, m, s)] == x[(k, r, m, s + 1)]
                      for k in self.EMPLOYEES
                      for r in self.CARMOVES
                      for m in self.TASKS_FIRST_STAGE
                      for s in self.SCENARIOS[:-1]), name="c15")

        m.addConstrs((t[(k, m, s)] == t[(k, m, s + 1)]
                      for k in self.EMPLOYEES
                      for m in self.TASKS_FIRST_STAGE
                      for s in self.SCENARIOS[:-1]), name="extra")

        return m