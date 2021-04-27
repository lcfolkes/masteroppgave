import copy
import itertools

import numpy as np

from Heuristics.helper_functions_heuristics import get_first_stage_solution
from InstanceGenerator.instance_components import ParkingNode, CarMove
from InstanceGenerator.world import World


class ObjectiveFunction():

    def __init__(self, world_instance):
        self.world_instance = world_instance
        self.parking_nodes = world_instance.parking_nodes
        self.num_scenarios = world_instance.num_scenarios
        self.y = {parking_node.node_id: parking_node.parking_state for parking_node in self.parking_nodes}
        self.node_demands = {parking_node.node_id: {
            'customer_requests': parking_node.customer_requests,
            'car_returns': parking_node.car_returns} for parking_node in self.parking_nodes}
        #self.z = {parking_node.node_id: 0 for parking_node in self.parking_nodes}

        self.w = {n.node_id: (n.ideal_state - n.parking_state - n.car_returns) for n in self.parking_nodes}
        self.num_cars_in_need_of_charging = sum(pnode.charging_state for pnode in self.parking_nodes)


    def calculate_z(self, first_stage_car_moves: [CarMove], second_stage_car_moves: [[CarMove]],
                    new_car_move: CarMove = None, scenario: int = None, first_stage_hash = None, second_stage_hash = None,  verbose: bool = False) -> {int: np.array([int])}:

        # z is the number of customer requests served. It must be the lower of the two values
        # available cars and the number of customer requests D_is
        # number of available cars in the beginning of the second stage, y_i
        """
        :param parking_nodes: list of parking node objects, [pn1, pn2]
        :param first_stage_car_moves: list of car_move objects for first stage, [cm1, cm2]
        :param second_stage_car_moves: [[cm2],[cm2],[cm2, cm3]]
        :param verbose: if True print information
        :return: z, dictionary with node_id as keys and a numpy array for each scenario as value,
        {node_id: np.array([0, 2, 1])}
        """
        start_nodes_first_stage = [car_move.start_node for car_move in first_stage_car_moves if not car_move.is_charging_move]
        end_nodes_first_stage = [car_move.end_node for car_move in first_stage_car_moves if not car_move.is_charging_move]

        y = self.y.copy()
        for n in start_nodes_first_stage:
            y[n.node_id] -= 1
        for n in end_nodes_first_stage:
            y[n.node_id] += 1

        node_demands = self.node_demands.copy()

        z = {}

        start_nodes_second_stage = [
            [car_move.start_node.node_id for car_move in scenarios if not car_move.is_charging_move]
            for scenarios in second_stage_car_moves]

        # car_move.start_node should be a list of car moves with len(list) = num_scenarios
        for n in self.parking_nodes:
            second_stage_moves_out = np.array([scenario.count(n.node_id) for scenario in start_nodes_second_stage])
            #y[n.node_id] = np.maximum(y[n.node_id], 0)

            z_val = np.minimum(y[n.node_id] + node_demands[n.node_id]['car_returns'] - second_stage_moves_out,
                               node_demands[n.node_id]['customer_requests'])
            z_val = np.maximum(z_val, 0)
            z[n.node_id] = z_val



        return z


    def calculate_profit_customer_requests(self, z: {int: np.array([int])}, scenario: int = None) -> float:
        """
        :param z: dictionary with node_id as keys and a numpy array for each scenario as value,
        {node_id: np.array([0, 2, 1])}
        :param scenario: integer describing scenario. If given, only calculate profit from request for that scenario
        :return: float with profit. if scenarios is given, then avg. over scenarios is returned, else value for given
        scenario is returned
        """
        # sum across scenarios for all nodes
        z_sum = sum(v for k, v in z.items())
        if scenario is None:
            # print(f"z_sum {z_sum}")
            z_sum_scenario_average = np.mean(z_sum)
            return World.PROFIT_RENTAL * z_sum_scenario_average
        else:
            # print(f"z_sum[{scenario+1}] {z_sum[scenario]}")
            return World.PROFIT_RENTAL * z_sum[scenario]


    def calculate_costs_relocation(self, car_moves: [CarMove],
                                   individual_scenario: bool = False) -> float:
        """
        :param car_moves: list of scenarios containing car_moves, [[cm1],[cm1],[cm1, cm2]]
        :param num_scenarios: int
        :param individual_scenario: boolean. if individual scenario then you do not average over scenarios
        :return: float with costs for relocation
        """
        # Sum of all travel times across all car moves

        sum_travel_time = sum(car_move.handling_time for car_move in car_moves)
        if individual_scenario:
            # print(f"individual_scenario {sum_travel_time}")
            return World.COST_RELOCATION * sum_travel_time
        else:
            sum_travel_time_scenario_avg = sum_travel_time / self.num_scenarios
            # print(f"sum_scenarios {sum_travel_time}")
            # print(f"avg_scenarios {sum_travel_time_scenario_avg}")
            return World.COST_RELOCATION * sum_travel_time_scenario_avg


    def calculate_cost_deviation_ideal_state(self, z: {int: np.array([int])},
                                             first_stage_car_moves: [CarMove], second_stage_car_moves: [[CarMove]],
                                             scenario: int = None, verbose: bool = False) -> float:
        """
        :param parking_nodes: list of parking node objects
        :param z: dictionary with node_id as keys and a numpy array for each scenario as value, {node_id: np.array([0, 2, 1])}
        :param first_stage_car_moves: list of car_move objects for first stage, [cm1, cm2]
        :param second_stage_car_moves: [[cm2],[cm2],[cm2, cm3]]
        :param scenario: if None, then average for all scenario is calculated, else for a specific scenario
        :param verbose: True if you want to print information
        :return: return float with the cost associated with deviation from the ideal state.
        """

        start_nodes_first_stage = [car_move.start_node for car_move in first_stage_car_moves if not car_move.is_charging_move]
        end_nodes_first_stage = [car_move.end_node for car_move in first_stage_car_moves if not car_move.is_charging_move]

        w = {}
        for node_id in z.keys():
            w[node_id] = self.w[node_id] + z[node_id]

        for n in start_nodes_first_stage:
            w[n.node_id] += 1
        for n in end_nodes_first_stage:
            w[n.node_id] -= 1

        start_nodes_second_stage = [
            [car_move.start_node.node_id for car_move in scenarios if not car_move.is_charging_move] for
            scenarios in second_stage_car_moves]

        end_nodes_second_stage = [
            [car_move.end_node.node_id for car_move in scenarios if not car_move.is_charging_move] for
            scenarios in second_stage_car_moves]

        for n in self.parking_nodes:
            second_stage_moves_out = np.array([cm.count(n.node_id) for cm in start_nodes_second_stage])
            second_stage_moves_in = np.array([cm.count(n.node_id) for cm in end_nodes_second_stage])

            w[n.node_id] += second_stage_moves_out - second_stage_moves_in
            # require w_is >= 0
            w[n.node_id] = np.maximum(w[n.node_id], 0)

            '''
            if verbose:
                print(f"\nw[{n.node_id}] {w[n.node_id]}")
                print(f"ideal state {n.ideal_state}")
                print(f"initial_state {n.parking_state}")
                print(f"car returns {n.car_returns}")
                print(f"customer requests {n.customer_requests}")'''



        w_sum = sum(v for k, v in w.items())

        if scenario is None:
            w_sum_scenario_average = np.mean(w_sum)
            # print(f"w_sum {w_sum}")
            return World.COST_DEVIATION * w_sum_scenario_average
        else:
            # print(f"w_sum[{scenario+1}] {w_sum[scenario]}")
            return World.COST_DEVIATION * w_sum[scenario]

    def get_first_stage_hash(self, first_stage_car_moves):
        cm_ids = [cm.car_move_id for cm in first_stage_car_moves]
        #permutation_object = itertools.permutations(cm_ids)
        return hash(str(cm_ids.sort()))

    def get_second_stage_hash(self, second_stage_car_moves):
        hash_list = []
        for scenario in second_stage_car_moves:
            cm_ids = [cm.car_move_id for cm in scenario]
            # permutation_object = itertools.permutations(cm_ids)
            hash_list.append(hash(str(cm_ids.sort())))
        return hash_list

    def calculate_cost_deviation_charging_moves(self, first_stage_car_moves: [CarMove],
                                                second_stage_car_moves: [[CarMove]], scenario=None):
        # print("first_stage_car_moves: ", first_stage_car_moves)
        # print("second_stage_car_moves: ", second_stage_car_moves)
        # print("scenario: ", scenario)
        num_cars_in_need_of_charging = self.num_cars_in_need_of_charging
        num_charging_moves_first_stage = sum(
            1 for cm in first_stage_car_moves if cm.is_charging_move)
        num_charging_moves_second_stage = [0 for _ in range(len(second_stage_car_moves))]
        if any(second_stage_car_moves):
            for s in range(len(second_stage_car_moves)):
                num_charging_moves_second_stage[s] = sum(1 for cm in second_stage_car_moves[s]
                                                         if cm.is_charging_move)

        if scenario is None:
            num_charging_moves = num_charging_moves_first_stage + np.mean(num_charging_moves_second_stage)
        # print(f"w_sum {w_sum}")
        else:
            num_charging_moves = num_charging_moves_first_stage + num_charging_moves_second_stage[scenario]
        # print(f"w_sum[{scenario+1}] {w_sum[scenario]}")

        # return World.COST_DEVIATION_CHARGING * (num_cars_in_need_of_charging - num_charging_moves)
        return 1000 * (num_cars_in_need_of_charging - num_charging_moves)

    def get_obj_val_of_car_moves(self, first_stage_car_moves: [CarMove] = None,
                                 second_stage_car_moves: [[CarMove]] = None, scenario: int = None,
                                 new_car_move: CarMove = None, verbose: bool = False, include_employee_check=True) -> float:
        """
        :param parking_nodes: list of parking node objects
        :param num_scenarios:
        :param travel_time_between_car_moves: travel time between car_moves for all employees
        :param first_stage_car_moves: list of car_move objects for first stage, [cm1, cm2]
        :param second_stage_car_moves: [[cm2],[cm2],[cm2, cm3]]
        :param scenario: if None, then average for all scenario is calculated, else for a specific scenario
        :param verbose:  True if you want to print information
        :return: float of objective value of car_moves
        """
        '''
        print(f"y: {self.y}")
        print(f"node_demands: {self.node_demands}")
        print(f"w: {self.w}")
        print(f"num_cars_in_need_of_charging: {self.num_cars_in_need_of_charging}")'''
        first_stage_hash = self.get_first_stage_hash(first_stage_car_moves)
        if first_stage_hash not in self.first_stage_hash_set:
            first_stage_hash = None
        second_stage_hash = self.get_second_stage_hash(second_stage_car_moves)
        if second_stage_hash not in self.second_stage_hash_set:
            first_stage_hash = None


        if scenario is None:
            if (second_stage_car_moves is None) or second_stage_car_moves == []:
                second_stage_car_moves = [[]]
            z = self.calculate_z(first_stage_car_moves=first_stage_car_moves,
                                 second_stage_car_moves=second_stage_car_moves,
                                 new_car_move=new_car_move,
                                 first_stage_hash=first_stage_hash,
                                 second_stage_hash=second_stage_hash)
            profit_customer_requests = self.calculate_profit_customer_requests(z)
            cost_deviation_ideal_state = self.calculate_cost_deviation_ideal_state(z=z,
                                                                              first_stage_car_moves=first_stage_car_moves,
                                                                              second_stage_car_moves=second_stage_car_moves)

            first_stage_duplicate_for_scenarios = list(np.repeat(first_stage_car_moves, self.num_scenarios))
            cost_relocation = self.calculate_costs_relocation(first_stage_duplicate_for_scenarios)
            cost_deviation_charging_moves = self.calculate_cost_deviation_charging_moves(first_stage_car_moves=first_stage_car_moves,
                                                                                    second_stage_car_moves=second_stage_car_moves)

        else:
            car_moves_second_stage = [[] for _ in range(self.num_scenarios)]
            car_moves_second_stage[scenario] = second_stage_car_moves
            z = self.calculate_z(first_stage_car_moves=first_stage_car_moves, second_stage_car_moves=car_moves_second_stage)  # , verbose=True)
            profit_customer_requests = self.calculate_profit_customer_requests(z, scenario=scenario)
            cost_deviation_ideal_state = self.calculate_cost_deviation_ideal_state(z=z,
                                                                              first_stage_car_moves=first_stage_car_moves,
                                                                              second_stage_car_moves=car_moves_second_stage,
                                                                              scenario=scenario)

            # first_stage_duplicate_for_scenarios = list(np.repeat(first_stage_car_moves, self.num_scenarios))
            cost_relocation = self.calculate_costs_relocation(first_stage_car_moves + second_stage_car_moves, self.num_scenarios,
                                                         individual_scenario=True)
            cost_deviation_charging_moves = self.calculate_cost_deviation_charging_moves(first_stage_car_moves=first_stage_car_moves,
                                                                                    second_stage_car_moves=car_moves_second_stage,
                                                                                    scenario=scenario)
        if include_employee_check:
            cost_travel_time_between_car_moves = self.calculate_cost_travel_time_between_car_moves(
                first_stage_car_moves=first_stage_car_moves, second_stage_car_moves=second_stage_car_moves,
                scenario=scenario)
        else:
            cost_travel_time_between_car_moves = 0

        return profit_customer_requests - cost_relocation - cost_deviation_ideal_state - cost_deviation_charging_moves - \
               cost_travel_time_between_car_moves


if __name__ == "__main__":
    from Heuristics.construction_heuristic import ConstructionHeuristic
    from Heuristics.feasibility_checker import FeasibilityChecker

    filename = "InstanceGenerator/InstanceFiles/20nodes/20-10-2-1_e"
    ch = ConstructionHeuristic(filename + ".pkl")
    ch.construct()
    true_obj_val, best_obj_val = ch.get_obj_val(both=True)
    # print(f"Construction heuristic true obj. val {true_obj_val}")
    ch.print_solution()
    first_stage_solution = get_first_stage_solution(ch.assigned_car_moves, ch.world_instance.first_stage_tasks)
    feasibility_checker = FeasibilityChecker(ch.world_instance)
    feasibility_checker.is_first_stage_solution_feasible(first_stage_solution, False)
