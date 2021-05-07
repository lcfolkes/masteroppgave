import copy
import numpy as np

from Heuristics.helper_functions_heuristics import get_first_stage_solution, copy_numpy_dict
from InstanceGenerator.instance_components import CarMove, Node
from InstanceGenerator.world import World


class ObjectiveFunction:

    def __init__(self, world_instance):
        self.world_instance = world_instance
        self.parking_nodes = world_instance.parking_nodes
        self.parking_node_ids = [n.node_id for n in world_instance.parking_nodes]
        self.num_scenarios = world_instance.num_scenarios

        self._z = self._initialize_z()
        self._w = self._initialize_w()
        self._charging_deviation = self._initialize_charging_deviation()
        self._relocation_time = np.array([0.0 for _ in range(self.num_scenarios)])
        self._true_objective_value = \
            World.PROFIT_RENTAL * np.sum([np.sum([np.min(scenario) for scenario in node]) for node in self._z]) / self.num_scenarios \
            - World.COST_DEVIATION * np.sum([np.sum([np.maximum(w, 0) for w in node]) for node in self._w]) / self.num_scenarios

        self._heuristic_objective_value = \
            self._true_objective_value - 1000 * sum(self._charging_deviation) / self.num_scenarios

    @property
    def heuristic_objective_value(self):
        return self._heuristic_objective_value

    @property
    def true_objective_value(self):
        return self._true_objective_value

    def _initialize_z(self):
        z = [[[node.parking_state + node.car_returns[s], node.customer_requests[s]] for s in range(self.num_scenarios)] for node in self.parking_nodes]

        return z

    def _initialize_w(self):
        w = [[node.ideal_state - node.parking_state + np.min(self._z[node.node_id-1][s]) - node.car_returns[s] for s in range(self.num_scenarios)] for node in self.parking_nodes]

        return w

    def _initialize_charging_deviation(self):
        num_cars_in_need_of_charging = sum(pnode.charging_state for pnode in self.parking_nodes)
        return np.array([num_cars_in_need_of_charging for _ in range(self.num_scenarios)])

    def evaluate(self, added_car_moves: [CarMove] = None, removed_car_moves: [CarMove] = None, scenario: int = None,
                 both="both"):
        """
        :param added_car_moves: list of car moves, e.g: [cm1, cm2]
        :param removed_car_moves: list of car moves, e.g: [cm1, cm2]
        :param scenario: scenario if second stage, e.g.
        :return:
        """
        if added_car_moves is None:
            added_car_moves = []
        if removed_car_moves is None:
            removed_car_moves = []

        nodes_in, nodes_out = get_parking_nodes_in_out(added_car_moves, removed_car_moves)
        self._update_z(nodes_in, nodes_out, scenario)
        #self._update_w(nodes_in, nodes_out, scenario)
        self._update_relocation_time(added_car_moves, removed_car_moves, scenario)
        self._update_charging_deviation(added_car_moves, removed_car_moves, scenario)

        true_objective_value = copy.copy(self._true_objective_value)
        heuristic_objective_value = copy.copy(self._heuristic_objective_value)

        self._update_z(nodes_out, nodes_in, scenario)
        #self._update_w(nodes_out, nodes_in, scenario)
        self._update_relocation_time(removed_car_moves, added_car_moves, scenario)
        self._update_charging_deviation(removed_car_moves, added_car_moves, scenario)

        if both == "both":
            return true_objective_value, heuristic_objective_value
        elif both == "true":
            return true_objective_value
        elif both == "heuristic":
            return heuristic_objective_value

    def update(self, added_car_moves: [CarMove] = None, removed_car_moves: [CarMove] = None, scenario: int = None):

        if added_car_moves is None:
            added_car_moves = []
        if removed_car_moves is None:
            removed_car_moves = []

        nodes_in, nodes_out = get_parking_nodes_in_out(added_car_moves, removed_car_moves)
        self._update_z(nodes_in, nodes_out, scenario)
        #self._update_w(nodes_in, nodes_out, scenario)
        self._update_relocation_time(added_car_moves, removed_car_moves, scenario)
        self._update_charging_deviation(added_car_moves, removed_car_moves, scenario)

    def _update_z(self, nodes_in: [Node] = None, nodes_out: [Node] = None, scenario: int = None):
        """
        :param nodes_in: nodes with cars going into node
        :param nodes_out: nodes with cars leaving node
        :param scenario: if second stage, then the scenario is specified
        """
        # TODO: Only first stage cars that come into a node should update z
        if scenario is None:
            for n in nodes_out:
                for s in range(len(self._z[n.node_id-1])):
                    self._z[n.node_id-1][s][0] -= 1
                    self._w[n.node_id-1][s] += 1
                    #self._w[n.node_id-1][s] -= 1
                    if self._z[n.node_id-1][s][0] < self._z[n.node_id-1][s][1]:
                        self._true_objective_value -= World.PROFIT_RENTAL / self.num_scenarios
                        self._heuristic_objective_value -= World.PROFIT_RENTAL / self.num_scenarios
                    elif self._w[n.node_id-1][s] > 0 and self._w[n.node_id-1][s] <= n.ideal_state:
                        self._true_objective_value -= World.COST_DEVIATION / self.num_scenarios
                        self._heuristic_objective_value -= World.COST_DEVIATION / self.num_scenarios
            for n in nodes_in:
                for s in range(self.num_scenarios):
                    self._z[n.node_id-1][s][0] += 1
                    self._w[n.node_id-1][s] -= 1
                    if self._z[n.node_id-1][s][0] < self._z[n.node_id-1][s][1]:
                        self._true_objective_value += World.PROFIT_RENTAL / self.num_scenarios
                        self._heuristic_objective_value += World.PROFIT_RENTAL / self.num_scenarios
                    elif self._w[n.node_id-1][s] >= 0 and self._w[n.node_id-1][s] <= n.ideal_state:
                        self._true_objective_value += World.COST_DEVIATION / self.num_scenarios
                        self._heuristic_objective_value += World.COST_DEVIATION / self.num_scenarios
        else:
            for n in nodes_out:
                self._z[n.node_id-1][scenario][0] -= 1
                self._w[n.node_id-1][scenario] += 1
                if self._z[n.node_id-1][scenario][0] < self._z[n.node_id-1][scenario][1]:
                    self._true_objective_value -= World.PROFIT_RENTAL / self.num_scenarios
                    self._heuristic_objective_value -= World.PROFIT_RENTAL / self.num_scenarios
                elif self._w[n.node_id-1][scenario] > 0 and self._w[n.node_id-1][scenario] <= n.ideal_state:
                    self._true_objective_value -= World.COST_DEVIATION / self.num_scenarios
                    self._heuristic_objective_value -= World.COST_DEVIATION / self.num_scenarios

            for n in nodes_in:
                self._z[n.node_id-1][scenario][0] += 1
                self._w[n.node_id-1][scenario] -= 1
                if self._z[n.node_id-1][scenario][0] < self._z[n.node_id-1][scenario][1]:
                    self._true_objective_value += World.PROFIT_RENTAL / self.num_scenarios
                    self._heuristic_objective_value += World.PROFIT_RENTAL / self.num_scenarios
                elif self._w[n.node_id-1][scenario] >= 0 and self._w[n.node_id-1][scenario] <= n.ideal_state:
                    self._true_objective_value += World.COST_DEVIATION / self.num_scenarios
                    self._heuristic_objective_value += World.COST_DEVIATION / self.num_scenarios

    def _update_w(self, nodes_in: [Node] = None, nodes_out: [Node] = None, scenario: int = None):
        """
        :param nodes_in: nodes with cars going into node
        :param nodes_out: nodes with cars leaving node
        :param scenario: if second stage, then the scenario is specified
        """

        if scenario is None:
            for n in nodes_out:
                for s in range(self.num_scenarios):
                    self._w[n.node_id][s] += 1
                    if self._w[n.node_id][s] > 0:
                        self._true_objective_value -= World.COST_DEVIATION / self.num_scenarios
                        self._heuristic_objective_value -= World.COST_DEVIATION / self.num_scenarios
            for n in nodes_in:
                for s in range(self.num_scenarios):
                    self._w[n.node_id][s] -= 1
                    if self._w[n.node_id][s] >= 0:
                        self._true_objective_value += World.COST_DEVIATION / self.num_scenarios
                        self._heuristic_objective_value += World.COST_DEVIATION / self.num_scenarios
        else:
            for n in nodes_out:
                self._w[n.node_id][scenario] += 1
                if self._w[n.node_id][scenario] > 0:
                    self._true_objective_value -= World.COST_DEVIATION / self.num_scenarios
                    self._heuristic_objective_value -= World.COST_DEVIATION / self.num_scenarios
            for n in nodes_in:
                self._w[n.node_id][scenario] -= 1
                if self._w[n.node_id][scenario] >= 0:
                    self._true_objective_value += World.COST_DEVIATION / self.num_scenarios
                    self._heuristic_objective_value += World.COST_DEVIATION / self.num_scenarios

    def _update_relocation_time(self, added_car_moves, removed_car_moves, scenario):

        if scenario is None:
            self._relocation_time += sum(car_move.handling_time for car_move in added_car_moves)
            self._relocation_time -= sum(car_move.handling_time for car_move in removed_car_moves)
            self._true_objective_value += World.COST_RELOCATION * (np.sum(car_move.handling_time for car_move in
                                                                       removed_car_moves) - np.sum(car_move.handling_time
                                                                                                for car_move in
                                                                                                added_car_moves))
            self._heuristic_objective_value += World.COST_RELOCATION * (np.sum(car_move.handling_time for car_move in
                                                                            removed_car_moves) - np.sum(car_move.
                                                                                                     handling_time for
                                                                                                     car_move in
                                                                                                     added_car_moves))
        else:
            self._relocation_time[scenario] += sum(car_move.handling_time for car_move in added_car_moves)
            self._relocation_time[scenario] -= sum(car_move.handling_time for car_move in removed_car_moves)
            self._true_objective_value += World.COST_RELOCATION * ((np.sum(car_move.handling_time for car_move in
                                                                       removed_car_moves) - np.sum(car_move.handling_time
                                                                                                for car_move in
                                                                                                added_car_moves)) / self.num_scenarios)
            self._heuristic_objective_value += World.COST_RELOCATION * ((np.sum(car_move.handling_time for car_move in
                                                                            removed_car_moves) - np.sum(car_move.
                                                                                                     handling_time for
                                                                                                     car_move in
                                                                                                     added_car_moves)) / self.num_scenarios)

    def _update_charging_deviation(self, added_car_moves, removed_car_moves, scenario):
        num_charging_moves_added = sum(1 for cm in added_car_moves if cm.is_charging_move)
        num_charging_moves_removed = sum(1 for cm in removed_car_moves if cm.is_charging_move)

        if scenario is None:
            self._charging_deviation += num_charging_moves_removed - num_charging_moves_added
            self._heuristic_objective_value += 1000 * (num_charging_moves_added - num_charging_moves_removed)
        else:
            self._charging_deviation[scenario] += num_charging_moves_removed - num_charging_moves_added
            self._heuristic_objective_value += 1000 * (
                    num_charging_moves_added - num_charging_moves_removed) / self.num_scenarios


def get_parking_nodes_in_out(added_car_moves: [CarMove], removed_car_moves: [CarMove]) -> ([Node], [Node]):
    """
    :param added_car_moves: car_moves added to solution
    :param removed_car_moves: car_moves removed from solution
    :return: two lists. one list of nodes with cars entering a node, and one list with cars leaving a node.
    """
    start_nodes_added = [car_move.start_node for car_move in added_car_moves if not car_move.is_charging_move]
    end_nodes_added = [car_move.end_node for car_move in added_car_moves if not car_move.is_charging_move]
    start_nodes_removed = [car_move.start_node for car_move in removed_car_moves if not car_move.is_charging_move]
    end_nodes_removed = [car_move.end_node for car_move in removed_car_moves if not car_move.is_charging_move]

    # Create list of nodes where a car is leaving or if it was delivered there in a removed car_move
    nodes_out = []

    # Create list of nodes where a car is entering or if it left the node in a removed car_move
    nodes_in = []

    if start_nodes_added is not None and end_nodes_added is not None:
        nodes_out.extend(start_nodes_added)
        nodes_in.extend(end_nodes_added)
    if start_nodes_removed is not None and end_nodes_removed is not None:
        nodes_out.extend(end_nodes_removed)
        nodes_in.extend(start_nodes_removed)

    return nodes_in, nodes_out


if __name__ == "__main__":
    from pyinstrument import Profiler
    from src.Gurobi.Model.run_model import run_model
    from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
    from new_new_construction_heuristic import ConstructionHeuristic

    filename = "InstanceGenerator/InstanceFiles/2nodes/2-2-1-1_a"
    profiler = Profiler()
    profiler.start()

    ch = ConstructionHeuristic(filename + ".pkl")
    ch.construct()
    print(ch.objective_function.heuristic_objective_value)
    ch.print_solution()
    '''
    nodes_in, nodes_out = get_parking_nodes_in_out([ch.car_moves[0]], [])
    print(ch.objective_function.heuristic_objective_value)
    print("z",ch.objective_function._z)
    print("w",ch.objective_function._w)
    ch.objective_function.update([ch.car_moves[0]])
    print("added CM1")
    print("z",ch.objective_function._z)
    print("w",ch.objective_function._w)
    print(ch.objective_function.heuristic_objective_value)
    print("removed CM1")
    ch.objective_function.update([],[ch.car_moves[0]])
    print("z",ch.objective_function._z)
    print("w",ch.objective_function._w)
    print(ch.objective_function.heuristic_objective_value)
    '''
    #ch.objective_function.update([ch.car_moves[0]])
    #print(ch.objective_function.heuristic_objective_value)



    #ch.construct()

    #ch.construct()

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

    # print(f"Construction heuristic true obj. val {true_obj_val}")
    ch.print_solution()

    print("\n############## Evaluate solution ##############")
    #gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=False)
    #run_model(gi)
