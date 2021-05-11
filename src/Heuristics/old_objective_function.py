import numpy as np

from Heuristics.helper_functions_heuristics import copy_numpy_dict
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
        self._heuristic_objective_value = None
        self._true_objective_value = None
        self.update()

    @property
    def heuristic_objective_value(self):
        return np.mean(self._heuristic_objective_value)

    @property
    def true_objective_value(self):
        return np.mean(self._true_objective_value)

    @property
    def heuristic_objective_value_scenarios(self):
        return self._heuristic_objective_value

    @property
    def true_objective_value_scenarios(self):
        return self._true_objective_value

    def _initialize_z(self):
        z = {node.node_id: (node.parking_state + node.car_returns) for node in self.parking_nodes}
        return z

    def _initialize_w(self):
        #w = {node.node_id: (node.ideal_state - node.parking_state - node.car_returns - self.z[node.node_id]) for node in self.parking_nodes}
        w = {node.node_id: (node.ideal_state - node.parking_state - node.car_returns) for node in self.parking_nodes}
        #w = {node.node_id: (node.ideal_state) for node in self.parking_nodes}
        return w

    def _initialize_charging_deviation(self):
        num_cars_in_need_of_charging = sum(pnode.charging_state for pnode in self.parking_nodes)
        return np.array([num_cars_in_need_of_charging for _ in range(self.num_scenarios)])

    def _get_constrained_z(self, z):
        new_z = {}
        for node in self.parking_nodes:
            z_val = np.minimum(z[node.node_id], node.customer_requests)
            z_val = np.maximum(z_val, 0)
            new_z[node.node_id] = z_val
        return new_z

    def _get_constrained_w(self, w):
        new_w = {}
        for n in self.parking_nodes:
            new_w[n.node_id] = np.maximum(w[n.node_id], 0)
        return new_w

    def _get_parking_nodes_in_out(self, added_car_moves: [CarMove], removed_car_moves: [CarMove]) -> ([Node], [Node]):
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

    def evaluate(self, added_car_moves: [CarMove] = None, removed_car_moves: [CarMove] = None, scenario: int = None):
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

        nodes_in, nodes_out = self._get_parking_nodes_in_out(added_car_moves, removed_car_moves)
        z = self._update_z(nodes_in, nodes_out, scenario)
        w = self._update_w(nodes_in, nodes_out, scenario, z=z)
        relocation_time = self._update_relocation_time(added_car_moves, removed_car_moves, scenario)
        charging_deviation = self._update_charging_deviation(added_car_moves, removed_car_moves, scenario)
        heuristic_obj_val, _ = self._calculate_obj_val(z, w, relocation_time, charging_deviation)

        if scenario is not None:
            return heuristic_obj_val[scenario]
        else:
            return np.mean(heuristic_obj_val)

    def update(self, added_car_moves: [CarMove] = None, removed_car_moves: [CarMove] = None, scenario: int = None):

        if added_car_moves is None:
            added_car_moves = []
        if added_car_moves is None:
            added_car_moves = []
        if removed_car_moves is None:
            removed_car_moves = []

        nodes_in, nodes_out = self._get_parking_nodes_in_out(added_car_moves, removed_car_moves)
        z = self._update_z(nodes_in, nodes_out, scenario, update=True)
        w = self._update_w(nodes_in, nodes_out, scenario, z=z, update=True)
        relocation_time = self._update_relocation_time(added_car_moves, removed_car_moves, scenario, update=True)
        charging_deviation = self._update_charging_deviation(added_car_moves, removed_car_moves, scenario, update=True)
        self._heuristic_objective_value, self._true_objective_value = self._calculate_obj_val(z, w, relocation_time, charging_deviation)


    def _update_z(self, nodes_in: [Node]=None, nodes_out: [Node]=None, scenario: int=None, update=False):
        """
        :param nodes_in: nodes with cars going into node
        :param nodes_out: nodes with cars leaving node
        :param scenario: if second stage, then the scenario is specified
        :param evaluate: if you want to return a value for evaluating a solution
        :return: only if evaluate=True, else z is changed inplace. dictionary with node ids as keys and numpy array with dimension equal to the number of
                 scenarios as value e.g: {node_id: np.array([0, 2, 1])}
        """

        z = copy_numpy_dict(self._z)

        if scenario is None:
            for n in nodes_out:
                z[n.node_id] -= 1
            for n in nodes_in:
                z[n.node_id] += 1
        else:
            for n in nodes_out:
                z[n.node_id][scenario] -= 1

        if update:
            self._z = z

        return self._get_constrained_z(z)



    def _update_w(self, nodes_in: [Node]=None, nodes_out: [Node]=None, scenario: int=None, z=None, update=False):
        """
        :param nodes_in: nodes with cars going into node
        :param nodes_out: nodes with cars leaving node
        :param scenario: if second stage, then the scenario is specified
        :param evaluate: if you want to return a value for evaluating a solution
        :param z: if you want to evaluate a solution, you must provide a z value, else the instance's z value is used
        :return: only if evaluate=True, else w is changed inplace.
        """
        w = copy_numpy_dict(self._w)

        if scenario is None:
            for n in nodes_out:
                w[n.node_id] += 1
            for n in nodes_in:
                w[n.node_id] -= 1
        else:
            for n in nodes_out:
                w[n.node_id][scenario] += 1
            for n in nodes_in:
                w[n.node_id][scenario] -= 1

        if update:
            self._w = copy_numpy_dict(w)

        for node in self.parking_nodes:
            w[node.node_id] += z[node.node_id]

        return self._get_constrained_w(w)

    def _update_relocation_time(self, added_car_moves, removed_car_moves, scenario, update=False):

        #relocation_time = np.copy(self._relocation_time)
        relocation_time = np.array(self._relocation_time) #np.array is faster than np.copy and also makes a copy

        if scenario is None:
            relocation_time += sum(car_move.handling_time for car_move in added_car_moves)
            relocation_time -= sum(car_move.handling_time for car_move in removed_car_moves)
        else:
            relocation_time[scenario] += sum(car_move.handling_time for car_move in added_car_moves)
            relocation_time[scenario] -= sum(car_move.handling_time for car_move in removed_car_moves)

        if update:
            self._relocation_time = relocation_time
        return relocation_time

    def _update_charging_deviation(self, added_car_moves, removed_car_moves, scenario, update=False):
        num_charging_moves_added = sum(1 for cm in added_car_moves if cm.is_charging_move)
        num_charging_moves_removed = sum(1 for cm in removed_car_moves if cm.is_charging_move)
        #charging_deviation = np.copy(self._charging_deviation)
        charging_deviation = np.array(self._charging_deviation)



        if scenario is None:
            charging_deviation += num_charging_moves_removed - num_charging_moves_added
        else:
            charging_deviation[scenario] += num_charging_moves_removed - num_charging_moves_added

        if update:
            self._charging_deviation = charging_deviation

        return charging_deviation

    def _update_inter_move_travel_time(self):
        pass

    def _calculate_obj_val(self, z, w, relocation_time, charging_deviation):
        profit_customer_requests = self._calculate_profit_customer_requests(z)
        cost_deviation_ideal_state = self._calculate_cost_deviation_ideal_state(w)
        cost_relocation = self._calculate_costs_relocation(relocation_time)
        cost_charging_deviation = self._calculate_cost_deviation_charging_moves(charging_deviation)

        '''
        for n in self.parking_nodes:
            print(f"\nw[{n.node_id}] {w[n.node_id]}")
            print(f"ideal state {n.ideal_state}")
            print(f"initial_state {n.parking_state}")
            print(f"car returns {n.car_returns}")
            print(f"customer requests {n.customer_requests}")
        '''
        '''
        print("\nprofit_customer_requests: ", profit_customer_requests)
        print("cost_relocation: ", cost_relocation)
        print("cost_deviation_ideal_state: ", cost_deviation_ideal_state)
        print("cost_deviation_charging_moves: ", cost_charging_deviation)
        '''

        true_obj_val = profit_customer_requests - cost_deviation_ideal_state - cost_relocation
        heuristic_obj_val = true_obj_val - cost_charging_deviation

        return heuristic_obj_val, true_obj_val


    def _calculate_profit_customer_requests(self, z: {int: np.array([int])}) -> [float]:
        """
        :param z: dictionary with node_id as keys and a numpy array for each scenario as value,
        {node_id: np.array([0, 2, 1])}
        :return: float with profit. if scenarios is given, then avg. over scenarios is returned, else value for given
        scenario is returned
        """
        # sum across scenarios for all nodes
        z_sum = sum(v for k, v in z.items())
        return World.PROFIT_RENTAL*z_sum



    def _calculate_cost_deviation_ideal_state(self, w) -> [float]:
        """
        :param w:
        :return:
        """
        w_sum = sum(v for k, v in w.items())
        return World.COST_DEVIATION*w_sum


    def _calculate_costs_relocation(self, relocation_time) -> [float]:
        """
        :param relocation_time:
        :return:
        """
        return World.COST_RELOCATION * relocation_time

    def _calculate_cost_deviation_charging_moves(self, charging_deviation, scenario=None):
        """
        :param charging_deviation:
´        :return:
        """
        return 1000*charging_deviation


if __name__ == "__main__":
    from pyinstrument import Profiler
    from src.Gurobi.Model.run_model import run_model
    from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
    from old_construction_heuristic import ConstructionHeuristic

    filename = "InstanceGenerator/InstanceFiles/40nodes/40-10-1-1_a"
    ch = ConstructionHeuristic(filename + ".pkl")
    profiler = Profiler()
    profiler.start()

    ch.construct(verbose=True)

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
    true_obj_val, best_obj_val = ch.get_obj_val(both=True)
    # print(f"Construction heuristic true obj. val {true_obj_val}")
    ch.print_solution()

    print("\n############## Evaluate solution ##############")
    gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=False)
    run_model(gi)