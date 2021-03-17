

def get_obj_value_first_stage(current_solution, car_move, employee, task_nr):
    solution_copy = current_solution.copy()
    solution_copy[employee].insert(task_nr-1, car_move)






def _calculate_z(first_stage_car_moves):
    # z is the number of customer requests served. It must be the lower of the two values
    # available cars and the number of customer requests D_is
    # number of available cars in the beginning of the second stage, y_i

    start_nodes_first_stage = [car_move.start_node for car_move in first_stage_car_moves if
                               isinstance(car_move.end_node, ParkingNode)]
    end_nodes_first_stage = [car_move.end_node for car_move in first_stage_car_moves if
                             isinstance(car_move.end_node, ParkingNode)]

    y = {parking_node.node_id: parking_node.parking_state for parking_node in self.parking_nodes}
    for n in start_nodes_first_stage:
        y[n.node_id] -= 1
    for n in end_nodes_first_stage:
        y[n.node_id] += 1

    node_demands = {parking_node.node_id: {'customer_requests': parking_node.customer_requests,
                                           'car_returns': parking_node.car_returns} for parking_node in
                    self.parking_nodes}

    z = {}

    start_nodes_second_stage = [[car_move.start_node.node_id for car_move in scenarios
                                 if isinstance(car_move.end_node, ParkingNode)] for scenarios in
                                second_stage_car_moves]

    # car_move.start_node should be a list of car moves with len(list) = num_scenarios
    for n in self.parking_nodes:
        second_stage_moves_out = np.array([scenario.count(n.node_id) for scenario in start_nodes_second_stage])
        y[n.node_id] = np.maximum(y[n.node_id], 0)

        z_val = np.minimum(y[n.node_id] + node_demands[n.node_id]['car_returns'] - second_stage_moves_out,
                           node_demands[n.node_id]['customer_requests'])
        z_val = np.maximum(z_val, 0)
        z[n.node_id] = z_val
        if verbose:
            print(f"z[{n.node_id}] {z[n.node_id]}")
    return z

def _calculate_profit_customer_requests(self, z, scenario=None):
    # sum across scenarios for all nodes
    z_sum = sum(v for k, v in z.items())
    if scenario is None:
        # print(f"z_sum {z_sum}")
        z_sum_scenario_average = np.mean(z_sum)
        return World.PROFIT_RENTAL * z_sum_scenario_average
    else:
        # print(f"z_sum[{scenario+1}] {z_sum[scenario]}")
        return World.PROFIT_RENTAL * z_sum[scenario]

def _calculate_costs_relocation(self, car_moves, individual_scenario=False):
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

def _calculate_cost_deviation_ideal_state(self, z, first_stage_car_moves, second_stage_car_moves, scenario=None,
                                          verbose=False):
    # TODO: Some car moves moving into a node (maybe when only happens in one/some scenarios) are not counted in calculation of w variable.

    start_nodes_first_stage = [car_move.start_node for car_move in first_stage_car_moves if
                               isinstance(car_move.end_node, ParkingNode)]
    end_nodes_first_stage = [car_move.end_node for car_move in first_stage_car_moves if
                             isinstance(car_move.end_node, ParkingNode)]

    w = {n.node_id: (n.ideal_state - n.parking_state + z[n.node_id] - n.car_returns) for n in self.parking_nodes}

    for n in start_nodes_first_stage:
        w[n.node_id] += 1
    for n in end_nodes_first_stage:
        w[n.node_id] -= 1

    start_nodes_second_stage = [
        [car_move.start_node.node_id for car_move in scenarios if isinstance(car_move.end_node, ParkingNode)] for
        scenarios in
        second_stage_car_moves]

    end_nodes_second_stage = [
        [car_move.end_node.node_id for car_move in scenarios if isinstance(car_move.end_node, ParkingNode)] for
        scenarios in second_stage_car_moves]

    for n in self.parking_nodes:
        second_stage_moves_out = np.array([cm.count(n.node_id) for cm in start_nodes_second_stage])
        second_stage_moves_in = np.array([cm.count(n.node_id) for cm in end_nodes_second_stage])

        w[n.node_id] += second_stage_moves_out - second_stage_moves_in
        # require w_is >= 0
        w[n.node_id] = np.maximum(w[n.node_id], 0)

        if verbose:
            print(f"\nw[{n.node_id}] {w[n.node_id]}")
            print(f"ideal state {n.ideal_state}")
            print(f"initial_state {n.parking_state}")
            print(f"car returns {n.car_returns}")
            print(f"customer requests {n.customer_requests}")

    w_sum = sum(v for k, v in w.items())

    if scenario is None:
        w_sum_scenario_average = np.mean(w_sum)
        # print(f"w_sum {w_sum}")
        return World.COST_DEVIATION * w_sum_scenario_average
    else:
        # print(f"w_sum[{scenario+1}] {w_sum[scenario]}")
        return World.COST_DEVIATION * w_sum[scenario]
    # only return first scenario for now

def _get_obj_val_of_car_move(first_stage_car_moves):

    z = calculate_z(first_stage_car_moves=first_stage_car_moves)
    profit_customer_requests = self._calculate_profit_customer_requests(z)
    cost_deviation_ideal_state = self._calculate_cost_deviation_ideal_state(z,
                                                                            first_stage_car_moves=first_stage_car_moves,
                                                                            second_stage_car_moves=[[]])
    first_stage_duplicate_for_scenarios = list(np.repeat(first_stage_car_moves, self.num_scenarios))
    cost_relocation = self._calculate_costs_relocation(first_stage_duplicate_for_scenarios)

    return profit_customer_requests - cost_relocation - cost_deviation_ideal_state