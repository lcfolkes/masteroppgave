import os

os.chdir('../InstanceGenerator')
from InstanceGenerator.instance_components import ParkingNode
from InstanceGenerator.world import World
from src.HelperFiles.helper_functions import load_object_from_file
import random
import numpy as np

# filename = "InstanceFiles/6nodes/6-3-1-1_b.yaml"
dill_filename = "InstanceFiles/6nodes/6-3-1-1_a.pkl"

world_instance = load_object_from_file(dill_filename)
beta = []
employees = world_instance.employees.copy()
parking_nodes = world_instance.parking_nodes.copy()
gamma_k = {k.employee_id: [] for k in employees}

for car in world_instance.cars:
	for car_move in car.car_moves:
		beta.append(car_move)


def print_solution(employees):
	first_stage_car_moves = {e.employee_id: [] for e in employees}
	second_stage_car_moves = {e.employee_id: [] for e in employees}
	for employee in employees:
		counter = 0
		for car_move in employee.car_moves:
			if counter < world_instance.first_stage_tasks:
				first_stage_car_moves[employee.employee_id].append(car_move)
			else:
				second_stage_car_moves[employee.employee_id].append(car_move)
			counter += 1

	print("-------------- First stage routes --------------")
	for employee_id, car_moves in first_stage_car_moves.items():
		for car_move in car_moves:
			print(f"employee: {employee_id}, " + car_move.to_string())

	print("-------------- Second stage routes --------------")
	for employee_id, car_moves in second_stage_car_moves.items():
		for car_move in car_moves:
			print(f"employee: {employee_id}, " + car_move.to_string())


def _calculate_z(first_stage_car_moves, second_stage_car_moves):
	# z is the number of customer requests served. It must be the lower of the two values
	# available cars and the number of customer requests D_is
	# number of available cars in the beginning of the second stage, y_i

	start_nodes_first_stage = [car_move.start_node for car_move in first_stage_car_moves]
	end_nodes_first_stage = [car_move.end_node for car_move in first_stage_car_moves if
							 isinstance(car_move.end_node, ParkingNode)]

	y = {parking_node.node_id: parking_node.parking_state for parking_node in parking_nodes}
	for n in start_nodes_first_stage:
		y[n.node_id] -= 1
	for n in end_nodes_first_stage:
		y[n.node_id] += 1

	node_demands = {parking_node.node_id: {'customer_requests': parking_node.customer_requests,
										   'car_returns': parking_node.car_returns} for parking_node in parking_nodes}

	# TODO: when calculating z, we must subtract x_krms for the second stage

	z = {}
	start_nodes_second_stage = [car_move.start_node for car_move in second_stage_car_moves]
	for n in parking_nodes:
		z_val = np.minimum(y[n.node_id] + node_demands[n.node_id]['car_returns'] - start_nodes_second_stage.count(n),
						   node_demands[n.node_id]['customer_requests'])
		z[n.node_id] = z_val

	return z


def _calculate_profit_customer_requests(z):
	#print(z)
	z_sum = sum(v for k, v in z.items())
	# only return first scenario for now
	return World.PROFIT_RENTAL * z_sum[0]


def _calculate_costs_relocation(car_moves):
	# Sum of all travel times across all car moves
	total_travel_time = sum(car_move.handling_time for car_move in car_moves)
	return World.COST_RELOCATION * total_travel_time


def _calculate_cost_deviation_ideal_state(car_moves, z):
	start_nodes = [car_move.start_node for car_move in car_moves]
	end_nodes = [car_move.end_node for car_move in car_moves if
							 isinstance(car_move.end_node, ParkingNode)]
	w = {n.node_id: (n.ideal_state - n.parking_state + z[n.node_id] - n.car_returns) for n in parking_nodes}

	for n in start_nodes:
		w[n.node_id] += 1
	for n in end_nodes:
		w[n.node_id] -= 1

	#print(w)
	w_sum = sum(v for k, v in w.items())
	# only return first scenario for now
	return World.COST_DEVIATION * w_sum[0]


def _calculate_objective_function(employees):
	first_stage_car_moves = []
	second_stage_car_moves = []
	for employee in employees:
		counter = 0
		for car_move in employee.car_moves:
			if counter < world_instance.first_stage_tasks:
				first_stage_car_moves.append(car_move)
			else:
				second_stage_car_moves.append(car_move)
			counter += 1

	all_car_moves = first_stage_car_moves + second_stage_car_moves
	z = _calculate_z(first_stage_car_moves, second_stage_car_moves)
	profit_customer_requests = _calculate_profit_customer_requests(z)
	cost_relocation = _calculate_costs_relocation(all_car_moves)
	cost_deviation_ideal_state = _calculate_cost_deviation_ideal_state(all_car_moves, z)

	#print(profit_customer_requests - cost_relocation - cost_deviation_ideal_state)
	return profit_customer_requests - cost_relocation - cost_deviation_ideal_state


def main():
	cars = world_instance.cars.copy()
	obj_val = 0
	while cars:
		current_solution = []
		charging_prioritized = False
		for employee in employees:
			best_car_move = None
			car_moved = None
			new_obj_val = 0
			for car in cars:
				if car.needs_charging or charging_prioritized:
					for car_move in car.car_moves:
						if world_instance.check_legal_move(car_move, employee):
							# objective_function(car_move):
							world_instance.add_car_move_to_employee(car_move, employee)
							temp_val = _calculate_objective_function(employees)
							#print(temp_val)
							world_instance.remove_car_move_from_employee(car_move, employee)
							#print(f"temp_val: {temp_val}, new_obj_val: {new_obj_val}, obj_val: {obj_val}")
							if temp_val > new_obj_val:
								new_obj_val = temp_val
								best_car_move = car_move
								car_moved = car_move.car

			if best_car_move is None:
				if charging_prioritized:
					cars = []
				else:
					#print("Charging prioritized")
					charging_prioritized = True
			else:
				#print('\nEmployee id', employee.employee_id)
				#print('Employee node before', employee.current_node.node_id)
				#print('Employee time before', employee.current_time)
				#print(best_car_move.to_string())
				world_instance.add_car_move_to_employee(best_car_move, employee)
				#print('Employee node after', employee.current_node.node_id)
				#print('Employee time after', employee.current_time)
				gamma_k[employee.employee_id].append(best_car_move)
				cars.remove(car_moved)
				beta.remove(best_car_move)
				obj_val = new_obj_val
				#print(f"obj_val: {obj_val}")


	print("obj_val", obj_val)
	print_solution(employees)

main()


''' EXAMPLE OUTPUT
-------------- First stage routes --------------
  Employee Task   Route  Travel Time to Task  Start time  Relocation Time  End time
0        2    1  (2, 6)                  7.7        12.7              7.6      20.3

-------------- Second stage routes --------------
  Employee Task Scenario   Route  Travel Time to Task  Start time  Relocation Time  End time
0        2    2        3  (4, 1)                 19.8        40.1             14.1      54.2
'''
