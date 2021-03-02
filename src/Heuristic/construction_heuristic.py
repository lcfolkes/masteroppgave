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
cars = world_instance.cars.copy()
employees = world_instance.employees.copy()
parking_nodes = world_instance.parking_nodes.copy()
gamma_k = {k.employee_id: [] for k in employees}

for car in world_instance.cars:
	for car_move in car.car_moves:
		beta.append(car_move)

# calculate_solution()

def objective_function_val(car_move, employee):
	return random.randint(0, 10)


obj_val = 0
# TODO: start with charging moves. perhaps make list of charging moves

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
						temp_val = objective_function_val(car_move, employee)
						if temp_val > new_obj_val:
							new_obj_val = temp_val
							best_car_move = car_move
							car_moved = car_move.car
		if best_car_move is None:
			if charging_prioritized:
				cars = []
			else:
				print("Charging prioritized")
				charging_prioritized = True
		else:
			print('\nEmployee id', employee.employee_id)
			print('Employee node before', employee.current_node.node_id)
			print('Employee time before', employee.current_time)
			print(best_car_move.to_string())
			world_instance.add_car_move_to_employee(best_car_move, employee)
			print('Employee node after', employee.current_node.node_id)
			print('Employee time after', employee.current_time)
			gamma_k[employee.employee_id].append(best_car_move)
			cars.remove(car_moved)
			beta.remove(best_car_move)
			obj_val += new_obj_val


def calculate_solution():
	for employee in employees:
		for car_move in employee.car_moves:
			print(car_move.to_string())

		print('Employee node', employee.current_node)
		print('Employee time', employee.current_time)

def _calculate_profit_customer_requests(car_moves):
	# z is the number of customer requests served. It must be the lower of the two values
	# available cars and the number of customer requests D_is
	# number of available cars in the beginning of the second stage, y_i

	start_nodes = [car_move.start_node for car_move in car_moves]
	end_nodes = [car_move.end_node for car_move in car_moves if isinstance(car_move.end_node, ParkingNode)]

	y = {parking_node.node_id: parking_node.parking_state for parking_node in parking_nodes}
	for n in start_nodes:
		y[n.node_id] -= 1
	for n in end_nodes:
		y[n.node_id] += 1

	node_demands = {parking_node.node_id: {'customer_requests': parking_node.customer_requests, 'car_returns': parking_node.car_returns} for parking_node in parking_nodes}

	#TODO: when calculating z, we must subtract x_krms for the second stage
	z = {}
	for n in parking_nodes:
		z_val = np.minimum(node_demands[n.node_id]['car_returns'] + y[n.node_id], node_demands[n.node_id]['customer_requests'])
		z[n.node_id] = z_val

	z_sum = sum(v for k, v in z.items())
	# only return first scenario for now
	return World.PROFIT_RENTAL * z_sum[0]

def _calculate_costs_relocation(car_moves):
	# Sum of all travel times across all car moves
	total_travel_time = sum(car_move.handling_time for car_move in car_moves)
	return World.COST_RELOCATION * total_travel_time

def _calculate_cost_deviation_ideal_state(car_moves):
	return World.COST_DEVIATION

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

	profit_customer_requests = _calculate_profit_customer_requests(car_moves)
	cost_relocation = _calculate_costs_relocation(car_moves)
	cost_deviation_ideal_state = _calculate_cost_deviation_ideal_state(car_moves)

	return profit_customer_requests - cost_relocation - cost_deviation_ideal_state

_calculate_objective_function(employees)

'''
def objective_function():
	### OBJECTIVE FUNCTION ###
	# profit from customer requests
	profit_customer_requests = gp.quicksum(
		(PROFIT_RENTAL) * z[(i, s)] for i in self.PARKING_NODES for s in self.SCENARIOS)

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
			
'''

''' EXAMPLE OUTPUT
-------------- First stage routes --------------
  Employee Task   Route  Travel Time to Task  Start time  Relocation Time  End time
0        2    1  (2, 6)                  7.7        12.7              7.6      20.3

-------------- Second stage routes --------------
  Employee Task Scenario   Route  Travel Time to Task  Start time  Relocation Time  End time
0        2    2        3  (4, 1)                 19.8        40.1             14.1      54.2
'''
