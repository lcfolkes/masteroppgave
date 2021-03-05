import os
os.chdir('../InstanceGenerator')
from InstanceGenerator.instance_components import ParkingNode, Employee, ChargingNode
from InstanceGenerator.world import World
from src.HelperFiles.helper_functions import load_object_from_file
import numpy as np


class ConstructionHeuristic:
	#instance_file = "InstanceFiles/6nodes/6-3-1-1_a.pkl"
	# filename = "InstanceFiles/6nodes/6-3-1-1_b.yaml"

	def __init__(self, instance_file):

		self.world_instance = load_object_from_file(instance_file)
		self.beta = []
		self.employees = self.world_instance.employees
		self.parking_nodes = self.world_instance.parking_nodes
		self.gamma_k = {k.employee_id: [] for k in self.employees}
		self.cars = self.world_instance.cars
		self.car_moves = []#self.world_instance.car_moves
		self.charging_moves = []
		self.parking_moves = []
		self.num_scenarios = self.world_instance.num_scenarios

		for car in self.world_instance.cars:
			for car_move in car.car_moves:
				self.car_moves.append(car_move)
				if car.needs_charging:
					self.charging_moves.append(car_move)
				else:
					self.parking_moves.append(car_move)



	def _calculate_z(self, first_stage_car_moves, second_stage_car_moves):
		# z is the number of customer requests served. It must be the lower of the two values
		# available cars and the number of customer requests D_is
		# number of available cars in the beginning of the second stage, y_i

		start_nodes_first_stage = [car_move.start_node for car_move in first_stage_car_moves]
		end_nodes_first_stage = [car_move.end_node for car_move in first_stage_car_moves if
								 isinstance(car_move.end_node, ParkingNode)]

		y = {parking_node.node_id: parking_node.parking_state for parking_node in self.parking_nodes}
		for n in start_nodes_first_stage:
			y[n.node_id] -= 1
		for n in end_nodes_first_stage:
			y[n.node_id] += 1

		node_demands = {parking_node.node_id: {'customer_requests': parking_node.customer_requests,
											   'car_returns': parking_node.car_returns} for parking_node in self.parking_nodes}

		# TODO: when calculating z, we must subtract x_krms for the second stage (parking moves starting in a node)

		z = {}
		start_nodes_second_stage = [car_move.start_node for car_move in second_stage_car_moves]
		#car_move.start_node should be a list of car moves with len(list) = num_scenarios
		for n in self.parking_nodes:
			z_val = np.minimum(y[n.node_id] + node_demands[n.node_id]['car_returns'] - start_nodes_second_stage.count(n),
							   node_demands[n.node_id]['customer_requests'])
			z[n.node_id] = z_val

		return z


	def _calculate_profit_customer_requests(self, z):
		# print(z)
		# sum across scenarios for all nodes
		z_sum = sum(v for k, v in z.items())
		z_sum_scenario_average = np.mean(z_sum)
		return World.PROFIT_RENTAL * z_sum_scenario_average


	def _calculate_costs_relocation(self, car_moves):
		# Sum of all travel times across all car moves
		# TODO: must also handle second stage moves (scenario dependent)
		total_travel_time = sum(car_move.handling_time for car_move in car_moves)
		return World.COST_RELOCATION * total_travel_time


	def _calculate_cost_deviation_ideal_state(self, car_moves, z):
		# TODO: Must handle car_moves for each scenario in second stage
		start_nodes = [car_move.start_node for car_move in car_moves]
		end_nodes = [car_move.end_node for car_move in car_moves if
								 isinstance(car_move.end_node, ParkingNode)]
		w = {n.node_id: (n.ideal_state - n.parking_state + z[n.node_id] - n.car_returns) for n in self.parking_nodes}

		for n in start_nodes:
			w[n.node_id] += 1
		for n in end_nodes:
			w[n.node_id] -= 1

		# print(w)
		w_sum = sum(v for k, v in w.items())
		w_sum_scenario_average = np.mean(w_sum)

		# only return first scenario for now
		return World.COST_DEVIATION * w_sum_scenario_average

	def _calculate_objective_function(self, employees):
		first_stage_car_moves = []
		second_stage_car_moves = []
		for employee in employees:
			counter = 0
			for car_move in employee.car_moves:
				if counter < self.world_instance.first_stage_tasks:
					first_stage_car_moves.append(car_move)
				else:
					second_stage_car_moves.append(car_move)
				counter += 1

		all_car_moves = first_stage_car_moves + second_stage_car_moves
		z = self._calculate_z(first_stage_car_moves, second_stage_car_moves)
		profit_customer_requests = self._calculate_profit_customer_requests(z)
		cost_relocation = self._calculate_costs_relocation(all_car_moves)
		cost_deviation_ideal_state = self._calculate_cost_deviation_ideal_state(all_car_moves, z)

		#print(profit_customer_requests - cost_relocation - cost_deviation_ideal_state)
		return profit_customer_requests - cost_relocation - cost_deviation_ideal_state

	def get_obj_val_of_car_move(self, car_move, first_stage, scenario=None):
		assert(first_stage or (not first_stage and scenario is not None))
		if first_stage:
			z = self._calculate_z(first_stage_car_moves=[car_move], second_stage_car_moves=[])
		else:
			car_moves = []*self.num_scenarios
			car_moves.insert(scenario-1, [car_move])
			z = self._calculate_z(first_stage_car_moves=[], second_stage_car_moves=car_moves)
			# E.g. if car_move is performed in scenario 3
			#z = self._calculate_z(first_stage_car_moves=[], second_stage_car_moves=[[],[],[car_move]])


		profit_customer_requests = self._calculate_profit_customer_requests(z)
		cost_relocation = self._calculate_costs_relocation([car_move])
		cost_deviation_ideal_state = self._calculate_cost_deviation_ideal_state([car_move], z)

		#print(profit_customer_requests - cost_relocation - cost_deviation_ideal_state)
		return profit_customer_requests - cost_relocation - cost_deviation_ideal_state


	def add_car_moves_to_employees(self):
		available_employees = True
		prioritize_charging = True
		first_stage = True
		charging_moves = self.charging_moves #[car_move for car_move in self.car_moves if isinstance(car_move.end_node, ChargingNode)]
		parking_moves = self.parking_moves #[car_move for car_move in self.car_moves if isinstance(car_move.end_node, ParkingNode)]
		num_scenarios = self.num_scenarios

		#first_stage_car_moves = []
		#second_stage_car_moves = []

		while available_employees:
			best_car_move = None
			best_obj_val = 0
			best_car_move_second_stage = [None]*num_scenarios
			best_obj_val_second_stage = [0] * num_scenarios

			# check if charging_moves_list is not empty
			if charging_moves:
				prioritize_charging = True
				car_moves = charging_moves
			else:
				prioritize_charging = False
				car_moves = parking_moves

			for car_move in car_moves:
				if first_stage:
					obj_val = self.get_obj_val_of_car_move(car_move=car_move, first_stage=first_stage)
					if obj_val > best_obj_val:
						best_obj_val = obj_val
						best_car_move = car_move
				else:
					obj_val = [0] * num_scenarios
					# zero indexed scenario
					for s in range(num_scenarios):
						obj_val.append(self.get_obj_val_of_car_move(car_move=car_move, first_stage=first_stage, scenario=s))

						if obj_val[s] > best_obj_val_second_stage[s]:
							best_obj_val_second_stage[s] = obj_val[s]
							best_car_move_second_stage[s] = car_move

			best_employee = None
			best_travel_time_to_car_move = 100
			best_employee_second_stage = [None]*num_scenarios
			best_travel_time_to_car_move_second_stage = [100]*num_scenarios

			end_node = best_car_move.start_node
			for employee in self.employees:
				task_num = len(employee.car_moves)
				# if first stage and the number of completed task for employee is below the number of tasks in first stage,
				# or if second stage and the number of completed tasks are the same or larger than the number of tasks in first stage
				if first_stage == (task_num < self.world_instance.first_stage_tasks):
					if first_stage:
						legal_move = self.world_instance.check_legal_move(car_move=best_car_move, employee=employee)
						if legal_move:
							start_node = employee.current_node
							travel_time_to_car_move = self.world_instance.get_employee_travel_time_to_node(start_node, end_node)
							if travel_time_to_car_move < best_travel_time_to_car_move:
								best_travel_time_to_car_move = travel_time_to_car_move
								best_employee = employee
					else:
						for s in range(num_scenarios):
							legal_move = self.world_instance.check_legal_move(car_move=best_car_move, employee=employee)
							if legal_move:
								start_node = employee.current_node_second_stage[s]
								travel_time_to_car_move = self.world_instance.get_employee_travel_time_to_node(start_node, end_node)
								if travel_time_to_car_move < best_travel_time_to_car_move_second_stage[s]:
									best_travel_time_to_car_move_second_stage[s] = travel_time_to_car_move
									best_employee_second_stage[s] = employee
			if first_stage:
				if best_employee is not None:
					print('\nEmployee id', best_employee.employee_id)
					print('Employee node before', best_employee.current_node.node_id)
					print('Employee time before', best_employee.current_time)
					print('Travel time to start node', best_travel_time_to_car_move)
					print(best_car_move.to_string())
					self.world_instance.add_car_move_to_employee(best_car_move, best_employee)
					print('Employee node after', best_employee.current_node.node_id)
					print('Employee time after', best_employee.current_time)
					if prioritize_charging:
						charging_moves = self.remove_car_move(best_car_move, car_moves) # should remove car move and other car-moves with the same car
					else:
						parking_moves = self.remove_car_move(best_car_move, car_moves) # should remove car move and other car-moves with the same car
					first_stage = False
					for employee in self.employees:
						task_num = len(employee.car_moves)
						if task_num < self.world_instance.first_stage_tasks:
							first_stage = True
				else:
					available_employees = False
			# Second stage
			else:
				# If any employee is not note, continue
				if not all(e is None for e in best_employee_second_stage):
					for s in range(num_scenarios):
						if best_employee_second_stage[s] is not None:
							print('\nEmployee id', best_employee.employee_id)
							print('\nScenario', s+1)
							print('Employee node before', best_employee.current_node_second_stage[s].node_id)
							print('Employee time before', best_employee.current_time_second_stage)
							print('Travel time to start node', best_travel_time_to_car_move_second_stage[s])
							self.world_instance.add_car_move_to_employee(best_car_move, best_employee_second_stage[s], s)
							#if prioritize_charging:
							#	charging_moves = self.remove_car_move(best_car_move,
							#										  car_moves)  # should remove car move and other car-moves with the same car
							#else:
							#	parking_moves = self.remove_car_move(best_car_move,
							#										 car_moves)  # should remove car move and other car-moves wit
				else:
					available_employees = False



	def remove_car_move(self, chosen_car_move, car_moves):
		car = chosen_car_move.car.car_id
		# return list of car moves that are not associated with the car of the chosen car move
		return [cm for cm in car_moves if cm.car.car_id != car]


		# print(f"obj_val: {obj_val}")
	def find_nearest_car_move(self, employee, car_moves_copy):
		start_node = employee.current_node
		nearest_car_move = None
		best_travel_time = 100 #some high number
		for car_move in car_moves_copy:
			end_node = car_move.start_node
			travel_time = self.world_instance.get_employee_travel_time_to_node(start_node, end_node)
			legal_move = self.world_instance.check_legal_move(employee, car_move)
			if travel_time < best_travel_time and legal_move:
				best_travel_time = travel_time
				nearest_car_move = car_move

		return nearest_car_move

	def print_solution(self):
		first_stage_car_moves = {e.employee_id: [] for e in self.employees}
		second_stage_car_moves = {e.employee_id: [] for e in self.employees}
		for employee in self.employees:
			counter = 0
			for car_move in employee.car_moves:
				if counter < self.world_instance.first_stage_tasks:
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


ch = ConstructionHeuristic("InstanceFiles/6nodes/6-3-1-1_b.pkl")
#print("obj_val", obj_val)
#ch.add_car_moves_to_employees3()
ch.add_car_moves_to_employees()

ch.print_solution()





''' EXAMPLE OUTPUT
	-------------- First stage routes --------------
	  Employee Task   Route  Travel Time to Task  Start time  Relocation Time  End time
	0        2    1  (2, 6)                  7.7        12.7              7.6      20.3
	
	-------------- Second stage routes --------------
	  Employee Task Scenario   Route  Travel Time to Task  Start time  Relocation Time  End time
	0        2    2        3  (4, 1)                 19.8        40.1             14.1      54.2
'''
