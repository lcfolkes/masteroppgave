import os

os.chdir('../InstanceGenerator')
from InstanceGenerator.instance_components import ParkingNode, Employee, ChargingNode
from InstanceGenerator.world import World
from src.HelperFiles.helper_functions import load_object_from_file
import numpy as np


class ConstructionHeuristic:
	# instance_file = "InstanceFiles/6nodes/6-3-1-1_a.pkl"
	# filename = "InstanceFiles/6nodes/6-3-1-1_b.yaml"

	def __init__(self, instance_file):

		self.world_instance = load_object_from_file(instance_file)
		self.beta = []
		self.employees = self.world_instance.employees
		self.parking_nodes = self.world_instance.parking_nodes
		self.gamma_k = {k.employee_id: [] for k in self.employees}
		self.cars = self.world_instance.cars
		self.car_moves = []  # self.world_instance.car_moves
		self.charging_moves = []
		self.parking_moves = []

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
											   'car_returns': parking_node.car_returns} for parking_node in
						self.parking_nodes}

		# TODO: when calculating z, we must subtract x_krms for the second stage

		z = {}
		start_nodes_second_stage = [car_move.start_node for car_move in second_stage_car_moves]
		for n in self.parking_nodes:
			z_val = np.minimum(
				y[n.node_id] + node_demands[n.node_id]['car_returns'] - start_nodes_second_stage.count(n),
				node_demands[n.node_id]['customer_requests'])
			z[n.node_id] = z_val

		return z

	def _calculate_profit_customer_requests(self, z):
		# print(z)
		z_sum = sum(v for k, v in z.items())
		# only return first scenario for now
		return World.PROFIT_RENTAL * z_sum[0]

	def _calculate_costs_relocation(self, car_moves):
		# Sum of all travel times across all car moves
		total_travel_time = sum(car_move.handling_time for car_move in car_moves)
		return World.COST_RELOCATION * total_travel_time

	def _calculate_cost_deviation_ideal_state(self, car_moves, z):
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
		# only return first scenario for now
		return World.COST_DEVIATION * w_sum[0]

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

		# print(profit_customer_requests - cost_relocation - cost_deviation_ideal_state)
		return profit_customer_requests - cost_relocation - cost_deviation_ideal_state

	def get_obj_val_of_car_move(self, car_move, first_stage):
		if first_stage:
			z = self._calculate_z(first_stage_car_moves=[car_move], second_stage_car_moves=[])
		else:
			z = self._calculate_z(first_stage_car_moves=[], second_stage_car_moves=[car_move])
		profit_customer_requests = self._calculate_profit_customer_requests(z)
		cost_relocation = self._calculate_costs_relocation([car_move])
		cost_deviation_ideal_state = self._calculate_cost_deviation_ideal_state([car_move], z)

		# print(profit_customer_requests - cost_relocation - cost_deviation_ideal_state)
		return profit_customer_requests - cost_relocation - cost_deviation_ideal_state

	def add_car_moves_to_employees3(self):
		available_employees = True
		prioritize_charging = True
		first_stage = True
		charging_moves = self.charging_moves  # [car_move for car_move in self.car_moves if isinstance(car_move.end_node, ChargingNode)]
		parking_moves = self.parking_moves  # [car_move for car_move in self.car_moves if isinstance(car_move.end_node, ParkingNode)]

		# first_stage_car_moves = []
		# second_stage_car_moves = []

		while available_employees:
			best_car_move = None
			best_obj_val = 0

			# check if charging_moves_list is not empty
			if charging_moves:
				prioritize_charging = True
				car_moves = charging_moves
			else:
				prioritize_charging = False
				car_moves = parking_moves

			for car_move in car_moves:
				obj_val = self.get_obj_val_of_car_move(car_move, first_stage)
				if obj_val > best_obj_val:
					best_obj_val = obj_val
					best_car_move = car_move

			best_employee = None
			best_travel_time_to_car_move = 100

			end_node = best_car_move.start_node
			for employee in self.employees:
				task_num = len(employee.car_moves)
				# if first stage and the number of completed task for employee is below the number of tasks in first stage,
				# or if second stage and the number of completed tasks are the same or larger than the number of tasks in first stage
				if first_stage == (task_num < self.world_instance.first_stage_tasks):
					legal_move = self.world_instance.check_legal_move(car_move=best_car_move, employee=employee)
					if legal_move:
						start_node = employee.current_node
						travel_time_to_car_move = self.world_instance.get_employee_travel_time_to_node(start_node,
																									   end_node)
						if travel_time_to_car_move < best_travel_time_to_car_move:
							best_travel_time_to_car_move = travel_time_to_car_move
							best_employee = employee

			if best_employee is not None:
				self.world_instance.add_car_move_to_employee(best_car_move, best_employee)
				if prioritize_charging:
					charging_moves = self.remove_car_move(best_car_move,
														  car_moves)  # should remove car move and other car-moves with the same car
				else:
					parking_moves = self.remove_car_move(best_car_move,
														 car_moves)  # should remove car move and other car-moves with the same car

				first_stage = False
				for employee in self.employees:
					task_num = len(employee.car_moves)
					if task_num < self.world_instance.first_stage_tasks:
						first_stage = True
			else:
				available_employees = False

	def remove_car_move(self, chosen_car_move, car_moves):
		car = chosen_car_move.car.car_id
		# return list of car moves that are not associated with the car of the chosen car move
		return [cm for cm in car_moves if cm.car.car_id != car]

	def add_car_moves_to_employees(self):
		cars_copy = self.cars  # C
		car_moves_copy = self.car_moves  # beta
		obj_val = 0
		while cars_copy:
			charging_prioritized = False
			for employee in self.employees:
				best_car_move = None
				car_moved = None
				new_obj_val = 0
				for car in cars_copy:
					if car.needs_charging or charging_prioritized:
						for car_move in car.car_moves:
							if self.world_instance.check_legal_move(car_move, employee):
								# objective_function(car_move):
								self.world_instance.add_car_move_to_employee(car_move, employee)
								temp_val = self._calculate_objective_function(self.employees)
								# print(temp_val)
								self.world_instance.remove_car_move_from_employee(car_move, employee)
								# print(f"temp_val: {temp_val}, new_obj_val: {new_obj_val}, obj_val: {obj_val}")
								if temp_val > new_obj_val:
									new_obj_val = temp_val
									best_car_move = car_move
									car_moved = car_move.car

				if best_car_move is None:
					if charging_prioritized:
						cars_copy = []
					else:
						# print("Charging prioritized")
						charging_prioritized = True
				else:
					# print('\nEmployee id', employee.employee_id)
					# print('Employee node before', employee.current_node.node_id)
					# print('Employee time before', employee.current_time)
					# print(best_car_move.to_string())
					self.world_instance.add_car_move_to_employee(best_car_move, employee)
					# print('Employee node after', employee.current_node.node_id)
					# print('Employee time after', employee.current_time)
					self.gamma_k[employee.employee_id].append(best_car_move)
					cars_copy.remove(car_moved)
					car_moves_copy.remove(best_car_move)
					obj_val = new_obj_val
			# print(f"obj_val: {obj_val}")

		# TODO: go through each car and their respective moves and then choose the
		# switch up for loops (car before employee)

	def add_car_moves_to_employees2(self):
		obj_val = 0
		car_moves_copy = self.car_moves
		used_cars = []
		prioritize_charging = True
		employee_available = True
		while employee_available:
			employee_available = False
			available_car_moves = [car.car_moves for car in self.cars not in used_cars]
			for employee in self.employees:
				chosen_car_move = self.find_nearest_car_move(employee, available_car_moves, prioritize_charging)
				if chosen_car_move is not None:
					employee_available = True
					self.world_instance.add_car_move_to_employee(chosen_car_move, employee)
					car_moves_copy.remove(chosen_car_move)

	# print(f"obj_val: {obj_val}")
	def find_nearest_car_move(self, employee, car_moves_copy):
		start_node = employee.current_node
		nearest_car_move = None
		best_travel_time = 100  # some high number
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


ch = ConstructionHeuristic("InstanceFiles/6nodes/6-3-1-1_a.pkl")
# print("obj_val", obj_val)
ch.add_car_moves_to_employees3()
ch.print_solution()

''' EXAMPLE OUTPUT
	-------------- First stage routes --------------
	  Employee Task   Route  Travel Time to Task  Start time  Relocation Time  End time
	0        2    1  (2, 6)                  7.7        12.7              7.6      20.3

	-------------- Second stage routes --------------
	  Employee Task Scenario   Route  Travel Time to Task  Start time  Relocation Time  End time
	0        2    2        3  (4, 1)                 19.8        40.1             14.1      54.2
'''