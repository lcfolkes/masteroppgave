import os
from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance

os.chdir('../InstanceGenerator')
from src.InstanceGenerator.instance_components import ParkingNode, Employee, ChargingNode, CarMove
from InstanceGenerator.world import World
from src.HelperFiles.helper_functions import load_object_from_file
from src.Gurobi.Model.run_model import run_model
import numpy as np

def remove_car_move(chosen_car_move, car_moves):
	car = chosen_car_move.car.car_id
	# return list of car moves that are not associated with the car of the chosen car move
	return [cm for cm in car_moves if cm.car.car_id != car]

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
		self.num_scenarios = self.world_instance.num_scenarios

		self.available_employees = True
		self.prioritize_charging = True
		self.first_stage = True
		self.charging_moves_second_stage = []
		self.parking_moves_second_stage = []

		for car in self.world_instance.cars:
			for car_move in car.car_moves:
				self.car_moves.append(car_move)
				if car.needs_charging:
					self.charging_moves.append(car_move)
				else:
					self.parking_moves.append(car_move)

	def _calculate_z(self, first_stage_car_moves, second_stage_car_moves, verbose=False):
		# z is the number of customer requests served. It must be the lower of the two values
		# available cars and the number of customer requests D_is
		# number of available cars in the beginning of the second stage, y_i

		start_nodes_first_stage = [car_move.start_node for car_move in first_stage_car_moves]
		end_nodes_first_stage = [car_move.end_node for car_move in first_stage_car_moves if isinstance(car_move.end_node, ParkingNode)]

		if verbose:
			print([[car_move.car_move_id for car_move in scenarios] for scenarios in second_stage_car_moves])
			#print(end_nodes_first_stage)

		y = {parking_node.node_id: parking_node.parking_state for parking_node in self.parking_nodes}
		for n in start_nodes_first_stage:
			y[n.node_id] -= 1
		for n in end_nodes_first_stage:
			y[n.node_id] += 1


		node_demands = {parking_node.node_id: {'customer_requests': parking_node.customer_requests,
											   'car_returns': parking_node.car_returns} for parking_node in
						self.parking_nodes}

		z = {}

		start_nodes_second_stage = [[car_move.start_node.node_id for car_move in scenarios] for scenarios in
									second_stage_car_moves]
		# car_move.start_node should be a list of car moves with len(list) = num_scenarios
		for n in self.parking_nodes:
			second_stage_moves_out = np.array([cm.count(n.node_id) for cm in start_nodes_second_stage])
			y[n.node_id] = np.maximum(y[n.node_id], 0)
			if verbose:
				print(f"y[{n.node_id}] {y[n.node_id]}")

			z_val = np.minimum(y[n.node_id] + node_demands[n.node_id]['car_returns'] - second_stage_moves_out,
							   node_demands[n.node_id]['customer_requests'])
			z_val = np.maximum(z_val, 0)
			z[n.node_id] = z_val
		if verbose:
			for n in self.parking_nodes:
				print(f"z[{n.node_id}] {z[n.node_id]}")

		return z

	def _calculate_profit_customer_requests(self, z):
		# print(z)
		# sum across scenarios for all nodes
		z_sum = sum(v for k, v in z.items())
		z_sum_scenario_average = np.mean(z_sum)
		return World.PROFIT_RENTAL * z_sum_scenario_average

	def _calculate_costs_relocation(self, car_moves):
		# Sum of all travel times across all car moves
		sum_travel_time = sum(car_move.handling_time for car_move in car_moves)
		sum_travel_time_scenario_avg = sum_travel_time / self.num_scenarios
		return World.COST_RELOCATION * sum_travel_time_scenario_avg

	def _calculate_cost_deviation_ideal_state(self, z, first_stage_car_moves, second_stage_car_moves, verbose=False):
		start_nodes_first_stage = [car_move.start_node for car_move in first_stage_car_moves]
		end_nodes_first_stage = [car_move.end_node for car_move in first_stage_car_moves if
								 isinstance(car_move.end_node, ParkingNode)]

		w = {n.node_id: (n.ideal_state - n.parking_state + z[n.node_id] - n.car_returns) for n in self.parking_nodes}


		for n in start_nodes_first_stage:
			w[n.node_id] += 1
		for n in end_nodes_first_stage:
			w[n.node_id] -= 1

		# print(w)

		start_nodes_second_stage = [[car_move.start_node for car_move in scenarios] for scenarios in
									second_stage_car_moves]
		end_nodes_second_stage = [[car_move.end_node for car_move in scenarios] for scenarios in second_stage_car_moves]
		for n in self.parking_nodes:
			second_stage_moves_out = np.array([cm.count(n) for cm in start_nodes_second_stage])
			second_stage_moves_in = np.array([cm.count(n) for cm in end_nodes_second_stage])
			w[n.node_id] += second_stage_moves_out - second_stage_moves_in
			# require w_is >= 0
			w[n.node_id] = np.maximum(w[n.node_id], 0)
			if verbose:
				print(f"w[{n.node_id}] {w[n.node_id]}")


		w_sum = sum(v for k, v in w.items())
		w_sum_scenario_average = np.mean(w_sum)
		# only return first scenario for now
		return World.COST_DEVIATION * w_sum_scenario_average

	def _get_obj_val_of_car_move(self, first_stage_car_moves: [CarMove] = None, second_stage_car_moves: [CarMove] = None,
								scenario=None):
		'''pa = argparse.ArgumentParser()
		args = pa.parse_args()
		if args.first_stage_car_moves is None and args.second_stage_car_moves is None:
			pa.error("at least one of --first_stage_car_moves and --second_stage_car_moves required")
		if args.second_stage_car_moves is not None and args.second_stage_car_moves is None:
			pa.error(" --scenario required when --second_stage_car_moves specified")
		if args.second_stage_car_moves is None and args.second_stage_car_moves is not None:
			pa.error(" --second_stage_car_moves required when --scenario specified")'''

		# first stage
		if first_stage_car_moves is not None:
			z = self._calculate_z(first_stage_car_moves=first_stage_car_moves, second_stage_car_moves=[[]])
			cost_deviation_ideal_state = self._calculate_cost_deviation_ideal_state(z,
																					first_stage_car_moves=first_stage_car_moves,
																					second_stage_car_moves=[[]])
			cost_relocation = self._calculate_costs_relocation(first_stage_car_moves)

		else:
			car_moves_second_stage = list(range(self.num_scenarios))
			car_moves_second_stage.insert(scenario, second_stage_car_moves)
			z = self._calculate_z(first_stage_car_moves=first_stage_car_moves,
								  second_stage_car_moves=car_moves_second_stage)
			cost_deviation_ideal_state = self._calculate_cost_deviation_ideal_state(z,
																					first_stage_car_moves=first_stage_car_moves,
																					second_stage_car_moves=car_moves_second_stage)
			cost_relocation = self._calculate_costs_relocation(first_stage_car_moves + car_moves_second_stage)

		profit_customer_requests = self._calculate_profit_customer_requests(z)

		#print(profit_customer_requests - cost_relocation - cost_deviation_ideal_state)
		#print("profit_customer_requests: ", profit_customer_requests)
		#print("cost_relocation: ", cost_relocation)
		#print("cost_deviation_ideal_state: ", cost_deviation_ideal_state)
		return profit_customer_requests - cost_relocation - cost_deviation_ideal_state

	def get_objective_function_val(self):
		first_stage_car_moves = []
		second_stage_car_moves = [[] for _ in range(self.num_scenarios)]
		for employee in self.employees:

			for car_move in employee.car_moves:
				first_stage_car_moves.append(car_move)

			for s in range(self.num_scenarios):
				for car_move in employee.car_moves_second_stage[s]:
					second_stage_car_moves[s].append(car_move)

		all_second_stage_car_moves = [cm for s in second_stage_car_moves for cm in s]
		all_car_moves = first_stage_car_moves + all_second_stage_car_moves
		print([[car_move.car_move_id for car_move in scenarios] for scenarios in second_stage_car_moves])
		z = self._calculate_z(first_stage_car_moves, second_stage_car_moves, True)
		profit_customer_requests = self._calculate_profit_customer_requests(z)
		cost_relocation = self._calculate_costs_relocation(all_car_moves)
		cost_deviation_ideal_state = self._calculate_cost_deviation_ideal_state(z, first_stage_car_moves,
																				second_stage_car_moves,True)
		print(f"profit_customer_requests: {profit_customer_requests}")
		print(f"cost_relocation: {cost_relocation}")
		print(f"cost_deviation_ideal_state: {cost_deviation_ideal_state}")
		obj_val = profit_customer_requests - cost_relocation - cost_deviation_ideal_state
		print("Objective function value: ", round(obj_val, 2))
		return obj_val



	def add_car_moves_to_employees(self):
		it = 0

		while self.available_employees:
			it += 1
			print("\niteration: ", it)
			print("first_stage: ", self.first_stage)
			# check if charging_moves_list is not empty
			if self.charging_moves:
				self.prioritize_charging = True
				if self.first_stage:
					car_moves = self.charging_moves
				else:
					car_moves = self.charging_moves_second_stage

			else:
				self.prioritize_charging = False
				if self.first_stage:
					car_moves = self.parking_moves
				else:
					car_moves = self.parking_moves_second_stage

			if it > 50:
				print(len(car_moves[0]))
				self.available_employees = False

			if self.first_stage:
				#### GET BEST CAR MOVE ###
				best_car_move_first_stage = self._get_best_car_move(car_moves=car_moves)
				#### GET BEST EMPLOYEE ###
				best_employee_first_stage = self._get_best_employee(best_car_move=best_car_move_first_stage)
				if best_employee_first_stage is not None:
					#### ADD CAR MOVE TO EMPLOYEE ###
					self._add_car_move_to_employee(car_moves=car_moves, best_car_move=best_car_move_first_stage,
												   best_employee=best_employee_first_stage)

			else:
				#### GET BEST CAR MOVE ###
				best_car_move_second_stage = self._get_best_car_move(car_moves=car_moves)
				#### GET BEST EMPLOYEE ###
				best_employee_second_stage = self._get_best_employee(best_car_move=best_car_move_second_stage)
				#### ADD CAR MOVE TO EMPLOYEE ###
				if best_employee_second_stage is not None:
					self._add_car_move_to_employee(car_moves=car_moves, best_car_move=best_car_move_second_stage,
											   best_employee=best_employee_second_stage)



	def _get_assigned_car_moves(self, scenario: int = None):
		car_moves = []
		if scenario is None:
			for employee in self.employees:
				for car_move in employee.car_moves:
					car_moves.append(car_move)
		else:
			for employee in self.employees:
				for car_move in employee.car_moves_second_stage[scenario]:
					car_moves.append(car_move)

		return car_moves

	def _get_best_car_move(self, car_moves):

		if self.first_stage:
			best_car_move_first_stage = None
			assigned_car_moves_first_stage = self._get_assigned_car_moves()
			best_obj_val_first_stage = -1000
			#if not self.prioritize_charging:
			#	best_obj_val_first_stage = self._get_obj_val_of_car_move(first_stage_car_moves=assigned_car_moves_first_stage)

			for r in range(len(car_moves)):
				obj_val = self._get_obj_val_of_car_move(first_stage_car_moves=assigned_car_moves_first_stage + [car_moves[r]])
				if obj_val > best_obj_val_first_stage:
					best_obj_val_first_stage = obj_val
					best_car_move_first_stage = car_moves[r]

			print("obj_val: ", obj_val)
			print("best_obj_val: ", best_obj_val_first_stage)
			print(f"best_car_move: {best_car_move_first_stage.car_move_id}, {best_car_move_first_stage.start_node.node_id} --> {best_car_move_first_stage.end_node.node_id}")
			return best_car_move_first_stage

		else:
			best_car_move_second_stage = [None] * self.num_scenarios
			best_obj_val_second_stage = [-1000] * self.num_scenarios
			obj_val = [0] * self.num_scenarios
			assigned_first_stage_car_moves = self._get_assigned_car_moves()

			for s in range(self.num_scenarios):
				# zero indexed scenario
				assigned_second_stage_car_moves = self._get_assigned_car_moves(scenario=s)
				for r in range(len(car_moves[s])):
					obj_val[s] = self._get_obj_val_of_car_move(first_stage_car_moves=assigned_first_stage_car_moves,
															   second_stage_car_moves=assigned_second_stage_car_moves + [
																   car_moves[s][r]], scenario=s)
					if obj_val[s] > best_obj_val_second_stage[s]:
						best_obj_val_second_stage[s] = obj_val[s]
						best_car_move_second_stage[s] = car_moves[s][r]

			return best_car_move_second_stage

	def _get_best_employee(self, best_car_move):
		if self.first_stage:
			best_employee = None
			best_travel_time_to_car_move = 100
			end_node = best_car_move.start_node
		else:
			best_employee_second_stage = [None] * self.num_scenarios
			best_travel_time_to_car_move_second_stage = [100] * self.num_scenarios
			end_node = [(cm.start_node if cm is not None else cm) for cm in best_car_move]

		best_move_not_legal = True

		for employee in self.employees:
			task_num = len(employee.car_moves)
			# if first stage and the number of completed task for employee is below the number of tasks in first stage,
			# or if second stage and the number of completed tasks are the same or larger than the number of tasks in first stage
			if self.first_stage == (task_num < self.world_instance.first_stage_tasks):
				if self.first_stage:
					legal_move = self.world_instance.check_legal_move(car_move=best_car_move, employee=employee)
					if legal_move:
						start_node = employee.current_node
						travel_time_to_car_move = self.world_instance.get_employee_travel_time_to_node(start_node,
																									   end_node)
						if travel_time_to_car_move < best_travel_time_to_car_move:
							best_travel_time_to_car_move = travel_time_to_car_move
							best_employee = employee
						best_move_not_legal = False

					else:
						# TODO: if no employee can perform car move, remove car move.
						# if best car_move gives negative objfunc value
						# if no car_move possible or no improving car moves,
						# allow employees to enter first and second stage independently?
						pass
					print("employee: ", employee.employee_id)
					print("legal_move: ", legal_move)
				else:
					for s in range(self.num_scenarios):
						if best_car_move[s] is not None:
							legal_move = self.world_instance.check_legal_move(
								car_move=best_car_move[s], employee=employee, scenario=s)
							if legal_move:
								start_node = employee.current_node_second_stage[s]
								travel_time_to_car_move = self.world_instance.get_employee_travel_time_to_node(
									start_node, end_node[s])
								if travel_time_to_car_move < best_travel_time_to_car_move_second_stage[s]:
									best_travel_time_to_car_move_second_stage[s] = travel_time_to_car_move
									best_employee_second_stage[s] = employee

								best_move_not_legal = False

							print("employee: ", employee.employee_id)
							print("legal_move: ", legal_move)


		# Remove best move if not legal. Else return best employee
		if self.first_stage:
			if best_move_not_legal:
				if self.prioritize_charging:
					self.charging_moves.remove(best_car_move)
				else:
					self.parking_moves.remove(best_car_move)
				return
			else:
				return best_employee
		else:
			if best_move_not_legal:
				if self.prioritize_charging:
					for s in range(self.num_scenarios):
						self.charging_moves_second_stage[s] = [cm for cm in self.charging_moves_second_stage[s] if cm != best_car_move[s]]
				else:
					for s in range(self.num_scenarios):
						self.parking_moves_second_stage[s] = [cm for cm in self.parking_moves_second_stage[s] if cm != best_car_move[s]]
				return
			else:
				return best_employee_second_stage

	# print(best_travel_time_to_car_move_second_stage)

	def _add_car_move_to_employee(self, car_moves, best_car_move, best_employee):
		if self.first_stage:
			if best_employee is not None:
				print('\nEmployee id', best_employee.employee_id)
				print('Employee node before', best_employee.current_node.node_id)
				print('Employee time before', best_employee.current_time)
				#print('Travel time to start node', best_travel_time_to_car_move)
				print(best_car_move.to_string())
				self.world_instance.add_car_move_to_employee(best_car_move, best_employee)
				print('Employee node after', best_employee.current_node.node_id)
				print('Employee time after', best_employee.current_time)
				if self.prioritize_charging:
					self.charging_moves = remove_car_move(best_car_move, car_moves)  # should remove car move and other car-moves with the same car
				else:
					self.parking_moves = remove_car_move(best_car_move, car_moves)  # should remove car move and other car-moves with the same car

				self.first_stage = False
				for employee in self.employees:
					task_num = len(employee.car_moves)
					if task_num < self.world_instance.first_stage_tasks:
						self.first_stage = True
				if not self.first_stage:
					# initialize charging and parking moves for second stage
					self.charging_moves_second_stage = [self.charging_moves for s in range(self.num_scenarios)]
					self.parking_moves_second_stage = [self.parking_moves for s in range(self.num_scenarios)]
			else:
				self.available_employees = False
		# Second stage
		else:
			# print(best_car_move_second_stage)
			# If any employee is not note, continue
			if not all(e is None for e in best_employee):
				for s in range(self.num_scenarios):
					# print(best_employee_second_stage[s].to_string())
					if best_employee[s] is not None:
						print('\nEmployee id', best_employee[s].employee_id)
						print('\nScenario', s+1)
						print('Employee node before', best_employee[s].current_node_second_stage[s].node_id)
						print('Employee time before', best_employee[s].current_time_second_stage[s])
						#print('Travel time to start node', best_travel_time_to_car_move_second_stage[s])
						self.world_instance.add_car_move_to_employee(best_car_move[s], best_employee[s], s)
						print('Employee node after', best_employee[s].current_node_second_stage[s].node_id)
						print('Employee time after', best_employee[s].current_time_second_stage[s])
						# When first stage is finished, initialize car_moves to be list of copies of car_moves (number of copies = num_scenarios)
						if self.prioritize_charging:
							self.charging_moves_second_stage[s] = remove_car_move(best_car_move[s],
																				  car_moves[s])  # should remove car move and other car-moves with the same car
						else:
							self.parking_moves_second_stage[s] = remove_car_move(best_car_move[s],
																				 car_moves[s])  # should remove car move and other car-moves wit
				# print(f"car_moves: {len(car_moves[s])}")
				if not any(self.parking_moves_second_stage):
					self.available_employees = False
			else:
				self.available_employees = False



	# print(f"obj_val: {obj_val}")

	def print_solution(self):

		print("-------------- First stage routes --------------")
		for employee in self.employees:
			for car_move in employee.car_moves:
				print(f"employee: {employee.employee_id}, " + car_move.to_string())

		print("-------------- Second stage routes --------------")
		for employee in self.employees:
			if any(employee.car_moves_second_stage):
				for s in range(self.num_scenarios):
					for car_move in employee.car_moves_second_stage[s]:
						print(f"employee: {employee.employee_id}, scenario: {s + 1} " + car_move.to_string())




print("\n---- HEURISTIC ----")
ch = ConstructionHeuristic("InstanceFiles/6nodes/6-3-1-1_b.pkl")
ch.add_car_moves_to_employees()
ch.print_solution()
ch.get_objective_function_val()
print("\n---- GUROBI ----")
gi = GurobiInstance("InstanceFiles/6nodes/6-3-1-1_b.yaml", ch.employees)
# gi = GurobiInstance("InstanceFiles/6nodes/6-3-1-1_a.yaml")
run_model(gi)
