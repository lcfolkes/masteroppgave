import os
import copy
import numpy as np
from InstanceGenerator.world import World
from path_manager import path_to_src
from InstanceGenerator.instance_components import CarMove, Employee
os.chdir(path_to_src)

class FeasibilityChecker():
	def __init__(self, world_instance: World):
		self.world_instance = world_instance

	def _initialize_employees(self):
		employees = copy.deepcopy(self.world_instance.employees)
		for employee in employees:
			employee.reset()
		return employees

	def check_assigned_solution(self, employees):
		pass

	def is_first_stage_solution_feasible(self, solution: {Employee: [CarMove]}, return_inter_node_travel_time=False, verbose=False):
		if return_inter_node_travel_time:
			inter_node_travel_time = 0
		for employee, car_moves in solution.items():
			travel_time = employee.start_time
			current_node = employee.start_node
			if verbose:
				print(f"\nEmp {employee.employee_id} before: {travel_time}\n")
			for car_move in car_moves:
				start_node = car_move.start_node
				end_node = car_move.end_node
				emp_travel_time_to_node = self.world_instance.get_employee_travel_time_to_node(current_node, start_node)
				if return_inter_node_travel_time:
					inter_node_travel_time += emp_travel_time_to_node
				if verbose:
					print(f"Car_move: {car_move.car_move_id}")
					print(f"-Employee travel time from {current_node.node_id} to {start_node.node_id}: {emp_travel_time_to_node}")
					print(f"-Car move travel time from {start_node.node_id} to {end_node.node_id}: {car_move.handling_time}")
				travel_time += emp_travel_time_to_node + car_move.handling_time
				current_node = end_node
				if verbose:
					print(f"-Emp {employee.employee_id} after: {travel_time}")

				# Checks if best car move is a charging move to a node where the remaining charging capacity is zero
				if car_move.is_charging_move:
					if car_move.end_node.num_charging[0] > car_move.end_node.capacity:
						return False

			if travel_time > self.world_instance.planning_period:
				if return_inter_node_travel_time:
					return False, inter_node_travel_time
				return False
		if return_inter_node_travel_time:
			return True, inter_node_travel_time
		return True

	# check charging capacity constraint
	def is_solution_feasible(self, solution: {Employee: [CarMove]}, return_inter_node_travel_time=False):
		if return_inter_node_travel_time:
			inter_node_travel_time = [0 for _ in range(self.world_instance.num_scenarios)]
		for employee, car_moves in solution.items():
			travel_time = [employee.start_time for _ in range(len(car_moves))]
			current_node = [employee.start_node for _ in range(len(car_moves))]
			for s in range(len(car_moves)):
				for car_move in car_moves[s]:
					start_node = car_move.start_node
					end_node = car_move.end_node
					emp_travel_time_to_node = self.world_instance.get_employee_travel_time_to_node(current_node[s],
																								   start_node)
					if return_inter_node_travel_time:
						inter_node_travel_time[s]+=emp_travel_time_to_node
					travel_time[s] += emp_travel_time_to_node + car_move.handling_time
					current_node[s] = end_node

					if car_move.is_charging_move:
						if car_move.end_node.num_charging[s] > car_move.end_node.capacity:
							return False

				if travel_time[s] > self.world_instance.planning_period:
					if return_inter_node_travel_time:
						return False, np.mean(inter_node_travel_time)
					return False
		if return_inter_node_travel_time:
			return True, np.mean(inter_node_travel_time)
		return True


	def check_legal_move(self, car_move: CarMove, employee: Employee, scenario: int = None, get_employee_travel_time=False):  # return total travel time
		start_node = car_move.start_node
		if scenario is None:
			current_time = employee.current_time
			current_node = employee.current_node
		else:
			# zero-indexed scenario
			current_time = employee.current_time_second_stage[scenario]
			current_node = employee.current_node_second_stage[scenario]

		employee_travel_time = self.world_instance.get_employee_travel_time_to_node(current_node, start_node)

		total_time = current_time + employee_travel_time + car_move.handling_time

		if total_time <= self.world_instance.planning_period:
			legal_move = True
		else:
			legal_move = False

		if get_employee_travel_time:
			return legal_move, employee_travel_time
		else:
			return legal_move



if __name__ == "__main__":
	'''ch = ConstructionHeuristic("./InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_special_case.pkl")
	rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
					   neighborhood_size=2)
	#rr.to_string()
	gi = RegretInsertion(destroyed_solution_object=rr,
						 construction_heuristic=ch, regret_nr=1)
	#gi.to_string()

	fc = FeasibilityChecker(ch.world_instance)
	fc.is_first_stage_solution_feasible(gi.repaired_solution)
	'''
	pass



