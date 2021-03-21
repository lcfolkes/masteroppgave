import os
import copy

from InstanceGenerator.world import World
from path_manager import path_to_src
from InstanceGenerator.instance_components import CarMove, Employee

os.chdir(path_to_src)


os.chdir(path_to_src)

class FeasibilityChecker():
	def __init__(self, world_instance: World):
		self.world_instance = world_instance
		#self.employees = self._initialize_employees()

	def _initialize_employees(self):
		employees = copy.deepcopy(self.world_instance.employees)
		for employee in employees:
			employee.reset()
		return employees

	def check_assigned_solution(self, employees):
		pass

	def is_first_stage_solution_feasible(self, solution: {Employee: [CarMove]}):
		#employees = self._initialize_employees()
		feasible = True
		for employee, car_moves in solution.items():
			travel_time = employee.start_time
			current_node = employee.start_node
			for car_move in car_moves:
				start_node = car_move.start_node
				travel_time += self.world_instance.get_employee_travel_time_to_node(current_node, start_node) + car_move.handling_time
				current_node = start_node

			if travel_time > self.world_instance.PLANNING_PERIOD:
				feasible = False

		return feasible

	def check_legal_move(self, car_move: CarMove, employee: Employee, scenario: int = None):  # return total travel time
		current_time = None
		current_node = None
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

		'''
		print(f"\nEmployee {employee.employee_id}")
		print(f"Current node {current_node.node_id}")
		print(car_move.to_string())
		print(f"total_time = current_time + employee_travel_time_to_node + handling_time")
		print(f" {total_time} = {current_time} + {employee_travel_time} + {car_move.handling_time}")
		print(f"planning_period: {World.PLANNING_PERIOD}")
		'''

		if total_time < World.PLANNING_PERIOD:
			return True
		else:
			# print("Car move exceeds planning period")
			return False



if __name__ == "__main__":
	'''ch = ConstructionHeuristic("./InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_a.pkl")
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




