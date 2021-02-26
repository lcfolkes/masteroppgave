from src.HelperFiles.helper_functions import load_object_from_file
import random
import os

os.chdir('../InstanceGenerator')
# filename = "InstanceFiles/6nodes/6-3-1-1_b.yaml"
dill_filename = "InstanceFiles/6nodes/6-3-1-1_a.pkl"

world_instance = load_object_from_file(dill_filename)
beta = []
cars = world_instance.cars.copy()
employees = world_instance.employees.copy()
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
	for employee in employees:
		best_car_move = None
		car_moved = None
		new_obj_val = 0
		for car in cars:
			for car_move in car.car_moves:
				if world_instance.check_legal_move(car_move, employee) > 0:
					# objective_function(car_move):
					temp_val = objective_function_val(car_move, employee)
					if temp_val > new_obj_val:
						new_obj_val = temp_val
						best_car_move = car_move
						car_moved = car_move.car
		print(employee.employee_id)
		if best_car_move is None:
			cars = []
		else:
			print(best_car_move.to_string())
			print('Employee node', employee.current_node.node_id)
			print('Employee time', employee.current_time)
			world_instance.add_car_move_to_employee(best_car_move, employee)
			print('Employee node', employee.current_node.node_id)
			print('Employee time', employee.current_time)
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


'''
while cars:
	obj_val = 0
	current_solution = []
	for employee in employees:
		best_car_move = None
		car_moved = None
		new_obj_val = 0
		for car in cars:
			for car_move in car.car_moves:
				if world_instance.check_legal_move(car_move, employee) > 0:
					#objective_function(car_move):
					world_instance.add_car_move_to_employee(car_move, employee)
					print(car_move.to_string())
					break
		#	print(f"{car}:{carmoves}")
	#car_index =
	#print(C)
	#C.pop()
'''

'''
cf = read_config(filename)
CARS = list(np.arange(1, cf['num_cars'] + 1))  # C, set of cars potentially subject to relocation
CARMOVES = np.arange(1, cf['num_car_moves_parking'] + cf['num_car_moves_charging'] + 1)  # R, set of car-moves

PARKING_MOVES = np.arange(1, cf['num_car_moves_parking'] + 1)
CHARGING_MOVES = np.arange(cf['num_car_moves_parking'] + 1, cf['num_car_moves_parking'] + cf['num_car_moves_charging'] + 1)  # set of charging-moves, trenger  R_i^CD and R_i^CO

CARMOVES_CAR = create_dict_of_indices(CARS, cf['car_move_cars'])  # R_c, set of car-moves for car c
CARMOVE_ORIGIN = dict(zip(CARMOVES, cf['car_move_origin']))  # o(r), origin node of car move r
CARMOVE_DESTINATION = dict(zip(CARMOVES, cf['car_move_destination']))  # d(r), destination node of car move r

EMPLOYEES = np.arange(1, cf['num_employees'] + 1)  # K, set of service employees

PROFIT_RENTAL = cf['profit_rental']  # C^(RC), Profit per unit of time of renting out cars
COST_RELOCATION = cf['cost_relocation']  # C^R, Cost per unit of time of relocation activities
COST_DEVIATION = cf['cost_deviation']  # C^D, Cost per car of not reaching ideal state

gamma_k = {}
beta = CARMOVES.tolist()


#print(PARKING_MOVES)
#print(CHARGING_MOVES)
print(CARMOVES_CAR)
#print(C)

while CARS:
	current_obj_val = 0
	current_solution = []
	for employee in EMPLOYEES:
		best_car_move = None
		car_moved = None
		for c in CARS:
			for cm in CARMOVES_CAR[c]:
				if objective_function(cm):
		#	print(f"{car}:{carmoves}")
	#car_index =
	#print(C)
	#C.pop()

def objective_function(carmove, employee):
	pass
'''
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
