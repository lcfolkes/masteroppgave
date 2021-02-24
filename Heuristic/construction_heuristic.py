from HelperFiles.helper_functions import read_config, read_2d_array_to_dict, create_dict_of_indices, create_car_moves_origin_destination
import numpy as np
import gurobipy as gp


filename = "../InstanceGenerator/InstanceFiles/6nodes/6-3-1-1_b.yaml"

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