from configparser import ConfigParser
import numpy as np
import os
from itertools import product

class ConfigReader:
	def __init__(self, file_path):
		if os.path.isfile(file_path):

			# initialize parser
			parser = ConfigParser()
			parser.read(file_path)

			# Extract data from config file
			self.num_scenarios = int(parser['INITIALIZE']['num_scenarios'])
			self.num_parking_nodes = int(parser['INITIALIZE']['num_parking_nodes']) # number of parking nodes
			self.num_charging_nodes = int(parser['INITIALIZE']['num_charging_nodes']) # number of charging nodes
			self.num_nodes = self.num_parking_nodes + self.num_charging_nodes # number of nodes
			self.num_employees = int(parser['INITIALIZE']['num_employees']) # number of employees
			self.start_node_employee = self.__read_1d_array(parser['INITIALIZE']['start_node_employee']) # start node of employee
			self.charging_slots_available = self.__read_1d_array(parser['INITIALIZE']['charging_slots_available']) # N_i^(CS)
			self.profit_rental = float(parser['INITIALIZE']['profit_rental']) # reward for renting out in second stage
			self.cost_relocation = float(parser['INITIALIZE']['cost_relocation']) # cost of relocation
			self.cost_deviation = float(parser['INITIALIZE']['cost_deviation']) # cost of deviating from ideal state

			#Travel matrices
			self.travel_time_bike = self.__read_2d_array(parser['INITIALIZE']['travel_time_bike'])

			self.travel_time_to_origin = self.__read_1d_array(parser['INITIALIZE']['travel_time_to_origin']) # T_k^(SO)
			self.planning_period = int(parser['INITIALIZE']['planning_period']) # T^bar, planning period
			self.parking_state = self.__read_1d_array(parser['INITIALIZE']['parking_state']) #pstate, all cars parked in node
			self.charging_state = self.__read_1d_array(parser['INITIALIZE']['charging_state']) # cstate, cars in need of charging in node
			self.ideal_state = self.__read_1d_array(parser['INITIALIZE']['ideal_state']) # istate

			self.customer_requests = self.__read_2d_array(parser['INITIALIZE']['customer_requests'], dtype=int)
			self.car_returns = self.__read_2d_array(parser['INITIALIZE']['car_returns'], dtype=int)

			self.num_car_moves_parking = int(parser['INITIALIZE']['num_car_moves_parking']) # number of car moves to parking nodes
			self.num_car_moves_charging = int(parser['INITIALIZE']['num_car_moves_charging']) # number of car moves to charging nodes
			self.num_car_moves = self.num_car_moves_parking + self.num_car_moves_charging
			self.num_cars = int(parser['INITIALIZE']['num_cars']) # number of cars
			self.num_tasks = int(parser['INITIALIZE']['num_tasks']) # number of tasks
			self.num_first_stage_tasks = int(parser['INITIALIZE']['num_first_stage_tasks']) #number of tasks in first stage

			self.car_move_cars = self.__read_1d_array(parser['INITIALIZE']['car_move_cars']) # R_C
			self.car_move_origin = self.__read_1d_array(parser['INITIALIZE']['car_move_origin'])

			self.car_move_destination = self.__read_1d_array(parser['INITIALIZE']['car_move_destination'])
			self.car_move_start_time = self.__read_1d_array(parser['INITIALIZE']['car_move_start_time'])
			self.car_move_handling_time = self.__read_1d_array(parser['INITIALIZE']['car_move_handling_time'], dtype=float)
			self.nodes_with_cars_in_need_of_charging = self.__read_1d_array(parser['INITIALIZE']['nodes_with_cars_in_need_of_charging'])
			self.cars_in_need_of_charging_at_nodes = self.__read_1d_array(parser['INITIALIZE']['cars_in_need_of_charging_at_nodes'])
			self.bigM = self.__read_1d_array(parser['INITIALIZE']['bigM'], dtype=float)
			self.cars_available_in_node = self.__read_1d_array(parser['INITIALIZE']['cars_available_in_node'])

		else:
			print("Config file not found")

	def __read_1d_array(self, arr_str, dtype=int):
		# arr_str = cp.travelTimeVehicle
		arr_str = arr_str.replace("[ ", "")
		arr_str = arr_str.replace(" ]", "")
		try:
			arr = np.array([x.split(' ') for x in arr_str.split('\n')], dtype=dtype).flatten()
		except:
			arr = np.array([])
		return arr

	def __read_2d_array(self, arr_str, dtype=float):
		# arr_str = cp.travelTimeVehicle
		arr_str = arr_str.replace("[ ", "")
		arr_str = arr_str.replace(" ]", "")
		arr = np.array([x.split(' ') for x in arr_str.split('\n')], dtype=dtype)
		return arr

def read_2d_array_to_dict(arr):
	rows = np.arange(1, arr.shape[0]+1) # nodes
	cols = np.arange(1, arr.shape[1]+1) # scenarios
	out_dict = {}
	for (r, c) in product(rows, cols):
		out_dict[(r, c)] = arr[r - 1, c - 1]
	return out_dict

def create_dict_of_indices(indices, item_list):
	d = {}
	for i in indices:
		d[i] = [x+1 for x, e in enumerate(item_list) if e == i]
	return d

def create_car_moves_origin_destination(parking_nodes, charging_nodes, origin_list, destination_list):
	RiPO = {new_list: [] for new_list in parking_nodes}
	RiPD = {new_list: [] for new_list in parking_nodes}
	RiCO = {new_list: [] for new_list in parking_nodes}
	RiCD = {new_list: [] for new_list in charging_nodes}

	for i in range(len(destination_list)):
		# parking moves
		if destination_list[i] in parking_nodes:
			RiPO[origin_list[i]].append(i+1)
			RiPD[destination_list[i]].append(i+1)
		# charging moves
		elif destination_list[i] in charging_nodes:
			RiCO[origin_list[i]].append(i + 1)
			RiCD[destination_list[i]].append(i + 1)

	return RiPO, RiPD, RiCO, RiCD

cp = ConfigReader("../tests/6nodes/6-3-0-1_a.txt")