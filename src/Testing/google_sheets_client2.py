from google.oauth2 import service_account
import pygsheets
import pandas as pd
import json

from Heuristics.helper_functions_heuristics import safe_zero_division
from path_manager import path_to_src
import os
import numpy as np
os.chdir(path_to_src)

class GoogleSheetClient():
	def __init__(self, url):
		google_drive_api_service_file = './Testing/google_drive_api_key.json'
		# authorization
		with open(google_drive_api_service_file) as source:
			info = json.load(source)
		credentials = service_account.Credentials.from_service_account_info(info)

		self.client = pygsheets.authorize(service_file=google_drive_api_service_file)
		self.url = url


	@property
	def sheet_id(self):
		url_list = self.url.split('/')
		idx = url_list.index('d') + 1
		return url_list[idx]

	def get_sheet(self):
		return self.client.open_by_url(self.url)

	def get_sheet_data(self):
		return self.client.sheet.get(self.sheet_id)




if __name__ == "__main__":
	#parameter_tuning_spreadsheet_url = "https://docs.google.com/spreadsheets/d/1xK9FoZhR_-J2ni06U7EBVq40Wno-QA5iIKeSh45H7-k/edit"
	#comp_study_spreadsheet_url = "https://docs.google.com/spreadsheets/d/1hikl4VcCQedeiH3j0tS9QvXm6WbG2EPxNYEh_0VK12Q/edit"
	spreadsheet_url = "https://docs.google.com/spreadsheets/d/1TY_zqaURu1tJwqZnEnUQjJyPlZZKu-jtkoQFeluHxvA/edit"
	gcs = GoogleSheetClient(url=spreadsheet_url)
	sheet = gcs.get_sheet()

	# Create empty dataframe
	#df = pd.DataFrame()

	# Create a column

	# open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)

	# select the first sheet
	work_sheet = sheet[0]
	#print(work_sheet)
	# update the first sheet with df, starting at cell B2.

	test_results = pd.DataFrame()
	test_dir = "./Testing/ComputationalTests/sensitivity_one_hour"
	param_dict = {"36-4": 2, "46-4": 4, "56-4": 6,
				  "36-6": 8, "46-6": 10, "56-6": 12,
				  "36-8": 14, "46-8": 16, "56-8": 18}
	header = np.array([["", "",
						"30-4", "30-4", "40-4", "40-4", "50-4", "50-4",
						"30-6", "30-6", "40-6", "40-6", "50-6", "50-6",
						"30-8", "30-8", "40-8", "40-8", "50-8", "50-8", "Cars In Need Of Charging"],
					   ["Instance", "Run",
						"Charging moves", "Profit", "Charging moves", "Profit", "Charging moves", "Profit",
						"Charging moves", "Profit", "Charging moves", "Profit", "Charging moves", "Profit",
						"Charging moves", "Profit", "Charging moves", "Profit", "Charging moves", "Profit", "Cars In Need Of Charging"]])

	#header_df = pd.DataFrame(header)
	#work_sheet.set_dataframe(header_df, (1, 0), copy_head=False)
	start_row = 0

	result_df = pd.DataFrame(header)
	for dir in os.listdir(test_dir):
		sub_dir = test_dir + "/" + dir
		for file in os.listdir(sub_dir):
			filepath = sub_dir + "/" + file
			f = open(filepath, "r")
			filename_list = file.split(".")[0].split("_")
			filename = "-".join(filename_list[:2])
			instance_mode = filename_list[-1]
			if filename_list[0].split("-")[1] == "1":
				continue

			result = np.zeros(shape=[10, 21]).astype(str)
			result[:, 0] = filename
			for x in f:
				line_list = x.split(':')
				line_list_space = x.split(' ')
				#if line_list[0] == "Run":
				#	run = int(line_list[1].strip())
				if line_list_space[0].strip().isnumeric():
					run = int(line_list_space[0].strip())
				#elif line_list[0] == "Problem type":
				#	problem_type = line_list[1].strip()
				elif line_list[0] == "Best objective value found after (s)":
					time_used = round(float(line_list[1]), 2)
				elif line_list[0] == "Objective value":
					profit = round(float(line_list[1].strip()), 2)
				elif line_list[0] == "Cars charged":
					charging_moves = int(line_list[1].strip())
				elif line_list[0] == "Cars in need of charging":
					cars_in_need = int(line_list[1].strip())
				elif line_list[0] == "Number of cars":
					num_cars = int(line_list[1].strip())
				elif line_list[0] == "Number of employees":
					num_employees = int(line_list[1].strip())
					#if problem_type == "Stochastic":
					param = f"{num_cars}-{num_employees}"
					result[run - 1][1] = run
					result[run - 1][param_dict[param]] = charging_moves
					result[run - 1][param_dict[param]+1] = profit
					#result[run - 1][7] = time_used
					result[run - 1][20] = cars_in_need
					#elif problem_type == "Upgrade":
					#	#result[run - 1][1] = run
					#	result[run - 1][2] = charging_moves
					#	result[run - 1][3] = profit
					#	result[run - 1][4] = time_used

				'''
				elif line_list[0] == "Objective value":
					obj_val = line_list[1].strip()
				elif line_list[0] == "Cars charged":
					cars_charged = line_list[1].strip()
				elif line_list[0] == "Cars in need of charging":
					cars_in_need_of_charging = line_list[1].strip()
				elif line_list[0] == "Construction heuristic time (s)":
					time_ch = round(float(line_list[1].strip()), 2)
				elif line_list[0] == "Construction heuristic, true objective value":
					obj_val_ch = line_list[1].strip()
				elif line_list[0] == "Construction heuristic cars charged":
					cars_charged_ch = line_list[1].strip()
					param = line_list[1].strip()
					result[run - 1][2] = obj_val_ch
					result[run - 1][3] = cars_charged_ch
					result[run - 1][4] = time_ch
					result[run - 1][5] = obj_val
					result[run - 1][6] = cars_charged
					result[run - 1][7] = time_obj
					result[run - 1][8] = cars_in_need_of_charging
				'''
			result_df = result_df.append(pd.DataFrame(result, columns=result_df.columns), ignore_index=True)


			'''
			avg_row = np.array([filename, "Average"])
			#relevant_cols = np.array(result[:, 2:], dtype=np.float64)
			avg_obj_val = np.mean(np.array(result[:, [2, 5]], dtype=np.float64), axis=0)
			avg_charge_val = np.mean(np.array(result[:, [3, 6]], dtype=np.float64), axis=0)
			avg_time_val = np.mean(np.array(result[:, [4, 7]], dtype=np.float64), axis=0)
			avg_row = np.concatenate((avg_row, np.ravel([avg_obj_val, avg_charge_val, avg_time_val], 'F')), axis=0)
			avg_need_charging = np.mean(np.array(result[:, [8]], dtype=np.float64), axis=0)
			avg_row = np.concatenate((avg_row, avg_need_charging), axis=0)
			result = np.vstack([result, avg_row])


			gap_row = np.array([filename, "Gap (%)"])
			max_val_obj = np.amax(np.array(result[:, [2, 5]], dtype=np.float64))
			max_val_time = np.amax(np.array(result[:, [4, 7]], dtype=np.float64))
			obj_val_gaps = np.abs(safe_zero_division(max_val_obj-avg_obj_val, max_val_obj))
			charg_gaps = np.abs(safe_zero_division(avg_need_charging-avg_charge_val, avg_need_charging))
			time_val_gaps = np.abs(safe_zero_division(max_val_time-avg_time_val, max_val_time))
			gap_row = np.concatenate((gap_row, np.ravel([obj_val_gaps, charg_gaps, time_val_gaps],'F')), axis=0)
			gap_row = np.concatenate((gap_row, avg_need_charging), axis=0)
			result = np.vstack([result, gap_row])

			result_df = pd.DataFrame(result)
			work_sheet.set_dataframe(result_df, (start_row, 0), copy_head=False)
			start_row += 12
			'''
		work_sheet.set_dataframe(result_df, (start_row, 0), copy_head=False)