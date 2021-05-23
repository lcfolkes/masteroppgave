from google.oauth2 import service_account
import pygsheets
import pandas as pd
import json
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
	spreadsheet_url = "https://docs.google.com/spreadsheets/d/1xK9FoZhR_-J2ni06U7EBVq40Wno-QA5iIKeSh45H7-k/edit"
	gcs = GoogleSheetClient(url=spreadsheet_url)
	sheet = gcs.get_sheet()

	# Create empty dataframe
	#df = pd.DataFrame()

	# Create a column

	# open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)

	# select the first sheet
	work_sheet = sheet[2]
	#print(work_sheet)
	# update the first sheet with df, starting at cell B2.

	test_results = pd.DataFrame()
	test_dir = "./Testing/destroy_repair_factor/"
	acceptance_percentage_dict = {"[0.05, 0.15]": 2, "[0.15, 0.30]": 4, "[0.05, 0.30]": 6, "[0.15, 0.50]": 8,
								  "[0.30, 0.70]": 10, "[0.50, 0.70]": 12, "[0.15, 0.70]": 14}
	header = np.array([["", "Travel time threshold",
						"[0.05, 0.15] Obj. Val.", "[0.05, 0.15] Time found (s)", "[0.15, 0.30] Obj. Val.", "[0.15, 0.30] Time found (s)",
						"0.5 Obj. Val.", "0.5 Time found (s)", "0.6 Obj. Val.", "0.6 Time found (s)",
						"0.7 Obj. Val.", "0.7 Time found (s)", "0.8 Obj. Val.", "0.8 Time found (s)",
						"0.9 Obj. Val.", "0.9 Time found (s)", "1.0 Obj. Val.", "1.0 Time found (s)"],
					   ["Instance", "Run",
						"Obj. Val.", "Time found (s)", "Obj. Val.", "Time found (s)",
						"Obj. Val.", "Time found (s)", "Obj. Val.", "Time found (s)",
						"Obj. Val.", "Time found (s)", "Obj. Val.", "Time found (s)",
						"Obj. Val.", "Time found (s)", "Obj. Val.", "Time found (s)"]])

	header_df = pd.DataFrame(header)
	work_sheet.set_dataframe(header_df, (1, 0), copy_head=False)
	start_row = 3

	for dir in os.listdir(test_dir):
		sub_dir = test_dir + "/" + dir
		for file in os.listdir(sub_dir):
			filepath = sub_dir + "/" + file
			f = open(filepath, "r")
			result = m = np.zeros(shape=[5, 18]).astype(str)
			filename = file.split(".")[0][:-8]
			result[:, 0] = filename
			result[:, 1] = [x+1 for x in range(5)]
			for x in f:
				line_list = x.split(':')
				if line_list[0] == "Run":
					run = int(line_list[1].split(',')[0].strip())
				elif line_list[0] == "Best objective value found after (s)":
					time = round(float(line_list[1].strip()), 2)
				elif line_list[0] == "Objective value":
					obj_val = line_list[1].strip()
				elif line_list[0] == "Travel time threshold":
					param = line_list[1].strip()
					result[run - 1][acceptance_percentage_dict[param]] = obj_val
					result[run - 1][acceptance_percentage_dict[param]+1] = time

			avg_row = np.array([filename, "Average"])
			relevant_cols = np.array(result[:, 2:], dtype=np.float64)
			avg_obj_val = np.mean(relevant_cols[:, ::2], axis=0)
			avg_time_val = np.mean(relevant_cols[:, 1::2], axis=0)

			avg_row = np.concatenate((avg_row, np.ravel([avg_obj_val, avg_time_val], 'F')), axis=0)

			result = np.vstack([result, avg_row])
			gap_row = np.array([filename, "Gap (%)"])
			max_val_obj = np.amax(relevant_cols[:, ::2])
			max_val_time = np.mean(relevant_cols[:, 1::2])
			obj_val_gaps = np.abs(np.divide(max_val_obj-avg_obj_val, max_val_obj))
			time_val_gaps = np.abs(np.divide(max_val_time-avg_time_val, max_val_time))
			gap_row = np.concatenate((gap_row, np.ravel([obj_val_gaps, time_val_gaps],'F')), axis=0)

			result = np.vstack([result, gap_row])
			result_df = pd.DataFrame(result)
			work_sheet.set_dataframe(result_df, (start_row, 0), copy_head=False)
			start_row += 7
