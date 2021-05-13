from google.oauth2 import service_account
import pygsheets
import pandas as pd
import json
from path_manager import path_to_src
import os
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
	df = pd.DataFrame()

	# Create a column
	df['pikk'] = ['penis', 'Steve', 'Sarah']

	# open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)

	# select the first sheet
	work_sheet = sheet[0]

	# update the first sheet with df, starting at cell B2.
	work_sheet.set_dataframe(df, (1, 1))