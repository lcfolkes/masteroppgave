from path_manager import path_to_src
import os
import json
os.chdir(path_to_src)
test_dir = "./Testing/Results"

for dir in os.listdir(test_dir):
	sub_dir = test_dir + "/" + dir
	for file in os.listdir(sub_dir):
		filepath = sub_dir + "/" + file
		f = open(filepath, "r")
		for x in f:
			line_list = x.split(':')
			if line_list[0] == "Operator weights":
				s = x[x.index("{"):]
				json_acceptable_string = s.replace("'", "\"")
				d = json.loads(json_acceptable_string)
				for k, v in d.items():
					print(k)
					print(v)
