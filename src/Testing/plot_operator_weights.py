#%%
from path_manager import path_to_src
import os
import json
import numpy as np
import matplotlib.pyplot as plt
os.chdir(path_to_src)
test_dir = "./Testing/ComputationalTests/medium_large_one_hour"


#%%
instance_dict = {}
for directory in os.listdir(test_dir):
	sub_dir = test_dir + "/" + directory
	for file in os.listdir(sub_dir):
		filepath = sub_dir + "/" + file
		f = open(filepath, "r")
		instance_name = filepath.split("/")[-1][:-12]
		weights_list = []
		for x in f:
			line_list = x.split(':')
			if line_list[0] == "Operator weights":
				s = x[x.index("{"):]
				json_acceptable_string = s.replace("'", "\"")
				d = json.loads(json_acceptable_string)
				weights_list.append(d)
		instance_dict[instance_name] = weights_list

new_instance_dict = {}
for instance in instance_dict.keys():
	for count, run in enumerate(instance_dict[instance]):
		if count == 0:
			new_instance_dict = {k: [] for k in run.keys()}
		for k, v in run.items():
			new_instance_dict[k].append(v)

	new_average_instance_dict = {}
	for k, v in new_instance_dict.items():
		n_segments = max([len(l) for l in v])
		segments = [[] for _ in range(n_segments)]
		for l in v:
			for i, x in enumerate(l):
				segments[i].append(float(x))
		segment_avgs = [round(np.mean(s), 2) for s in segments]
		new_average_instance_dict[k] = segment_avgs

	x_axis = [50*n for n in range(n_segments)]
	keys_list = sorted(list(new_average_instance_dict.keys()))

	print(keys_list)
	for i, k in enumerate(keys_list):
		plt.plot(x_axis, new_average_instance_dict[k], label=k)
		if i % 5 == 4:
			plt.title(f'Adaptive weights development for instance {instance}')
			plt.ylabel('Weight')
			plt.xlabel('Iterations')
			plt.legend()
			plt.show()






