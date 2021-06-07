import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from HelperFiles.helper_functions import read_config
import json
from path_manager import path_to_src
os.chdir(path_to_src)

relocation_time_threshold_factor = 0.3

def get_files(node_list): #[30,40,50]
	files = []
	for n in node_list:
		directory = f"./InstanceGenerator/InstanceFiles/{n}nodes/"
		for filename in os.listdir(directory):
			filename_list = filename.split(".")
			if filename_list[-1] == "yaml":
				if filename_list[0].split("-")[1] == '25':
					filename = ".".join(filename_list)
					files.append(os.path.join(directory, filename))
	return files

def get_car_moves_from_best_solutions_files():
	files = []
	for n in [30, 40, 50]:
		directory = f"./Testing/ComputationalTests/TravelTimeResults/{n}nodes/"
		for filename in os.listdir(directory):
			files.append(os.path.join(directory, filename))
	return files


def get_handling_times(files):
	handling_times = []
	instance_max_vals = {}
	instance_num_carmoves = {}
	for f in files:
		instance_name = f.split("/")[-1].split(".")[0]
		cf = read_config(f)
		instance_vals = cf['car_move_handling_time']
		max_val = max(instance_vals)
		instance_max_vals[instance_name] = max_val
		instance_num_carmoves[instance_name] = len(instance_vals)
		fractional_values = list(np.divide(instance_vals, max_val))
		handling_times += fractional_values
	return handling_times, instance_max_vals, instance_num_carmoves

def get_parking_handling_times(files):
	handling_times = []
	instance_max_vals = {}
	instance_num_carmoves = {}
	for f in files:
		instance_name = f.split("/")[-1].split(".")[0]
		cf = read_config(f)
		num_parking = cf['num_parking_nodes']
		parking_indices = [i for i, x in enumerate(cf['car_move_destination']) if x <= num_parking]
		instance_vals = [x for i, x in enumerate(cf['car_move_handling_time']) if i in parking_indices]

		max_val = max(instance_vals)
		instance_max_vals[instance_name] = max_val
		instance_num_carmoves[instance_name] = len(instance_vals)
		fractional_values = list(np.divide(instance_vals, max_val))
		handling_times += fractional_values
		print(f)
		print(f"Num car-moves: {len(cf['car_move_destination'])}")
		print(f"Remove car-moves: {sum(1 for x in fractional_values if x > relocation_time_threshold_factor)}")
		print(f"Car-moves considered: {len(cf['car_move_destination'])-sum(1 for x in fractional_values if x > relocation_time_threshold_factor)}")
	return handling_times, instance_max_vals, instance_num_carmoves

def plot_values(handling_times, parking_values, charging_values):
	plt.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"font.sans-serif": ["Computer Modern Roman"],
		"legend.fancybox": False

	})
	ax = sns.displot(data=[handling_times, parking_values, charging_values], color=['#228b21', 'darkorange', 'royalblue'],
					 kind="kde", bw_adjust=5, label='All available car-moves')
	sns.despine(left=True, top=True, right=True)
	#ax.set_ylabels([])
	#ax.set_yticklabels([])
	ax._legend.remove()
	#ax.spines['top'].set_visible(False)
	ax.set(xlabel='Fraction of longest relocation time')
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.legend(labels=['Charging-moves', 'Parking-moves', 'All available car-moves'])
	plt.tick_params(left=False, right=False, labelleft=False,
					labelbottom=True, bottom=True)
	plt.ylabel("")
	plt.show()

def plot_values_std(handling_times, parking_values=[], charging_values=[]):
	plt.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"font.sans-serif": ["Computer Modern Roman"],
		"legend.fancybox": False

	})

	#data = [handling_times, parking_values] #, , charging_values]
	data = [handling_times]
	colors = ['darkorange', '#228b21']
	#g = sns.displot(data=[parking_values, handling_times], color=['#228b21', 'darkorange'], kind="kde", bw_adjust=1.75,
	#				label='All available parking-moves')
	g = sns.displot(data=[handling_times], color=['#228b21'], kind="kde", bw_adjust=1.75,
					label='All available parking-moves')
	for ax in g.axes.flat:
		for i, line in enumerate(ax.lines):
			x = np.array(data[-(i + 1)])
			xs = line.get_xdata()
			ys = line.get_ydata()
			left = 0
			right = relocation_time_threshold_factor
			if i == 0:
				#middle = np.mean(x)
				percentage = str(round(100*len([i for i in x if i > relocation_time_threshold_factor])/len(x), 1)) + "\%"
				ax.text(relocation_time_threshold_factor + 0.015, 0.03, percentage, **dict(size=8))
				ax.fill_between(xs, 0, ys, facecolor=colors[i], alpha=0.1)
				ax.fill_between(xs, 0, ys, where=(left <= xs) & (xs <= right), interpolate=True, facecolor=colors[i], alpha=0.1)
				ax.vlines(relocation_time_threshold_factor, 0, np.interp(relocation_time_threshold_factor, xs, ys), color="black", ls=':')

			else:
				percentage = str(round(100*len([i for i in x if i > 0.5])/len(x), 1)) + "\%"
				ax.text(relocation_time_threshold_factor + 0.015, 0.45, percentage, **dict(size=8))
				ax.fill_between(xs, 0, ys, facecolor=colors[i], alpha=0.1)
				ax.fill_between(xs, 0, ys, where=(left <= xs) & (xs <= right), interpolate=True, facecolor=colors[i],
								alpha=0.15)
			line.set_color(colors[i])

	# ax.set_ylim(ymin=0)
	sns.despine(left=True, top=True, right=True)
	g._legend.remove()
	#ax.spines['top'].set_visible(False)
	g.set(xlabel='Fraction of longest relocation time')

	plt.gcf().subplots_adjust(bottom=0.15)
	labels = ['All available parking-moves', 'Parking-moves in best solutions'] #['Charging-moves', 'Parking-moves', 'All available car-moves']
	plt.tick_params(left=False, right=False, labelleft=False,
					labelbottom=True, bottom=True)
	plt.ylabel("")
	plt.xlim(0, 1)
	plt.xticks([0.1 * x for x in range(11)], [str(round(0.1*x,1)) if (x%2==0) else "" for x in range(11)])
	plt.legend(labels=labels, loc='upper right', bbox_to_anchor=(1.15, 1.0))

	plt.show()

def get_fractional_results():
	files = get_files([30, 40, 50])
	handling_times, instance_max_vals, _ = get_handling_times(files)
	files = get_car_moves_from_best_solutions_files()
	fraction_list_parking = []
	fraction_list_charging = []
	for file in files:
		filename = "_".join(file.split("/")[-1].split("_")[:2])
		with open(file, "r") as f:
			for x in f:
				line_list = x.split(":")
				if line_list[0] == "Car-moves duration":
					dict_string = x.split("{")[1].split("}")[:-1]
					dict_string = "{" + dict_string[0] + "}"
					solution_dict = eval(dict_string)
					for k, v in solution_dict.items():
						if v[-1]:  # charging move
							fraction_list_charging += [v[1] / instance_max_vals[filename] for _ in range(v[0])]
						else:
							fraction_list_parking += [v[1] / instance_max_vals[filename] for _ in range(v[0])]
	return fraction_list_parking, fraction_list_charging


def plot_travel_time_threshold():
	plt.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"font.sans-serif": ["Computer Modern Roman"],
		"legend.fancybox": False
	})
	x = np.array([0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
	gap = np.array([157, 36.98, 7.93, 5.90, 4.91, 5.22, 5.06, 4.67])
	time = np.array([2.5, 90, 166.5, 162.3, 137.0, 151.5, 189.9, 219.2])
	#gap = np.array([123.03, 18.63, 2.96, 5.26, 5.88, 6.25, 6.25, 5.52])
	#time = np.array([3.3, 68.1, 102.9, 162.6, 166.3, 212.6, 250.4, 277.0])
	fig, ax1 = plt.subplots()

	ax1.set_xlabel('Relocation time threshold factor')
	ax1.set_ylabel('Gap (\%) from best-known solution')
	ax1.plot(x, gap, label="Gap (\%)",  color="#228b21")
	ax1.tick_params(axis='y')
	ax1.spines['top'].set_visible(False)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	ax2.set_ylabel('Computation time (s)')  # we already handled the x-label with ax1
	ax2.plot(x, time, label="Computation time (s)", color="darkorange")
	ax2.tick_params(axis='y')
	ax2.spines['top'].set_visible(False)

	fig.legend(loc="upper center")
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()

def plot_construction_heuristic_run_time():
	plt.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"font.sans-serif": ["Computer Modern Roman"],
		"legend.fancybox": False
	})
	x = np.array([6, 8, 10, 15, 20, 25, 30, 40, 50])
	#gap = np.array([157, 36.98, 7.93, 5.90, 4.91, 5.22, 5.06, 4.67])
	#time = np.array([2.5, 90, 166.5, 162.3, 137.0, 151.5, 189.9, 219.2])
	construction_heuristic_time_all = np.array([[0.12, 0.11, 0.15], [0.24, 0.55, 0.15], [0.75, 0.65, 0.60],
											[0.45, 0.71, 0.63], [4.87, 2.47, 3.20], [7.95, 3.22, 4.26],
											[15.06, 9.71, 14.53], [66.21, 52.35, 40.75], [123.58, 138.90, 179.51]])


	instance_car_moves = np.array([[26, 26, 26], [43, 43, 43], [73,73,73], [172, 172, 172],
								   [310, 310, 310], [489, 489, 489], [708, 705, 708],
								   [1272, 1268, 1272], [1995, 1990, 1990]])

	'''instance_car_moves = np.array(
		[[26, 26, 26], [43, 43, 43], [73, 73, 73], [73, 105, 120],
		 [229, 212, 192], [255, 268, 319], [560, 478, 453], [993, 958, 970],
		 [1465, 1275, 1490]])
	'''
	df = pd.DataFrame(columns=["No. of Nodes", "Computational Time (s)", "No. of Car-moves"])
	for i, n in enumerate(x):
		for j, t in enumerate(construction_heuristic_time_all[i]):
			df = df.append({"No. of Nodes": n, "Computational Time (s)": t, "No. of Car-moves": instance_car_moves[i][j]}, ignore_index=True)

	fig, ax = plt.subplots()
	sns.lineplot(data=df, x="No. of Nodes", y="Computational Time (s)", color="#228b21", label="Computational Time (s) of\nthe Construction Heuristic", ci="sd")
	#sns.lineplot(data=df, x="No. of Nodes", y="Computational Time (s)", color="#228b21", ci="sd")
	ax2 = plt.twinx()
	sns.lineplot(data=df, x="No. of Nodes", y="No. of Car-moves", color="darkorange", ax=ax2, label="No. of Car-moves", ci="sd")
	#sns.lineplot(data=df, x="No. of Nodes", y="No. of Car-moves", color="darkorange", ax=ax2, ci="sd")
	#sns.despine(top=True)


	ax.get_legend().remove()
	lines, labels = ax.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(lines2 + lines, labels2 + labels, loc=0)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	#ax2.legend(labels=["No. of Car-moves", "Computational Time (s)"])
	#ax2.legend(loc="upper center")
	plt.show()


if __name__ == "__main__":

	#plot_construction_heuristic_run_time()
	files = get_files([100, 150, 200])
	handling_times, instance_max_vals, instance_num_carmoves = get_parking_handling_times(files)
	#fraction_list_parking, fraction_list_charging = get_fractional_results()
	plot_values_std(handling_times)#, fraction_list_parking, fraction_list_charging)

	#plot_travel_time_threshold()
	#files = get_files([6, 8, 10, 15, 20, 25, 30, 40, 50])
	#_,_,_ = get_parking_handling_times(files)
	#plot_construction_heuristic_run_time()
#print(num_car_moves)