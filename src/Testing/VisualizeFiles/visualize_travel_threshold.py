import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from HelperFiles.helper_functions import read_config
from path_manager import path_to_src
os.chdir(path_to_src)


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

def get_handling_times(files):
	handling_times = []
	for f in files:
		cf = read_config(f)
		instance_vals = cf['car_move_handling_time']
		max_val = max(instance_vals)
		fractional_values = list(np.divide(instance_vals, max_val))
		handling_times += fractional_values
	return handling_times

def plot_values(handling_times):
	from mpl_toolkits.axisartist.axislines import SubplotZero
	plt.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"font.sans-serif": ["Computer Modern Roman"]})
	ax = sns.displot(data=handling_times, kind="kde", bw_adjust=0.1)
	ax.set(xlabel='Fraction of longest travel distance')
	plt.gcf().subplots_adjust(bottom=0.15)

	plt.show()


if __name__ == "__main__":
	files = get_files([30, 40, 50])
	handling_times = get_handling_times(files)
	plot_values(handling_times)
