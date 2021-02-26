import os
import sys
import gurobipy as gp
import pandas as pd
from src.Gurobi.Model.gurobi_instance import GurobiInstance
from src.HelperFiles.helper_functions import write_gurobi_results_to_file


def run_model(model, stochastic=True, reset=False):
	m = model.m
	# Optimize Model
	m.write("out.attr")
	try:
		if reset:
			# disable warm start
			m.reset(0)
		m.optimize()
		print("Runtime: ", m.Runtime)

		if m.solCount == 0:
			print("Model is infeasible")
			m.computeIIS()
			m.write("model_iis.ilp")

		### Print node states ###
		# print("--- INITIAL NODE STATES ---")
		# for i in range(len(PARKING_NODES)):
		#     if stochastic:
		#         print("node: {0}, pstate: {1}, cstate: {2}, istate: {3}, requests: {4}, deliveries: {5}".format(
		#             i+1, cf.pStateP[i],cf.cStateP[i], IDEAL_STATE[i+1], cf.demandP[i], cf.deliveriesP[i]))
		#     else:
		#         print("node: {0}, pstate: {1}, cstate: {2}, istate: {3}, requests: {4:.2f}, deliveries: {5:.2f}".format(
		#             i+1, cf.pStateP[i],cf.cStateP[i], IDEAL_STATE[i+1], round(sum(cf.demandP[i])/len(cf.demandP[i])), round(sum(cf.deliveriesP[i])/len(cf.deliveriesP[i]))))

		# for k,v in EMPLOYEE_START_LOCATION.items():
		#    print("Employee: {}, Node: {}".format(k,v))

		# print("\n--- RESULTS ---")
		x_count, y_count, z_count, w_count = 1, 1, 1, 1
		# for v in m.getVars():
		#    if(v.varName[0] == 't'):
		#       print(v.varName, v.x)
		#         if (v.varName[0] == 'x' and x_count > 0):
		#             print("x[k,r,m,s]: Service employee k performs car-move r as task number m in scenario s")
		#             x_count = 0
		#         if (v.varName[0] == 'y' and y_count > 0):
		#             print("y[i]: Number of cars in node i by the beginning of the second stage")
		#             y_count = 0
		#         elif (v.varName[0] == 'z' and z_count > 0):
		#             print("z[i,s]: Number of customer requests served in second stage in node i in scenario s")
		#             z_count = 0
		#         elif (v.varName[0] == 'w' and w_count > 0):
		#             print("w[i,s]: Number of cars short of ideal state in node i in scenario s")
		#             w_count = 0
		#
		#    if (v.varName[0] == 'x' and v.x > 0):
		#        l = list(map(int, (v.varName[2:-1].split(','))))
		#        print("{0} {1} ({2} --> {3})".format(v.varName, int(v.x),CARMOVE_ORIGIN[l[1]],CARMOVE_DESTINATION[l[1]]))
		#         elif (v.varName[0] != 'x'):
		#             print('%s %g' % (v.varName, v.x))

		print('Obj: %g' % m.objVal)

		# print("--- ROUTES AND SCHEDULES ---")
		# print("Number of tasks in first stage : {}".format(len(TASKS_FIRST_STAGE)))
		# print("Planning period: {}".format(PLANNING_PERIOD))
		employee_routes = {new_list: [] for new_list in model.EMPLOYEES}
		# print(employee_routes)
		for v in m.getVars():
			if (v.varName[0] == 'x' and v.x > 0):
				x = list(map(int, (v.varName[2:-1].split(','))))
				employee_routes[x[0]].append(x[1:])

		# sort by ascending task number, then scenario
		for k in model.EMPLOYEES:
			employee_routes[k] = sorted(employee_routes[k], key=lambda x: (x[1], x[2]), reverse=False)

		# initialize dataframes of routes and schedules
		df_firststage_routes = pd.DataFrame(
			columns=["Employee", "Task", "Route", "Travel Time to Task", "Start time", "Relocation Time",
					 "End time"])
		df_secondstage_routes = pd.DataFrame(
			columns=["Employee", "Task", "Scenario", "Route", "Travel Time to Task", "Start time", "Relocation Time",
					 "End time"])

		# calculate travel times
		for k in model.EMPLOYEES:
			# employee_routes {k: [r,m,s]}

			# initialize time and scenario dictionaries
			# time for scenario s, t[x[2]]
			t = {s: 0 for s in [x[2] for x in employee_routes[k]]}
			destination_node = {s: 0 for s in [x[2] for x in employee_routes[k]]}
			for x in employee_routes[k]:
				# First task
				if x[1] == 1:
					# Initialize start time of employee
					t[x[2]] += model.START_TIME_EMPLOYEE[k]
					# Bike travel time between start location of employee and origin node of first task
					tt = model.TRAVEL_TIME[model.EMPLOYEE_START_LOCATION[k] - 1][model.CARMOVE_ORIGIN[x[0]] - 1]
				else:
					# Bike travel time between previous node and current node
					tt = model.TRAVEL_TIME[destination_node[x[2]] - 1][model.CARMOVE_ORIGIN[x[0]] - 1]

				# Start time of current task
				t[x[2]] += tt

				# Relocation time of car-move
				rt = model.RELOCATION_TIME[x[0]]

				## Add routes and schedules to dataframe
				# first stage routes
				if x[1] <= len(model.TASKS_FIRST_STAGE):
					if (x[2] == 1):
						first_stage_row = [k, x[1], (model.CARMOVE_ORIGIN[x[0]], model.CARMOVE_DESTINATION[x[0]]), tt, t[x[2]], rt,
										   t[x[2]] + rt]
						df_firststage_routes.loc[len(df_firststage_routes)] = first_stage_row

				# second stage routes
				else:
					second_stage_row = [k, x[1], x[2], (model.CARMOVE_ORIGIN[x[0]], model.CARMOVE_DESTINATION[x[0]]), tt, t[x[2]],
										rt, t[x[2]] + rt]
					df_secondstage_routes.loc[len(df_secondstage_routes)] = second_stage_row
				# print("Task: {0}, Scenario: {1}, ({2} --> {3}), Travel Time to Task: {4:.1f}, Start time: {5:.1f}, "
				#      "Relocation Time: {6:.1f}, End time: {7:.1f}".format(
				#    x[1], x[2], CARMOVE_ORIGIN[x[0]], CARMOVE_DESTINATION[x[0]], tt, t[x[2]], rt, t[x[2]]+rt))

				# End time
				t[x[2]] += rt

				# Update last visited  node in scenario s
				destination_node[x[2]] = model.CARMOVE_DESTINATION[x[0]]

		# TODO: sort dataframe with ascending endtime in addition to Employee and, Task and scenario
		pd.set_option('display.width', 320)
		pd.set_option('display.max_columns', 10)
		# print("-------------- First stage routes --------------")
		# print(df_firststage_routes)
		# print("\n-------------- Second stage routes --------------")
		# print(df_secondstage_routes)

		return m, df_firststage_routes, df_secondstage_routes, model.PLANNING_PERIOD

	except gp.GurobiError as e:
		print('Error code ' + str(e.errno) + ': ' + str(e))

	except AttributeError as e:
		print(e)
		print('Encountered an attribute error')

def run_test_instance(file: str):
	stochastic_model = GurobiInstance(file, model_type=0)
	stochastic_model.m.setParam('TimeLimit', 30 * 60)
	z_stochastic, firststageroutes, secondstageroutes, planningperiod = run_model(stochastic_model, stochastic=True, reset=True)
	deterministic_model = GurobiInstance(file, model_type=1)
	z_deterministic = run_model(deterministic_model, stochastic=False)[0]
	# EEV: expectation of expected value. Deterministic first stage values are fixed and then run on the stochastic Model
	eev_model = GurobiInstance(file, model_type=3, input_model=deterministic_model)
	z_eev = run_model(eev_model, stochastic=True)[0]
	vss = z_stochastic.ObjVal - z_eev.ObjVal
	print("Value of Stochastic Solution (VSS)", vss)

	# Write results to file
	out_str = ""
	out_str += "-------------- First stage routes --------------\n"
	out_str += firststageroutes.to_string(header=True, index=True)
	out_str += "\n\n-------------- Second stage routes --------------\n"
	out_str += secondstageroutes.to_string(header=True, index=True)
	out_str += "\n\nObjVal Stochastic: " + str(z_stochastic.ObjVal) + "\nEEV: " + str(z_eev.ObjVal) + "\nVSS: " + str(vss) + \
			 "\nPercentage Improvement: " + str(((z_stochastic.ObjVal / z_eev.ObjVal) - 1) * 100) + "\nRuntime stochastic: " \
			 + str(z_stochastic.Runtime) + "\nPlanning period: " + str(planningperiod)

	write_gurobi_results_to_file(filename=file, out_str=out_str)

	'''
	# Write results to excel file
	firststageroutes = firststageroutes.values[:, 2]
	secondstageroutes = secondstageroutes.values[:, 3]
	df_results = pd.DataFrame({
		"Test instance": [file[:-4]],  # [filepath[-12:-4]],
		"ObjVal Stochastic": [z_stochastic.ObjVal],
		"ObjBound Stochastic": [z_stochastic.ObjBound],
		"EEV": [z_eev.ObjVal],
		"VSS": [vss],
		"Percentage Improvement": [(z_stochastic.ObjVal / z_eev.ObjVal - 1) * 100],
		"Runtime Stochastic": [z_stochastic.Runtime],
		"First-stage Routes": [firststageroutes],
		"Second-stage Routes": [secondstageroutes]})
	
	append_df_to_excel(filename=file, df=df_results)'''

def main():
	print(sys.path)  # os.getcwd())

	directory = "../../InstanceGenerator/InstanceFiles/6nodes/"
	files = []
	for filename in os.listdir(directory):
		files.append(os.path.join(directory, filename))

	for file in files:
		print(file)
		run_test_instance(file)


main()
