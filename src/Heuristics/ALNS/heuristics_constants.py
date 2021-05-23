import random

class HeuristicsConstants:

	# RELATEDNESS MEASURE
	FROM_NODE_WEIGHT = 0.005
	TO_NODE_WEIGHT = 0.005
	IS_CHARGING_WEIGHT = 0.315
	TRAVEL_DISTANCE_WEIGHT = 0.005
	START_TIME_WEIGHT = 0.05

	# REWARDS
	BEST = 33.0
	BETTER = 9.0
	ACCEPTED = 1.0

	# DETERMINISM PARAMETER
	DETERMINISM_PARAMETER_WORST = 9
	DETERMINISM_PARAMETER_RELATED = 9
	DETERMINISM_PARAMETER_GREEDY = 9


	# ADAPTIVE WEIGHT LOWER THRESHOLD
	LOWER_THRESHOLD = 0.2

	# DESTROY/REPAIR FACTOR q hat
	DESTROY_REPAIR_FACTOR = (0.30, 0.70)
	#DESTROY_REPAIR_FACTOR = (0.10, 0.40)

	# REWARD DECAY PARAMETER / REACTION FACTOR
	REWARD_DECAY_PARAMETER = 0.1

	# ITERATIONS
	ITERATIONS_ALNS = 100
	ITERATIONS_SEGMENT = 50
	TIME_LIMIT = 600
	FIRST_CHECKPOINT = 200
	SECOND_CHECKPOINT = 400

	# AN ACCEPTANCE PERCENTAGE OF 2 MEANS ALL CAR MOVES ARE INCLUDED
	ACCEPTANCE_PERCENTAGE = 2.0
	TRAVEL_TIME_THRESHOLD = 0.7

	PLANNING_PERIOD = 60
	#NUM_FIRST_STAGE_TASKS = 2
