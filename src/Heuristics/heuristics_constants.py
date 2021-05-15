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
	ACCEPTED = 13.0

	# DETERMINISM PARAMETER
	DETERMINISM_PARAMETER_WORST = 5
	DETERMINISM_PARAMETER_RELATED = 5
	DETERMINISM_PARAMETER_GREEDY = 3


	# ADAPTIVE WEIGHT LOWER THRESHOLD
	LOWER_THRESHOLD = 0.2

	# DESTROY/REPAIR FACTOR q hat
	DESTROY_REPAIR_FACTOR = (0.10, 0.30)
	#DESTROY_REPAIR_FACTOR = (0.10, 0.40)

	# REWARD DECAY PARAMETER / REACTION FACTOR
	REWARD_DECAY_PARAMETER = 0.1

	# ITERATIONS
	ITERATIONS_ALNS = 15
	ITERATIONS_SEGMENT = 10
	TIME_LIMIT = 300
	FIRST_CHECKPOINT = 100
	SECOND_CHECKPOINT = 200
