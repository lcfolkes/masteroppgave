from abc import ABC, abstractmethod

import random
from Heuristics.DestroyAndRepairHeuristics.destroy import RandomRemoval
from Heuristics.DestroyAndRepairHeuristics.repair import GreedyInsertion
from Heuristics.construction_heuristic_new import ConstructionHeuristic


class LocalSearchOperator(ABC):

	def __init__(self, solution):
		'''
		:param solution: dictionary with employee object as key and a list of car moves as value
		'''
		self.initial_solution = solution
		self.mutated_solution = self._mutate(solution)

	def _mutate(self, solution):
		pass


class IntraMove(LocalSearchOperator):
	'''
    An Intra Move LSO moves a car-move within the list of car-moves for one service employee
    '''

	def __init__(self, solution):
		super().__init__(solution)

	def _mutate(self, solution):
		employees = [k for k, v in solution.items() if len(v) > 1]
		employee = random.choice(employees)
		moves = solution[employee]
		random.shuffle(moves)
		return solution


class InterSwap(LocalSearchOperator):
	'''
    An Inter Swap LSO swaps two car-moves between two service employees.
    '''

	def __init__(self, solution):
		super().__init__(solution)

	def _mutate(self, solution):
		employees = [k for k, v in solution.items() if len(v) > 0]
		chosen_employees = random.sample(employees, 2)
		idx_1 = random.randint(0, len(solution[chosen_employees[0]])-1)
		idx_2 = random.randint(0, len(solution[chosen_employees[1]])-1)
		solution[chosen_employees[1]][idx_2], solution[chosen_employees[0]][idx_1] = \
			solution[chosen_employees[0]][idx_1], solution[chosen_employees[1]][idx_2]

		return solution


class InterMove(LocalSearchOperator):
	'''
    An Inter Move LSO moves a car-move from one service employee to another.
    '''

	def __init__(self, solution):
		super().__init__(solution)


class Inter2Move(LocalSearchOperator):
	'''
    An Inter 2-Move LSO moves two consecutive car-moves from one employee to another employee.
    '''

	def __init__(self, solution):
		super().__init__(solution)


class EjectionInsert(LocalSearchOperator):
	'''
    An Ejection Insert LSO picks a car-move from , and randomly inserts it into gamma
    '''

	def __init__(self, solution, unused_car_moves):
		self.unused_car_moves = unused_car_moves
		super().__init__(solution)


class EjectionRemove(LocalSearchOperator):
	'''
    An Ejection Remove LSO removes a car-move from gamma and place it into the pool of unused car-moves beta.
    '''

	def __init__(self, solution, unused_car_moves):
		self.unused_car_moves = unused_car_moves
		super().__init__(solution)


class EjectionReplace(LocalSearchOperator):
	'''
    An Ejection Replace LSO swaps a car-move from gamma with a car-move from beta associated with the same car.
    In other words, the destinations of a car-move in gamma is changed.
    '''

	def __init__(self, solution, unused_car_moves):
		self.unused_car_moves = unused_car_moves
		super().__init__(solution)


class EjectionSwap(LocalSearchOperator):
	'''
    An Ejection Swap LSO swaps a car-move from gamma with a car-move from beta, associated with
    different cars. An Ejection Swap differs from an Ejection Replace by swapping cars, while an
    Ejection Replace only replace a car-move with another car-move for the same car
    '''

	def __init__(self, solution, unused_car_moves):
		self.unused_car_moves = unused_car_moves
		super().__init__(solution)


if __name__ == "__main__":
	print("\n---- HEURISTIC ----")
	ch = ConstructionHeuristic("./InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_a.pkl")
	ch.add_car_moves_to_employees()
	rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
					   neighborhood_size=2)
	# rr.to_string()
	gi = GreedyInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
						 parking_nodes=ch.parking_nodes, world_instance=ch.world_instance)
	print("Local Search")
	ls = InterSwap(gi.repaired_solution)
