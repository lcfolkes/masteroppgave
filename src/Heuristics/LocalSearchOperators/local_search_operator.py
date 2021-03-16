from abc import ABC, abstractmethod

class LocalSearchOperator(ABC):

	def __init__(self, solution, unused_car_moves):
		self.initial_solution = solution
		self.unused_car_moves = unused_car_moves
		self.mutated_solution = None


class IntraMove(LocalSearchOperator):
	def __init__(self, solution, unused_car_moves):
		super().__init__(solution, unused_car_moves)

class InterMove(LocalSearchOperator):
	def __init__(self, solution, unused_car_moves):
		super().__init__(solution, unused_car_moves)

class Inter2Move(LocalSearchOperator):
	def __init__(self, solution, unused_car_moves):
		super().__init__(solution, unused_car_moves)

class InterSwap(LocalSearchOperator):
	def __init__(self, solution, unused_car_moves):
		super().__init__(solution, unused_car_moves)

class EjectionInsert(LocalSearchOperator):
	def __init__(self, solution, unused_car_moves):
		super().__init__(solution, unused_car_moves)

class EjectionRemove(LocalSearchOperator):
	def __init__(self, solution, unused_car_moves):
		super().__init__(solution, unused_car_moves)

class EjectionReplace(LocalSearchOperator):
	def __init__(self, solution, unused_car_moves):
		super().__init__(solution, unused_car_moves)

class EjectionSwap(LocalSearchOperator):
	def __init__(self, solution, unused_car_moves):
		super().__init__(solution, unused_car_moves)
