from abc import ABC, abstractmethod

class LocalSearchOperator(ABC):

	@abstractmethod
	def get_id(self) -> int:
		pass

	@abstractmethod
	def equals(self, object) -> bool:
		pass

	@abstractmethod
	def hash_code(self) -> int:
		pass

	@abstractmethod
	#def compareTo(self, new_candidate: LocalSearchOperator) -> int:
	def compare_to(self, new_candidate) -> int:
		pass