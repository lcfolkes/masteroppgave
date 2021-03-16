from local_search_operator import LocalSearchOperator
from src.InstanceGenerator.instance_components import Employee, CarMove


class EjectionInsert(LocalSearchOperator):

	def __init__(self):
		self.id: int = 6
		self.emloyee: Employee = None
		self.carmove_index: int = 0
		self.carmove_replace: CarMove = None
		self.hash_code: int = 0


	def ejection_insert(self, employee: Employee, car_move_index: int, car_move_replace: CarMove):
		self.emloyee = employee
		self.carmove_index = car_move_index
		self.carmove_replace = car_move_replace
		replace_car_id = car_move_replace.car.car_id
		hash_string = "-" + self.id + employee.employee_id + "" + replace_car_id + car_move_index
		conv = int(hash_string) % 105943
		self.hash_code = int(conv)

