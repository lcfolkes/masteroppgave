import os
from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Heuristics.feasibility_checker import FeasibilityChecker
from Heuristics.helper_functions_heuristics import remove_car_move, get_best_car_move, \
    check_all_charging_moves_completed
from Heuristics.objective_function import get_objective_function_val
from InstanceGenerator.instance_components import ChargingNode
from src.HelperFiles.helper_functions import load_object_from_file
from src.Gurobi.Model.run_model import run_model
from path_manager import path_to_src

os.chdir(path_to_src)


class ConstructionHeuristic:
    # instance_file = "InstanceFiles/6nodes/6-3-1-1_d.pkl"
    # filename = "InstanceFiles/6nodes/6-3-1-1_b.yaml"

    def __init__(self, instance_file):

        self.world_instance = load_object_from_file(instance_file)
        self.num_scenarios = self.world_instance.num_scenarios
        self.employees = self.world_instance.employees
        self.parking_nodes = self.world_instance.parking_nodes
        self.cars = self.world_instance.cars
        self.unused_car_moves = [[] for _ in range(
            self.num_scenarios)]  # [beta] list of unused car_moves for scenario s (zero index)
        self.assigned_car_moves = {k.employee_id: [[] for _ in range(self.num_scenarios)] for k in
                                   self.employees}  # [gamma_k] dictionary containing ordered list of car_moves,
        # assigned to employee k in scenario s
        self.car_moves = []  # self.world_instance.car_moves
        self.charging_moves = []
        self.parking_moves = []
        self._initialize_car_moves()

        self.available_employees = True
        self.prioritize_charging = True
        self.first_stage = True
        self.charging_moves_second_stage = []
        self.parking_moves_second_stage = []

        self.add_car_moves_to_employees()

    def _initialize_car_moves(self):
        for car in self.world_instance.cars:
            for car_move in car.car_moves:
                self.car_moves.append(car_move)
                for s in range(self.num_scenarios):
                    self.unused_car_moves[s].append(car_move)
                if not car.needs_charging:
                    self.parking_moves.append(car_move)
            if car.needs_charging:
                fastest_time = 1000
                fastest_move = None
                for car_move in car.car_moves:
                    if car_move.handling_time < fastest_time:
                        fastest_time = car_move.handling_time
                        fastest_move = car_move
                self.charging_moves.append(fastest_move)

    def add_car_moves_to_employees(self):
        improving_car_move_exists = True
        while self.available_employees and improving_car_move_exists:
            # print([cm.car_move_id for cm in self.charging_moves])
            # print([[cm.car_move_id for cm in s] for s in self.charging_moves_second_stage])
            # check if charging_moves_list is not empty
            if (self.charging_moves and self.first_stage) or (any(self.charging_moves_second_stage) and
                                                              not self.first_stage):
                self.prioritize_charging = True
                if self.first_stage:
                    car_moves = self.charging_moves
                else:
                    car_moves = self.charging_moves_second_stage


            else:
                self.prioritize_charging = False
                if not check_all_charging_moves_completed(self.num_scenarios, self.employees, self.first_stage,
                                                          self.parking_nodes):
                    print("Instance not solvable. Cannot charge all cars.")
                    break
                if self.first_stage:
                    car_moves = self.parking_moves
                else:
                    car_moves = self.parking_moves_second_stage

            if self.first_stage:
                #### GET BEST CAR MOVE ###
                best_car_move_first_stage = get_best_car_move(self.parking_nodes, self.employees, car_moves,
                                                              self.first_stage, self.prioritize_charging,
                                                              self.num_scenarios)
                # print(best_car_move_first_stage.to_string())
                #### GET BEST EMPLOYEE ###
                best_employee_first_stage = self.get_best_employee(best_car_move=best_car_move_first_stage,
                                                                   employees=self.employees,
                                                                   first_stage=self.first_stage,
                                                                   num_scenarios=self.num_scenarios,
                                                                   world_instance=self.world_instance,
                                                                   prioritize_charging=self.prioritize_charging,
                                                                   charging_moves=self.charging_moves,
                                                                   charging_moves_second_stage=self.charging_moves_second_stage,
                                                                   parking_moves_second_stage=
                                                                   self.parking_moves_second_stage)
                # print(f"employee {best_employee_first_stage}")
                if best_employee_first_stage is not None:
                    #### ADD CAR MOVE TO EMPLOYEE ###
                    self._add_car_move_to_employee(car_moves=car_moves, best_car_move=best_car_move_first_stage,
                                                   best_employee=best_employee_first_stage)

            else:
                #### GET BEST CAR MOVE ###
                best_car_move_second_stage = get_best_car_move(self.parking_nodes, self.employees, car_moves,
                                                               self.first_stage, self.prioritize_charging,
                                                               self.num_scenarios)
                if all(cm is None for cm in best_car_move_second_stage):
                    improving_car_move_exists = False
                #### GET BEST EMPLOYEE ###
                best_employee_second_stage = get_best_employee(best_car_move=best_car_move_second_stage,
                                                               parking_moves=self.parking_moves,
                                                               employees=self.employees,
                                                               first_stage=self.first_stage,
                                                               num_scenarios=self.num_scenarios,
                                                               world_instance=self.world_instance,
                                                               prioritize_charging=self.prioritize_charging,
                                                               charging_moves=self.charging_moves,
                                                               charging_moves_second_stage=
                                                               self.charging_moves_second_stage,
                                                               parking_moves_second_stage=
                                                               self.parking_moves_second_stage)
                #### ADD CAR MOVE TO EMPLOYEE ###
                if best_employee_second_stage is not None:
                    self._add_car_move_to_employee(car_moves=car_moves, best_car_move=best_car_move_second_stage,
                                                   best_employee=best_employee_second_stage)

    def _add_car_move_to_employee(self, car_moves, best_car_move, best_employee):
        if self.first_stage:
            if best_employee is not None:
                print('\nEmployee id', best_employee.employee_id)
                print('Employee node before', best_employee.current_node.node_id)
                print('Employee time before', best_employee.current_time)
                # print('Travel time to start node', best_travel_time_to_car_move)
                print(best_car_move.to_string())
                self.world_instance.add_car_move_to_employee(best_car_move, best_employee)
                for s in range(self.num_scenarios):
                    self.assigned_car_moves[best_employee.employee_id][s].append(best_car_move)
                    self.unused_car_moves[s].remove(best_car_move)
                print('Employee node after', best_employee.current_node.node_id)
                print('Employee time after', best_employee.current_time)
                if self.prioritize_charging:
                    self.charging_moves = remove_car_move(best_car_move,
                                                          car_moves)  # should remove car move and other
                    # car-moves with the same car
                else:
                    self.parking_moves = remove_car_move(best_car_move,
                                                         car_moves)  # should remove car move and other
                    # car-moves with the same car

                self.first_stage = False
                for employee in self.employees:
                    task_num = len(employee.car_moves)
                    if task_num < self.world_instance.first_stage_tasks:
                        self.first_stage = True
                if not self.first_stage:
                    # initialize charging and parking moves for second stage
                    self.charging_moves_second_stage = [self.charging_moves for s in range(self.num_scenarios)]
                    self.parking_moves_second_stage = [self.parking_moves for s in range(self.num_scenarios)]
            else:
                self.available_employees = False
        # Second stage
        else:
            # print(best_car_move_second_stage)
            # If any employee is not note, continue
            if not all(e is None for e in best_employee):
                for s in range(self.num_scenarios):
                    # print(best_employee_second_stage[s].to_string())
                    if best_employee[s] is not None:
                        print('\nEmployee id', best_employee[s].employee_id)
                        print('Scenario', s + 1)
                        print('Employee node before', best_employee[s].current_node_second_stage[s].node_id)
                        print('Employee time before', best_employee[s].current_time_second_stage[s])
                        # print('Travel time to start node', best_travel_time_to_car_move_second_stage[s])
                        print(best_car_move[s].to_string())
                        self.world_instance.add_car_move_to_employee(best_car_move[s], best_employee[s], s)
                        if best_car_move[s] is not None:
                            self.assigned_car_moves[best_employee[s].employee_id][s].append(best_car_move[s])
                            self.unused_car_moves[s].remove(best_car_move[s])
                        print('Employee node after', best_employee[s].current_node_second_stage[s].node_id)
                        print('Employee time after', best_employee[s].current_time_second_stage[s])
                        # When first stage is finished, initialize car_moves to be list of copies of
                        # car_moves (number of copies = num_scenarios)
                        if self.prioritize_charging:
                            self.charging_moves_second_stage[s] = remove_car_move(best_car_move[s],
                                                                                  car_moves[s])
                            # self.charging_moves = remove_car_move(best_car_move[s],
                            #                                     car_moves[s])

                            # should remove car move and other car-moves with the same car
                        else:
                            self.parking_moves_second_stage[s] = remove_car_move(best_car_move[s],
                                                                                 car_moves[s])
                            # should remove car move and other car-moves with the same car
                # print(f"car_moves: {len(car_moves[s])}")
                if not any(self.parking_moves_second_stage):
                    self.available_employees = False
            else:
                self.available_employees = False

    def get_best_employee(self, employees, best_car_move, first_stage, num_scenarios, world_instance,
                          charging_moves_second_stage, parking_moves_second_stage):
        feasibility_checker = FeasibilityChecker(world_instance)
        if first_stage:
            best_employee = None
            best_travel_time_to_car_move = 100
            end_node = best_car_move.start_node
        else:
            best_employee_second_stage = [None for _ in range(num_scenarios)]
            best_travel_time_to_car_move_second_stage = [100 for _ in range(num_scenarios)]
            end_node = [(cm.start_node if cm is not None else cm) for cm in best_car_move]

        best_move_not_legal = True

        for employee in employees:
            task_num = len(employee.car_moves)
            # if first stage and the number of completed task for employee is below the number of tasks in first stage,
            # or if second stage and the number of completed tasks are the same or larger than the number of tasks
            # in first stage
            if first_stage == (task_num < world_instance.first_stage_tasks):
                if first_stage:
                    legal_move = feasibility_checker.check_legal_move(car_move=best_car_move, employee=employee)
                    print(f"legal_move {legal_move}")
                    if legal_move:
                        best_move_not_legal = False
                        start_node = employee.current_node
                        travel_time_to_car_move = world_instance.get_employee_travel_time_to_node(start_node,
                                                                                                  end_node)
                        if travel_time_to_car_move < best_travel_time_to_car_move:
                            best_travel_time_to_car_move = travel_time_to_car_move
                            best_employee = employee

                else:
                    for s in range(num_scenarios):
                        if best_car_move[s] is not None:
                            legal_move = feasibility_checker.check_legal_move(
                                car_move=best_car_move[s], employee=employee, scenario=s)
                            if legal_move:
                                best_move_not_legal = False
                                start_node = employee.current_node_second_stage[s]
                                travel_time_to_car_move = world_instance.get_employee_travel_time_to_node(
                                    start_node, end_node[s])
                                if travel_time_to_car_move < best_travel_time_to_car_move_second_stage[s]:
                                    best_travel_time_to_car_move_second_stage[s] = travel_time_to_car_move
                                    best_employee_second_stage[s] = employee

            # Remove best move if not legal. Else return best employee
        if first_stage:
            if best_move_not_legal:
                if isinstance(best_car_move.end_node, ChargingNode):
                    self.charging_moves.remove(best_car_move)
                else:
                    self.parking_moves.remove(best_car_move)
                return
            else:
                return best_employee
        else:
            if best_move_not_legal:
                for s in range(num_scenarios):
                    if isinstance(best_car_move[s].end_node, ChargingNode):
                        self.charging_moves_second_stage[s] = [cm for cm in charging_moves_second_stage[s] if
                                                               cm != best_car_move[s]]
                    else:
                        self.parking_moves_second_stage[s] = [cm for cm in parking_moves_second_stage[s] if
                                                              cm != best_car_move[s]]
                return
            else:
                return best_employee_second_stage

    def print_solution(self):

        print("-------------- First stage routes --------------")
        for employee in self.employees:
            for car_move in employee.car_moves:
                print(f"employee: {employee.employee_id}, " + car_move.to_string())

        print("-------------- Second stage routes --------------")
        for employee in self.employees:
            if any(employee.car_moves_second_stage):
                for s in range(self.num_scenarios):
                    for car_move in employee.car_moves_second_stage[s]:
                        print(f"employee: {employee.employee_id}, scenario: {s + 1} " + car_move.to_string())


if __name__ == "__main__":
    filename = "InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_a"

    print("\n---- HEURISTIC ----")
    ch = ConstructionHeuristic(filename + ".pkl")
    # try:
    ch.print_solution()
    get_objective_function_val(ch.parking_nodes, ch.employees, ch.num_scenarios)
    print(ch.assigned_car_moves)
    print(ch.unused_car_moves)
    print("\n---- GUROBI ----")
    # gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=True)
    gi = GurobiInstance(filename + ".yaml")
    run_model(gi)
    # except:
    #    print("Instance not solvable")
