import os

import numpy as np

import Heuristics.heuristics_constants
from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Heuristics.feasibility_checker import FeasibilityChecker
from Heuristics.helper_functions_heuristics import remove_all_car_moves_of_car_in_car_move, \
    get_first_stage_solution_and_removed_moves, get_first_stage_solution, get_first_and_second_stage_solution, \
    get_assigned_car_moves, get_first_stage_solution_list_from_dict, get_separate_assigned_car_moves, \
    get_first_stage_solution_list_from_solution
from Heuristics.best_objective_function import ObjectiveFunction
from HelperFiles.helper_functions import load_object_from_file
from Heuristics.helper_functions_heuristics import copy_unused_car_moves_2d_list
from Gurobi.Model.run_model import run_model
import pandas as pd
from path_manager import path_to_src

os.chdir(path_to_src)


class ConstructionHeuristic:
    # instance_file = "InstanceFiles/6nodes/6-3-1-1_d.pkl"
    # filename = "InstanceFiles/6nodes/6-3-1-1_b.yaml"

    def __init__(self, instance_file, acceptance_percentage):
        self.acceptance_percentage = acceptance_percentage
        self.instance_file = instance_file
        self.world_instance = load_object_from_file(instance_file)
        self.world_instance.initialize_relevant_car_moves(acceptance_percentage)
        self.objective_function = ObjectiveFunction(self.world_instance)
        self.world_instance.planning_period = Heuristics.heuristics_constants.HeuristicsConstants.PLANNING_PERIOD
        self.feasibility_checker = FeasibilityChecker(self.world_instance)
        self.num_scenarios = self.world_instance.num_scenarios
        #self.num_first_stage_tasks = Heuristics.heuristics_constants.HeuristicsConstants.NUM_FIRST_STAGE_TASKS
        self.num_first_stage_tasks = self.world_instance.first_stage_tasks
        self.employees = self.world_instance.employees
        self.parking_nodes = self.world_instance.parking_nodes
        self.cars = self.world_instance.cars
        self.unused_car_moves = [[] for _ in range(
            self.num_scenarios)]  # [beta] list of unused car_moves for scenario s (zero index)
        self.assigned_car_moves = {k: [[] for _ in range(self.num_scenarios)] for k in
                                   self.employees}  # [gamma_k] dictionary containing ordered list of car_moves,
        # assigned to employee k in scenario s
        self.employees_dict = {k.employee_id: k for k in self.employees}
        self.car_moves_dict = {}
        self.car_moves_first_stage = []  # self.world_instance.car_moves
        self.car_moves_second_stage = [[] for _ in range(self.num_scenarios)]
        self._initialize_car_moves()
        self.available_employees = True
        self.first_stage = True
        # self.hash_key = 0

        # self.add_car_moves_to_employees()

    def get_obj_val(self, true_objective=True, both=False):
        # return get_objective_function_val(parking_nodes=self.parking_nodes, employees=self.employees,
        #                                  num_scenarios=self.num_scenarios, true_objective=true_objective, both=both)
        if both:
            return self.objective_function.true_objective_value, self.objective_function.heuristic_objective_value
        if true_objective:
            return self.objective_function.true_objective_value
        else:
            return self.objective_function.heuristic_objective_value

    def _initialize_for_rebuild(self):
        self.unused_car_moves = [[] for _ in range(
            self.num_scenarios)]  # [beta] list of unused car_moves for scenario s (zero index)
        for k in self.employees:
            '''
            print(f"emp {k.employee_id}")
            print([cm.car_move_id for cm in k.car_moves])
            print([[cm.car_move_id for cm in scenario] for scenario in k.car_moves_second_stage])
            '''
            k.reset()
        self.assigned_car_moves = {k: [[] for _ in range(self.num_scenarios)] for k in
                                   self.employees}  # [gamma_k] dictionary containing ordered list of car_moves,
        # assigned to employee k in scenario s
        self.car_moves_first_stage = []  # self.world_instance.car_moves
        self.car_moves_second_stage = [[] for _ in range(self.num_scenarios)]
        self._initialize_car_moves()
        self._initialize_charging_nodes()
        self.available_employees = True
        self.first_stage = True
        self.objective_function = ObjectiveFunction(self.world_instance)

    @property
    def hash_key(self):
        hash_dict = {}
        for k, v in self.assigned_car_moves.items():
            emp_moves = []
            for s in v:
                scen_moves = []
                for cm in s:
                    scen_moves.append(cm.car_move_id)
                emp_moves.append(scen_moves)
            hash_dict[k.employee_id] = emp_moves

        # self.hash_key = hash(str(hash_dict))
        return hash(str(hash_dict))

    def rebuild(self, solution, stage="first", verbose=False, optimize=True):
        #print("\n --- REBUILD ---")
        self._initialize_for_rebuild()

        if stage == "first":
            first_stage_solution = solution
            for employee_obj, car_move_objs in first_stage_solution.items():
                emp = self.employees_dict[employee_obj.employee_id]
                for cm_obj in car_move_objs:
                    cm = self.car_moves_dict[cm_obj.car_move_id]
                    self._add_car_move_to_employee(best_car_move=cm, best_employee=emp)
            if optimize:
                self.construct()

        else:
            first_stage_solution, second_stage_solution = get_first_and_second_stage_solution(
                solution, self.world_instance.first_stage_tasks)

            ### FIRST STAGE ###
            for employee_obj, car_move_objs in first_stage_solution.items():
                for cm_obj in car_move_objs:
                    cm = self.car_moves_dict[cm_obj.car_move_id]
                    self._add_car_move_to_employee(best_car_move=cm, best_employee=employee_obj)

            ### SECOND STAGE ###
            #self.first_stage = False
            self.car_moves_second_stage = [[cm for cm in self.car_moves_first_stage] for _ in range(self.num_scenarios)]
            #print(second_stage_solution.items())
            for employee_obj, car_moves_scenarios in second_stage_solution.items():
                self._add_car_move_to_employee_from_dict(employee=employee_obj, car_moves_scenarios=car_moves_scenarios)
            if optimize:
                self.construct(stage="second")


    def _initialize_car_moves(self):
        for car_move in self.world_instance.relevant_car_moves:
            self.car_moves_first_stage.append(car_move)
            self.car_moves_dict[car_move.car_move_id] = car_move
            for s in range(self.num_scenarios):
                self.unused_car_moves[s].append(car_move)


    def _initialize_charging_nodes(self):
        for cn in self.world_instance.charging_nodes:
            cn.reset()

    def construct(self, verbose=False, stage=None):
        if verbose:
            print("------- CONSTRUCTION HEURISTIC -------\n")
        improving_car_move_exists_first_stage = True
        improving_car_move_exists_second_stage = True
        second_stage_move_counter = 0
        first_stage_move_counter = 0
        while self.available_employees and (improving_car_move_exists_first_stage or improving_car_move_exists_second_stage):

            if self.first_stage:
                #### GET BEST CAR MOVE ###

                best_car_move_first_stage = self._get_best_car_move()

                if best_car_move_first_stage is None:
                    improving_car_move_exists_first_stage = False
                    self.first_stage = False
                    if stage != "second":
                        self.car_moves_second_stage = [[cm for cm in self.car_moves_first_stage] for _ in range(self.num_scenarios)]
                    continue
                #### GET BEST EMPLOYEE ###
                best_employee_first_stage = self._get_best_employee(best_car_move_first_stage)

                if best_employee_first_stage is not None:
                    ### ADD CAR MOVE TO EMPLOYEE ###
                    self._add_car_move_to_employee(best_car_move=best_car_move_first_stage,
                                                   best_employee=best_employee_first_stage)
                    first_stage_move_counter += 1
                    if verbose:
                        print(f"{first_stage_move_counter} first stage insertions completed\n")

            else:
                ### GET BEST CAR MOVE ###
                best_car_move_second_stage = self._get_best_car_move()
                # print(f"Best car_move second stage: {[cm.car_move_id for cm in best_car_move_second_stage]}")
                if all(cm is None for cm in best_car_move_second_stage):
                    #print([cm for cm in best_car_move_second_stage])
                    improving_car_move_exists_second_stage = False

                ### GET BEST EMPLOYEE ###
                best_employee_second_stage = self._get_best_employee(best_car_move=best_car_move_second_stage)

                ### ADD CAR MOVE TO EMPLOYEE ###
                if not all(cm is None for cm in best_employee_second_stage):
                    self._add_car_move_to_employee(best_car_move=best_car_move_second_stage,
                                                   best_employee=best_employee_second_stage)

                    # Visual feedback for construction process
                    second_stage_move_counter += 1
                    added_scenarios = []
                    for i in range(len(best_car_move_second_stage)):
                        if best_car_move_second_stage[i] and best_employee_second_stage[i]:
                            added_scenarios.append(1)
                        else:
                            added_scenarios.append(0)


                    if verbose:
                        scenarios_with_insertion = [i for i, x in enumerate(added_scenarios) if x == 1]
                        print(f"Second stage insertion number {second_stage_move_counter}:")
                        print("{} second stage car moves added in scenarios {}\n".format(
                            len(scenarios_with_insertion), ([i + 1 for i in scenarios_with_insertion])))

    def _get_best_car_move(self):
        # FIRST STAGE
        if self.first_stage:
            best_car_move_first_stage = None
            best_obj_val_first_stage = self.objective_function.heuristic_objective_value

            # print("Iteration")
            for car_move in self.car_moves_first_stage:
                #car_move = self.car_moves_dict[cm_id]
                if car_move.is_charging_move:
                    # Checking if charging node has space for another car
                    if car_move.end_node.capacity == car_move.end_node.num_charging[0]:
                        # TODO: remove car_moves with this destination
                        continue


                obj_val = self.objective_function.evaluate(added_car_moves=[car_move], mode="heuristic")

                if obj_val > best_obj_val_first_stage:
                    best_obj_val_first_stage = obj_val
                    best_car_move_first_stage = car_move


            return best_car_move_first_stage


        # SECOND STAGE
        else:
            best_car_move_second_stage = [None for _ in range(self.num_scenarios)]
            best_obj_val_second_stage = [self.objective_function.heuristic_objective_value for _ in range(self.num_scenarios)]

            for s in range(self.num_scenarios):
                # Parking moves second stage
                for r in range(len(self.car_moves_second_stage[s])):
                    #car_move = self.car_moves_dict[self.car_moves_second_stage[s][r]]
                    car_move = self.car_moves_second_stage[s][r]
                    if car_move.is_charging_move:
                        # Checking if charging node has space for another car

                        if car_move.end_node.capacity == \
                                car_move.end_node.num_charging[s]:
                            # TODO: remove car_moves with this destination for this scenario
                            continue


                    obj_val = self.objective_function.evaluate(added_car_moves=[car_move],
                                                               scenario=s, mode="heuristic")
                    '''
                    if self.car_moves_second_stage[s][r].car_move_id == 8:
                        if s == 4:
                            print("start", self.car_moves_second_stage[s][r].start_node.node_id)
                            print("end", self.car_moves_second_stage[s][r].end_node.node_id)

                            print("obj_val", obj_val)
                            print("best obj val", best_obj_val_second_stage[s])
                    '''


                    if obj_val > best_obj_val_second_stage[s]:
                        '''
                        if self.car_moves_second_stage[s][r].is_charging_move:

                            # Checking if charging node has space for another car
                            if self.car_moves_second_stage[s][r].end_node.capacity == \
                                    self.car_moves_second_stage[s][r].end_node.num_charging[s]:
                                continue

                            else:

                                best_obj_val_second_stage[s] = obj_val
                                best_car_move_second_stage[s] = self.car_moves_second_stage[s][r]
                                # car_moves[s][r].end_node.add_car(scenario=s)
                        else:
                        '''
                        best_obj_val_second_stage[s] = obj_val
                        best_car_move_second_stage[s] = car_move

                    elif obj_val == best_obj_val_second_stage[s]:
                        if car_move.handling_time < best_car_move_second_stage[s].handling_time:
                            best_obj_val_second_stage[s] = obj_val
                            best_car_move_second_stage[s] = car_move
                    '''
                        best_car_move_second_stage[s] = self.car_moves_second_stage[s][r]
                        if self.car_moves_second_stage[s][r].is_charging_move:

                            # Checking if charging node has space for another car
                            if self.car_moves_second_stage[s][r].end_node.capacity == \
                                    self.car_moves_second_stage[s][r].end_node.num_charging[s]:
                                continue

                            elif self.car_moves_second_stage[s][r].handling_time < best_car_move_second_stage.handling_time:
                                best_obj_val_second_stage[s] = obj_val
                                best_car_move_second_stage[s] = self.car_moves_second_stage[s][r]
                                # car_moves[s][r].end_node.add_car(scenario=s)
                        elif self.car_moves_second_stage[s][r].handling_time < best_car_move_second_stage[s].handling_time:
                            best_obj_val_second_stage[s] = obj_val
                            best_car_move_second_stage[s] = self.car_moves_second_stage[s][r]
                    '''

            # print([round(o,2) for o in best_obj_val_second_stage])
            return best_car_move_second_stage

    def _get_best_employee(self, best_car_move):
        if self.first_stage:
            best_employee_first_stage = None
            best_travel_time_to_car_move = 100
        else:
            best_employee_second_stage = [None for _ in range(self.num_scenarios)]
            best_travel_time_to_car_move_second_stage = [100 for _ in range(self.num_scenarios)]

        for employee in self.employees:
            task_num = len(employee.car_moves)
            # if first stage and the number of completed task for employee is below the number of tasks in first stage,
            # or if second stage and the number of completed tasks are the same or larger than the number of tasks
            # in first stage
            if self.first_stage == (task_num < self.world_instance.first_stage_tasks):
                if self.first_stage:
                    legal_move, travel_time_to_car_move = self.feasibility_checker.check_legal_move(
                        car_move=best_car_move, employee=employee, get_employee_travel_time=True)
                    # print(f"legal_move {legal_move}\n{best_car_move.to_string()}")
                    if legal_move and travel_time_to_car_move < best_travel_time_to_car_move:
                        best_travel_time_to_car_move = travel_time_to_car_move
                        best_employee_first_stage = employee

                else:
                    for s in range(self.num_scenarios):
                        if best_car_move[s] is not None:
                            legal_move, travel_time_to_car_move = self.feasibility_checker.check_legal_move(
                                car_move=best_car_move[s], employee=employee, scenario=s, get_employee_travel_time=True)
                            if legal_move and (travel_time_to_car_move < best_travel_time_to_car_move_second_stage[s]):
                                best_travel_time_to_car_move_second_stage[s] = travel_time_to_car_move
                                best_employee_second_stage[s] = employee

        # Remove best move if not legal. Else return best employee
        if self.first_stage:
            if best_employee_first_stage is None and best_car_move is not None:
                #self.car_moves.remove(best_car_move.car_move_id)
                self.car_moves_first_stage.remove(best_car_move)
            return best_employee_first_stage
        else:
            for s, emp in enumerate(best_employee_second_stage):
                if emp is None and best_car_move[s] is not None:
                    #self.car_moves_second_stage[s].remove(best_car_move[s].car_move_id) #[cm for cm in self.car_moves_second_stage[s] if cm != best_car_move[s]]
                    self.car_moves_second_stage[s].remove(best_car_move[s])
            return best_employee_second_stage

    def _add_car_move_to_employee(self, best_car_move, best_employee):
        if self.first_stage:
            if best_employee is not None:

                self.world_instance.add_car_move_to_employee(best_car_move, best_employee)
                self.objective_function.update(added_car_moves=[best_car_move])

                for s in range(self.num_scenarios):
                    self.assigned_car_moves[best_employee][s].append(best_car_move)
                    self.unused_car_moves[s].remove(best_car_move)
                self.car_moves_first_stage = remove_all_car_moves_of_car_in_car_move(
                    best_car_move, self.car_moves_first_stage)
                '''
                self.first_stage = False
                for employee in self.employees:
                    task_num = len(employee.car_moves)
                    if task_num < self.world_instance.first_stage_tasks:
                        self.first_stage = True
                if not self.first_stage:
                    # initialize charging and parking moves for second stage
                    self.car_moves_second_stage = [[cm for cm in self.car_moves] for _ in range(self.num_scenarios)]
                '''
            #else:
            #    self.available_employees = False
        # Second stage
        else:
            # print(best_car_move_second_stage)
            # If any employee is not note, continue
            if not all(e is None for e in best_employee):
                for s in range(self.num_scenarios):
                    # print(best_employee_second_stage[s].to_string())
                    if best_employee[s] is not None:
                        if best_car_move[s] is not None:
                            self.world_instance.add_car_move_to_employee(best_car_move[s], best_employee[s], s)
                            self.objective_function.update(added_car_moves=[best_car_move[s]], scenario=s)
                            self.assigned_car_moves[best_employee[s]][s].append(best_car_move[s])
                            #self.unused_car_moves[s].remove(best_car_move[s].car_move_id)
                            try:
                                self.unused_car_moves[s].remove(best_car_move[s])
                            except:
                                print("scenario", s)
                                print("emp", best_employee[s].employee_id)
                                print("cm", best_car_move[s].car_move_id)
                                raise

                        # When first stage is finished, initialize car_moves to be list of copies of
                        # car_moves (number of copies = num_scenarios)
                        self.car_moves_second_stage[s] = remove_all_car_moves_of_car_in_car_move(
                            best_car_move[s], self.car_moves_second_stage[s])

                # print(f"car_moves: {len(car_moves[s])}")
                #if not any(self.car_moves_second_stage):
                #    self.available_employees = False
            #else:
            #    self.available_employees = False

    def _add_car_move_to_employee_from_dict(self, employee, car_moves_scenarios):
        for s, car_moves in enumerate(car_moves_scenarios):
            for car_move in car_moves:
                self.world_instance.add_car_move_to_employee(car_move, employee, s)
                self.objective_function.update(added_car_moves=[car_move], scenario=s)
                self.assigned_car_moves[employee][s].append(car_move)
                self.unused_car_moves[s].remove(car_move)
                self.car_moves_second_stage[s] = remove_all_car_moves_of_car_in_car_move(
                    car_move, self.car_moves_second_stage[s])

    def print_solution(self):
        true_obj_val, heuristic_obj_val = self.get_obj_val(both=True)

        # Computing number of charging moves performed
        num_charging_moves = 0

        for emp in self.assigned_car_moves.keys():
            for cm in self.assigned_car_moves[emp][0][:self.world_instance.first_stage_tasks]:
                if cm.is_charging_move:
                    num_charging_moves += 1

        for emp in self.assigned_car_moves.keys():
            for scenario in self.assigned_car_moves[emp]:
                for cm in scenario[self.world_instance.first_stage_tasks:]:
                    if cm.is_charging_move:
                        num_charging_moves += round(1 / self.num_scenarios, 2)

        num_charging_moves = round(num_charging_moves, 2)

        # Computing number of cars in need of charging
        cars_in_need = 0
        for car in self.world_instance.cars:
            if car.needs_charging:
                cars_in_need += 1

        df_firststage_routes = pd.DataFrame(
            columns=["           Type", "  Employee", "  Task number", "  Car Move", "  Car ID", "  Employee Route",
                     "  Start Time", "  Travel Time to Task", "  Relocation Time",
                     "  End time"])
        df_secondstage_routes = pd.DataFrame(
            columns=["Scenario", "  Type", "  Employee", "  Task number", "  Car Move", "  Car ID", "  Employee Route",
                     "  Start Time", "  Travel Time to Task", "  Relocation Time",
                     "  End time"])

        print("\n")
        print("----------- CONSTRUCTION HEURISTIC SOLUTION -----------\n")
        print(f"True objective value: {round(self.objective_function.true_objective_value, 2)}")
        print(f"Heuristic objective value: {round(self.objective_function.heuristic_objective_value, 2)}")
        print(f"\nNumber of charging moves performed: {num_charging_moves}/{cars_in_need}")
        # print(f"Number of cars charged: {num_charging_moves}\n")

        # print("-------------- First stage routes --------------")
        for employee in self.employees:
            # print("---- Employee {} ----".format(employee.employee_id))
            for car_move in employee.car_moves:
                car_move_type = "C" if car_move.is_charging_move else "P"
                if employee.car_moves.index(car_move) == 0:
                    start_node_id = employee.start_node.node_id
                else:
                    start_node_id = employee.car_moves[employee.car_moves.index(car_move) - 1].end_node.node_id

                first_stage_row = [car_move_type, employee.employee_id, employee.car_moves.index(car_move) + 1,
                                   car_move.car_move_id, car_move.car.car_id,
                                   f"{start_node_id} -> {car_move.start_node.node_id} -> {car_move.end_node.node_id}",
                                   round(employee.start_times_car_moves[employee.car_moves.index(car_move)]
                                         - employee.travel_times_to_car_moves[employee.car_moves.index(car_move)], 2),
                                   round(employee.travel_times_to_car_moves[employee.car_moves.index(car_move)], 2),
                                   round(employee.car_moves[employee.car_moves.index(car_move)].handling_time, 2),
                                   round(employee.start_times_car_moves[employee.car_moves.index(car_move)] + car_move.handling_time, 2)
                                   ]
                df_firststage_routes.loc[len(df_firststage_routes)] = first_stage_row
                '''
                print(
                    f"    {car_move_type}: Employee: {employee.employee_id}, Task nr: {employee.car_moves.index(car_move) + 1}, "
                    + car_move.to_string() + ", "
                    + f"Start node: {start_node_id}, "
                    + f"Start time: {round(employee.start_times_car_moves[employee.car_moves.index(car_move)] - employee.travel_times_car_moves[employee.car_moves.index(car_move)], 2)}, "
                    + f"Travel time to move: {round(employee.travel_times_car_moves[employee.car_moves.index(car_move)], 2)}, "
                    + f"Handling time: {round(employee.car_moves[employee.car_moves.index(car_move)].handling_time, 2)}, "
                    + f"Time after: {round(car_move.start_time + car_move.handling_time, 2)}\n")
                '''
        # print("-------------- Second stage routes --------------")
        for s in range(self.num_scenarios):
            # print("---- Scenario {} ----".format(s + 1))
            for employee in self.employees:
                if employee.car_moves_second_stage:
                    if any(employee.car_moves_second_stage[s]):
                        # print("     ---- Employee {} ----".format(employee.employee_id))
                        for car_move in employee.car_moves_second_stage[s]:
                            car_move_type = "C" if car_move.is_charging_move else "P"
                            if employee.car_moves_second_stage[s].index(car_move) == 0:
                                start_node_id = employee.car_moves[-1].end_node.node_id
                            else:
                                start_node_id = employee.car_moves_second_stage[s][
                                    employee.car_moves_second_stage[s].index(car_move) - 1].end_node.node_id

                            second_stage_row = [s + 1, car_move_type, employee.employee_id,
                                                employee.car_moves_second_stage[s].index(car_move) + 1 + len(
                                                    employee.car_moves),
                                                car_move.car_move_id, car_move.car.car_id,
                                                f"{start_node_id} -> {car_move.start_node.node_id} -> {car_move.end_node.node_id}",
                                                round(employee.start_times_car_moves_second_stage[s][
                                                          employee.car_moves_second_stage[s].index(car_move)]
                                                      - employee.travel_times_to_car_moves_second_stage[s][
                                                          employee.car_moves_second_stage[s].index(car_move)], 2),
                                                round(employee.travel_times_to_car_moves_second_stage[s][
                                                          employee.car_moves_second_stage[s].index(car_move)], 2),
                                                round(employee.car_moves_second_stage[s][
                                                          employee.car_moves_second_stage[s].index(
                                                              car_move)].handling_time, 2),
                                                round(employee.start_times_car_moves_second_stage[s][
                                                          employee.car_moves_second_stage[s].index(car_move)] + car_move.handling_time, 2)]
                            df_secondstage_routes.loc[len(df_secondstage_routes)] = second_stage_row

                            '''
                            # print(f"{car_move_type}: employee: {employee.employee_id}, scenario: {s + 1} " + car_move.to_string())
                            print(
                                f"        {car_move_type}: Employee: {employee.employee_id}, Task nr: {employee.car_moves_second_stage[s].index(car_move) + 1 + len(employee.car_moves)}, "
                                + car_move.to_string() + ", "
                                + f"Start node: {employee.car_moves_second_stage[s][employee.car_moves_second_stage[s].index(car_move)-1].end_node.node_id}, "
                                + f"Start time: {round(employee.start_times_car_moves_second_stage[s][employee.car_moves_second_stage[s].index(car_move)] - employee.travel_times_car_moves_second_stage[s][employee.car_moves_second_stage[s].index(car_move)], 2)}, "
                                + f"Travel time to move: {round(employee.travel_times_car_moves_second_stage[s][employee.car_moves_second_stage[s].index(car_move)], 2)}, "
                                + f"Handling time: {round(employee.car_moves_second_stage[s][employee.car_moves_second_stage[s].index(car_move)].handling_time, 2)}, "
                                + f"Time after: {round(car_move.start_times_second_stage[s] + car_move.handling_time, 2)}\n")
                            '''

        pd.set_option('display.width', 500)
        pd.set_option('display.max_columns', 15)

        print(
            "--------------------------------------------------------------- FIRST STAGE ROUTES --------------------------------------------------------------")
        print(df_firststage_routes.to_string(index=False))
        print(
            "\n-------------------------------------------------------------- SECOND STAGE ROUTES --------------------------------------------------------------")
        print(df_secondstage_routes.to_string(index=False))


if __name__ == "__main__":
    from pyinstrument import Profiler
    from Heuristics.LocalSearch.local_search import LocalSearch

    filename = "InstanceGenerator/InstanceFiles/20nodes/20-25-2-1_a"
    ch = ConstructionHeuristic(filename + ".pkl", acceptance_percentage=2)


    #ch.construct()


    emp1 = ch.employees_dict[1]
    emp2 = ch.employees_dict[2]
    emp3 = ch.employees_dict[3]
    emp4 = ch.employees_dict[4]
    cm2 = ch.car_moves_dict[2]
    cm300 = ch.car_moves_dict[300]
    cm6 = ch.car_moves_dict[6]
    cm242 = ch.car_moves_dict[242]
    cm207 = ch.car_moves_dict[207]
    cm157 = ch.car_moves_dict[157]
    cm112 = ch.car_moves_dict[112]
    cm137 = ch.car_moves_dict[137]
    cm141 = ch.car_moves_dict[141]
    cm248 = ch.car_moves_dict[248]
    cm193 = ch.car_moves_dict[193]
    cm47 = ch.car_moves_dict[47]
    cm184 = ch.car_moves_dict[184]
    cm204 = ch.car_moves_dict[204]
    cm159 = ch.car_moves_dict[159]
    cm155 = ch.car_moves_dict[155]
    cm60 = ch.car_moves_dict[60]
    cm216 = ch.car_moves_dict[216]
    cm80 = ch.car_moves_dict[80]
    cm93 = ch.car_moves_dict[93]
    cm273 = ch.car_moves_dict[273]
    cm122 = ch.car_moves_dict[122]



    second_solution_dict = {emp1: [[cm273, cm122]]*25,
                     emp2: [[cm2, cm300], [cm2, cm300, cm6], [cm2, cm300], [cm2, cm300, cm242], [cm2, cm300, cm207, cm157],
                           [cm2, cm300, cm112], [cm2, cm300, cm242], [cm2, cm300, cm157], [cm2, cm300, cm137], [cm2, cm300, cm141],
                           [cm2, cm300, cm248], [cm2, cm300, cm193], [cm2, cm300, cm242], [cm2, cm300, cm242], [cm2, cm300, cm47],
                           [cm2, cm300, cm242], [cm2, cm300, cm242], [cm2, cm300, cm47], [cm2, cm300, cm242], [cm2, cm300, cm242],
                           [cm2, cm300, cm204], [cm2, cm300, cm207], [cm2, cm300, cm242], [cm2, cm300, cm184], [cm2, cm300, cm159]],
                    emp3: [[]]*25,
                    emp4: [[cm155, cm60], [cm155, cm60, cm216]]+[[cm155, cm60]]*3+[[cm155, cm60, cm80]]+
                          [[cm155, cm60]]*11+[[cm155, cm60, cm216]]+[[cm155, cm60]]+[[cm155, cm60, cm80]]
                          +[[cm155, cm60]]*3+[[cm155, cm60, cm93]]+[[cm155, cm60]]}


    #first_solution_dict = {emp1: [cm60, cm10], emp2: [cm55, cm40]}
    first_solution_dict = {emp1: [], emp2: []}

    #first_solution_dict = {emp1: [cm15, cm26],
    #                 emp2: [cm4, cm12]}

    ch.construct()
    ch.print_solution()

    local_search = LocalSearch(ch.assigned_car_moves,
                               2, ch.feasibility_checker)
    local_search.search("best_first")
    print(local_search.solution)
    for emp, cms in local_search.solution.items():
        print(emp.employee_id)
        for s in cms:
            print("scenario", cms.index(s))
            if s:
                print([cm.car_move_id for cm in s])
            else:
                print("[]")

    #print(ch.car_moves_dict)
    #print(local_search.solution)
    ch.rebuild(local_search.solution, stage="second", optimize=True)
    ch.print_solution()
    #ch.print_solution()
    #ch.rebuild(local_search.solution, stage="second")
    #ch.print_solution()




    #profiler = Profiler()
    #profiler.start()

    #ch.construct()


    #profiler.stop()


    '''
    print("\n############## Evaluate solution ##############")
    gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=False)#, optimize=False)
    run_model(gi)

    print("\n############## Reoptimized solution ##############")
    gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=True)
    run_model(gi)
    '''