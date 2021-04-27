import os
from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Heuristics.feasibility_checker import FeasibilityChecker
from Heuristics.helper_functions_heuristics import remove_all_car_moves_of_car_in_car_move, get_best_car_move, \
    get_first_stage_solution_and_removed_moves, get_first_stage_solution, get_first_and_second_stage_solution
from Heuristics.objective_function import get_objective_function_val
from src.HelperFiles.helper_functions import load_object_from_file
from src.Gurobi.Model.run_model import run_model
import pandas as pd
from path_manager import path_to_src

os.chdir(path_to_src)


class ConstructionHeuristic:
    # instance_file = "InstanceFiles/6nodes/6-3-1-1_d.pkl"
    # filename = "InstanceFiles/6nodes/6-3-1-1_b.yaml"


    def __init__(self, instance_file):


        self.instance_file = instance_file
        self.world_instance = load_object_from_file(instance_file)
        # self.world_instance.planning_period = 100
        self.feasibility_checker = FeasibilityChecker(self.world_instance)
        self.num_scenarios = self.world_instance.num_scenarios
        self.employees = self.world_instance.employees
        self.parking_nodes = self.world_instance.parking_nodes
        self.cars = self.world_instance.cars
        self.unused_car_moves = [[] for _ in range(
            self.num_scenarios)]  # [beta] list of unused car_moves for scenario s (zero index)
        self.assigned_car_moves = {k: [[] for _ in range(self.num_scenarios)] for k in
                                   self.employees}  # [gamma_k] dictionary containing ordered list of car_moves,
        # assigned to employee k in scenario s
        self.car_moves = []  # self.world_instance.car_moves
        self.car_moves_second_stage = []
        self._initialize_car_moves()

        self.available_employees = True
        self.first_stage = True
        # self.hash_key = 0

        # self.add_car_moves_to_employees()

    def get_obj_val(self, true_objective=True, both=False):
        return get_objective_function_val(parking_nodes=self.parking_nodes, employees=self.employees,
                                          num_scenarios=self.num_scenarios, true_objective=true_objective, both=both)

    def _initialize_for_rebuild(self):
        self.unused_car_moves = [[] for _ in range(
            self.num_scenarios)]  # [beta] list of unused car_moves for scenario s (zero index)
        for k in self.employees:
            k.reset()
        self.assigned_car_moves = {k: [[] for _ in range(self.num_scenarios)] for k in self.employees}  # [gamma_k] dictionary containing ordered list of car_moves,
        # assigned to employee k in scenario s
        self.car_moves = []  # self.world_instance.car_moves
        self.car_moves_second_stage = []
        self._initialize_car_moves()
        self.available_employees = True
        self.first_stage = True

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

    def rebuild(self, solution, stage="first", verbose=False):
        self._initialize_for_rebuild()
        if stage == "first":
            # Check if this is not necessary for LNS
            employee_ids = {e.employee_id: e for e in self.employees}
            car_move_ids = {cm.car_move_id: cm for cm in self.car_moves}
            first_stage_solution = solution
            #print("rebuild")
            #print(first_stage_solution)
            for employee_obj, car_move_objs in first_stage_solution.items():
                emp = employee_ids[employee_obj.employee_id]
                for cm_obj in car_move_objs:
                    cm = car_move_ids[cm_obj.car_move_id]
                    self._add_car_move_to_employee(car_moves=self.car_moves, best_car_move=cm, best_employee=emp)
            self.add_car_moves_to_employees()

        else:
            first_stage_solution, second_stage_solution = get_first_and_second_stage_solution(solution, self.world_instance.first_stage_tasks)
            for employee_obj, car_move_objs in first_stage_solution.items():
                for cm_obj in car_move_objs:
                    self._add_car_move_to_employee(car_moves=self.car_moves,
                                                   best_car_move=cm_obj,
                                                   best_employee=employee_obj)

            for employee_obj, car_moves_scenarios in second_stage_solution.items():
                self._add_car_move_to_employee_from_dict(car_moves_second_stage=self.car_moves_second_stage,
                                                         employee=employee_obj,
                                                         car_moves_scenarios=car_moves_scenarios)


        if verbose:
            print("\nRepaired solution")
            self.print_solution()

    def _initialize_car_moves(self):
        for car in self.world_instance.cars:
            for car_move in car.car_moves:
                self.car_moves.append(car_move)
                for s in range(self.num_scenarios):
                    self.unused_car_moves[s].append(car_move)

    def get_best_employee(self, employees, best_car_move, first_stage, num_scenarios, world_instance,
                          car_moves_second_stage):
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
                    legal_move = self.feasibility_checker.check_legal_move(car_move=best_car_move, employee=employee)
                    # print(f"legal_move {legal_move}\n{best_car_move.to_string()}")
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
                            legal_move = self.feasibility_checker.check_legal_move(
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
                self.car_moves.remove(best_car_move)
                return
            else:
                return best_employee
        else:
            if best_move_not_legal:
                for s in range(num_scenarios):
                    self.car_moves_second_stage[s] = [cm for cm in car_moves_second_stage[s] if
                                                      cm != best_car_move[s]]
                return
            else:
                return best_employee_second_stage

    def add_car_moves_to_employees(self, verbose=False):
        if verbose:
            print("------- CONSTRUCTION HEURISTIC -------\n")
        improving_car_move_exists = True
        second_stage_move_counter = 0
        first_stage_move_counter = 0
        while self.available_employees and improving_car_move_exists:

            if self.first_stage:
                car_moves = self.car_moves
            else:
                car_moves = self.car_moves_second_stage

            if self.first_stage:
                #### GET BEST CAR MOVE ###

                best_car_move_first_stage = get_best_car_move(self.parking_nodes, self.employees, car_moves,
                                                              self.first_stage, self.num_scenarios)
                if best_car_move_first_stage is None:
                    improving_car_move_exists = False
                    continue
                #### GET BEST EMPLOYEE ###
                best_employee_first_stage = self.get_best_employee(best_car_move=best_car_move_first_stage,
                                                                   employees=self.employees,
                                                                   first_stage=self.first_stage,
                                                                   num_scenarios=self.num_scenarios,
                                                                   world_instance=self.world_instance,
                                                                   car_moves_second_stage=
                                                                   self.car_moves_second_stage)
                if best_employee_first_stage is not None:
                    ### ADD CAR MOVE TO EMPLOYEE ###
                    self._add_car_move_to_employee(car_moves=car_moves, best_car_move=best_car_move_first_stage,
                                                   best_employee=best_employee_first_stage)
                    first_stage_move_counter += 1
                    if verbose:
                        print(f"{first_stage_move_counter} first stage insertions completed\n")


            else:

                ### GET BEST CAR MOVE ###
                best_car_move_second_stage = get_best_car_move(self.parking_nodes, self.employees, car_moves,
                                                               self.first_stage, self.num_scenarios)
                if all(cm is None for cm in best_car_move_second_stage):
                    improving_car_move_exists = False

                ### GET BEST EMPLOYEE ###
                best_employee_second_stage = self.get_best_employee(best_car_move=best_car_move_second_stage,
                                                                    employees=self.employees,
                                                                    first_stage=self.first_stage,
                                                                    num_scenarios=self.num_scenarios,
                                                                    world_instance=self.world_instance,
                                                                    car_moves_second_stage=
                                                                    self.car_moves_second_stage)
                ### ADD CAR MOVE TO EMPLOYEE ###
                if best_employee_second_stage is not None:
                    self._add_car_move_to_employee(car_moves=car_moves, best_car_move=best_car_move_second_stage,
                                                   best_employee=best_employee_second_stage)

                    # Visual feedback for construction process
                    second_stage_move_counter += 1
                    added_scenarios = []
                    for i in range(len(best_car_move_second_stage)):
                        if best_car_move_second_stage[i] and best_employee_second_stage[i]:
                            added_scenarios.append(1)
                        else:
                            added_scenarios.append(0)

                    scenarios_with_insertion = [i for i, x in enumerate(added_scenarios) if x == 1]
                    if verbose:
                        print(f"Second stage insertion number {second_stage_move_counter}:")
                        print("{} second stage car moves added in scenarios {}\n".format(
                            len(scenarios_with_insertion), ([i + 1 for i in scenarios_with_insertion])))

    def _add_car_move_to_employee(self, car_moves, best_car_move, best_employee):

        if self.first_stage:
            if best_employee is not None:

                self.world_instance.add_car_move_to_employee(best_car_move, best_employee)
                print(f"add car_move {best_car_move.car_move_id} to employee {best_employee.employee_id}")
                print("best_obj_val_first: ", self.get_obj_val(False, False))

                for s in range(self.num_scenarios):
                    self.assigned_car_moves[best_employee][s].append(best_car_move)
                    self.unused_car_moves[s].remove(best_car_move)

                self.car_moves = remove_all_car_moves_of_car_in_car_move(best_car_move, car_moves)

                self.first_stage = False
                for employee in self.employees:
                    task_num = len(employee.car_moves)
                    if task_num < self.world_instance.first_stage_tasks:
                        self.first_stage = True
                if not self.first_stage:
                    # initialize charging and parking moves for second stage
                    self.car_moves_second_stage = [self.car_moves for _ in range(self.num_scenarios)]
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
                        '''
                        print('\nEmployee id', best_employee[s].employee_id)
                        print('Scenario', s + 1)
                        print('Employee node before', best_employee[s].current_node_second_stage[s].node_id)
                        print('Employee time before', best_employee[s].current_time_second_stage[s])
                        # print('Travel time to start node', best_travel_time_to_car_move_second_stage[s])
                        #print(best_car_move[s].to_string())
                        '''
                        # self.world_instance.add_car_move_to_employee(best_car_move[s], best_employee[s], s)
                        if best_car_move[s] is not None:
                            self.world_instance.add_car_move_to_employee(best_car_move[s], best_employee[s], s)
                            self.assigned_car_moves[best_employee[s]][s].append(best_car_move[s])
                            self.unused_car_moves[s].remove(best_car_move[s])
                            # print("start times emp {}:".format(best_employee[s].employee_id), best_employee[s].start_times_car_moves_second_stage)
                            # if best_car_move[s].is_charging_move:
                            #    best_car_move[s].end_node.add_car(scenario=s)
                            #    print("Added MoveID", best_car_move[s].car_move_id)
                            #    print("scenario", s)
                            #    print("EndNode", best_car_move[s].end_node.node_id)
                            #    print("Endnode", best_car_move[s].end_node)
                            #    print("Num charging", best_car_move[s].end_node.num_charging)
                        '''
                        print('Employee node after', best_employee[s].current_node_second_stage[s].node_id)
                        print('Employee time after', best_employee[s].current_time_second_stage[s])
                        '''
                        # When first stage is finished, initialize car_moves to be list of copies of
                        # car_moves (number of copies = num_scenarios)
                        self.car_moves_second_stage[s] = remove_all_car_moves_of_car_in_car_move(best_car_move[s],
                                                                                                 car_moves[s])

                # print(f"car_moves: {len(car_moves[s])}")
                if not any(self.car_moves_second_stage):
                    self.available_employees = False
            else:
                self.available_employees = False

        # self._set_hash_key()

    def _add_car_move_to_employee_from_dict(self, car_moves_second_stage, employee, car_moves_scenarios):
        for s, car_moves in enumerate(car_moves_scenarios):
            for car_move in car_moves:
                self.world_instance.add_car_move_to_employee(car_move, employee, s)
                self.assigned_car_moves[employee][s].append(car_move)
                self.unused_car_moves[s].remove(car_move)
                self.car_moves_second_stage[s] = remove_all_car_moves_of_car_in_car_move(car_move, car_moves_second_stage[s])

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
            columns=["           Type", "  Employee", "  Task number", "  Car Move", "  Car ID", "  Employee Route", "  Start Time", "  Travel Time to Task", "  Relocation Time",
                     "  End time"])
        df_secondstage_routes = pd.DataFrame(
            columns=["Scenario", "  Type", "  Employee", "  Task number", "  Car Move", "  Car ID", "  Employee Route", "  Start Time", "  Travel Time to Task", "  Relocation Time",
                     "  End time"])

        print("\n")
        print("----------- CONSTRUCTION HEURISTIC SOLUTION -----------\n")
        print(f"True objective value: {round(true_obj_val, 2)}")
        print(f"Heuristic objective value: {round(heuristic_obj_val, 2)}")
        print(f"Number of charging moves performed: {num_charging_moves}/{cars_in_need}")
        # print(f"Number of cars charged: {num_charging_moves}\n")

        #print("-------------- First stage routes --------------")
        for employee in self.employees:
            #print("---- Employee {} ----".format(employee.employee_id))
            for car_move in employee.car_moves:
                car_move_type = "C" if car_move.is_charging_move else "P"
                if employee.car_moves.index(car_move) == 0:
                    start_node_id = employee.start_node.node_id
                else:
                    start_node_id = employee.car_moves[employee.car_moves.index(car_move)-1].end_node.node_id

                first_stage_row = [car_move_type, employee.employee_id, employee.car_moves.index(car_move) + 1,
                                   car_move.car_move_id, car_move.car.car_id,
                                   f"{start_node_id} -> {car_move.start_node.node_id} -> {car_move.end_node.node_id}",
                                   round(employee.start_times_car_moves[employee.car_moves.index(car_move)]
                                         - employee.travel_times_car_moves[employee.car_moves.index(car_move)], 2),
                                   round(employee.travel_times_car_moves[employee.car_moves.index(car_move)], 2),
                                   round(employee.car_moves[employee.car_moves.index(car_move)].handling_time, 2),
                                   round(car_move.start_time + car_move.handling_time, 2)
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
        #print("-------------- Second stage routes --------------")
        for s in range(self.num_scenarios):
            #print("---- Scenario {} ----".format(s + 1))
            for employee in self.employees:
                if employee.car_moves_second_stage:
                    if any(employee.car_moves_second_stage[s]):
                        #print("     ---- Employee {} ----".format(employee.employee_id))
                        for car_move in employee.car_moves_second_stage[s]:
                            car_move_type = "C" if car_move.is_charging_move else "P"
                            if employee.car_moves_second_stage[s].index(car_move) == 0:
                                start_node_id = employee.car_moves[-1].end_node.node_id
                            else:
                                start_node_id = employee.car_moves_second_stage[s][employee.car_moves_second_stage[s].index(car_move)-1].end_node.node_id

                            second_stage_row = [s + 1, car_move_type, employee.employee_id, employee.car_moves_second_stage[s].index(car_move) + 1 + len(employee.car_moves),
                                                car_move.car_move_id, car_move.car.car_id,
                                                f"{start_node_id} -> {car_move.start_node.node_id} -> {car_move.end_node.node_id}",
                                                round(employee.start_times_car_moves_second_stage[s][employee.car_moves_second_stage[s].index(car_move)]
                                                      - employee.travel_times_car_moves_second_stage[s][employee.car_moves_second_stage[s].index(car_move)], 2),
                                                round(employee.travel_times_car_moves_second_stage[s][employee.car_moves_second_stage[s].index(car_move)], 2),
                                                round(employee.car_moves_second_stage[s][employee.car_moves_second_stage[s].index(car_move)].handling_time, 2),
                                                round(car_move.start_times_second_stage[s] + car_move.handling_time, 2)]
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

        print("--------------------------------------------------------------- FIRST STAGE ROUTES --------------------------------------------------------------")
        print(df_firststage_routes.to_string(index=False))
        print("\n-------------------------------------------------------------- SECOND STAGE ROUTES --------------------------------------------------------------")
        print(df_secondstage_routes.to_string(index=False))

if __name__ == "__main__":
    from pyinstrument import Profiler

    filename = "InstanceGenerator/InstanceFiles/4nodes/4-2-1-1_a"
    ch = ConstructionHeuristic(filename + ".pkl")
    profiler = Profiler()
    profiler.start()
    ch.add_car_moves_to_employees()
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
    true_obj_val, best_obj_val = ch.get_obj_val(both=True)
    # print(f"Construction heuristic true obj. val {true_obj_val}")
    ch.print_solution()