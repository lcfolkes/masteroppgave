import copy

from DestroyAndRepairHeuristics.destroy import RandomRemoval, WorstRemoval, ShawRemoval
from DestroyAndRepairHeuristics.repair import GreedyInsertion, RegretInsertion
from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Gurobi.Model.run_model import run_model
from construction_heuristic_new import ConstructionHeuristic
from path_manager import path_to_src
import os

os.chdir(path_to_src)


class ALNS():
    def __init__(self, filename):
        self.filename = filename
        print(filename)
        self.best_solution = None
        self.best_obj_val = 0
        self.run()

    def run(self):
        it = 100
        construction = ConstructionHeuristic(self.filename)
        construction.add_car_moves_to_employees()
        construction.print_solution()
        best_solution = copy.deepcopy(construction)
        # TODO: this is the old objective function val
        best_obj_val = construction.get_obj_val()
        obj_vals = []
        while it > 0:
            destroy = WorstRemoval(solution=construction.assigned_car_moves,
                                    num_first_stage_tasks=construction.world_instance.first_stage_tasks,
                                    neighborhood_size=2, randomization_degree=100, parking_nodes=construction.parking_nodes)
            repair = RegretInsertion(destroyed_solution_object=destroy, unused_car_moves=construction.unused_car_moves,
                                     parking_nodes=construction.parking_nodes,
                                     world_instance=construction.world_instance, regret_nr=1)

            construction.rebuild(repair.repaired_solution)

            obj_val = construction.get_obj_val()
            obj_vals.append(obj_val)
            if obj_val > best_obj_val:
                best_solution = copy.deepcopy(construction)
            it -= 1

        self.best_solution = best_solution
        self.best_obj_val = best_obj_val
        print(obj_vals)
        print(best_obj_val)
        best_solution.print_solution()


if __name__ == "__main__":
    filename = "InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_f"
    alns = ALNS(filename + ".pkl")
    gi = GurobiInstance(filename + ".yaml")
    run_model(gi)

    '''
    filename = "InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_b"
    print("\n---- HEURISTIC ----")
    ch = ConstructionHeuristic(filename+".pkl")
    ch.add_car_moves_to_employees()
    ch.print_solution()
    rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
                       neighborhood_size=2)
    rr.to_string()
    gi = GreedyInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
                         parking_nodes=ch.parking_nodes, world_instance=ch.world_instance)
    gi.to_string()
    #ri = RegretInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
    #                     parking_nodes=ch.parking_nodes, world_instance=ch.world_instance, regret_nr=1)
    #ri.to_string()

    fc = FeasibilityChecker(ch.world_instance)
    print("feasibilityChecker")
    print(fc.is_first_stage_solution_feasible(gi.repaired_solution))

    ch.rebuild(gi.repaired_solution, True)
    # gi = GurobiInstance(filename + ".yaml")
    # run_model(gi)

    print("\n---- GUROBI ----")
    print("Verify in Gurobi")
    gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=False)
    run_model(gi)

    ch.add_car_moves_to_employees()
    ch.print_solution()

    print("\n---- GUROBI ----")
    print("Verify in Gurobi")
    gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=False)
    run_model(gi)

    print("\nOptimize in Gurobi")
    gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=True)
    run_model(gi)
    '''
