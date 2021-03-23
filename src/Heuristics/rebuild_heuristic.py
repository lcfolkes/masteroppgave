from DestroyAndRepairHeuristics.destroy import RandomRemoval, WorstRemoval, ShawRemoval
from DestroyAndRepairHeuristics.repair import GreedyInsertion, RegretInsertion
from Gurobi.Model.gurobi_heuristic_instance import GurobiInstance
from Heuristics.feasibility_checker import FeasibilityChecker
from construction_heuristic_new import ConstructionHeuristic


class RebuildHeuristic():

    pass

if __name__ == "__main__":
    filename = "InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_a"
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
    print(fc.is_first_stage_solution_feasible(gi.repaired_solution, True))
    ch.rebuild(gi.repaired_solution)
    ch.print_solution()
    ch.add_car_moves_to_employees()
    ch.print_solution()

    print("\n---- GUROBI ----")
    gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=False)
    gi = GurobiInstance(filename + ".yaml", employees=ch.employees, optimize=True)
    # gi = GurobiInstance(filename + ".yaml")
    # run_model(gi)

