from DestroyAndRepairHeuristics.destroy import RandomRemoval, WorstRemoval, ShawRemoval
from DestroyAndRepairHeuristics.repair import GreedyInsertion, RegretInsertion
from construction_heuristic import ConstructionHeuristic


class RebuildHeuristic():
    pass

if __name__ == "__main__":
    print("\n---- HEURISTIC ----")
    ch = ConstructionHeuristic("InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_a.pkl")
    ch.add_car_moves_to_employees()
    ch.print_solution()
    ch.get_objective_function_val()
    rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
                       neighborhood_size=2)
    rr.to_string()
    gi = GreedyInsertion(destroyed_solution_object=rr, construction_heuristic=ch)
    gi.to_string()

    ri = RegretInsertion(destroyed_solution_object=rr, construction_heuristic=ch, regret_nr=1)
    ri.to_string()
