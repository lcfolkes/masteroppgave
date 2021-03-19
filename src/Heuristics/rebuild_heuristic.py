from DestroyAndRepairHeuristics.destroy import RandomRemoval, WorstRemoval, ShawRemoval
from DestroyAndRepairHeuristics.repair import GreedyInsertion
from construction_heuristic import ConstructionHeuristic


class RebuildHeuristic():
	pass

if __name__ == "__main__":
    print("\n---- HEURISTIC ----")
    ch = ConstructionHeuristic("InstanceGenerator/InstanceFiles/6nodes/6-3-1-1_g.pkl")
    ch.add_car_moves_to_employees()
    ch.print_solution()
    ch.get_objective_function_val()
    rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
                       neighborhood_size=2)
    rr.to_string()
    gi = GreedyInsertion(destroyed_solution=rr.destroyed_solution, unused_car_moves=rr.removed_moves,
                         num_first_stage_tasks=ch.world_instance.first_stage_tasks, neighborhood_size=2,
                         parking_nodes=ch.parking_nodes)
    gi.to_string()
