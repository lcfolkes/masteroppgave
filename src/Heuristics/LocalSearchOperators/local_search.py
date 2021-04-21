from Heuristics.construction_heuristic import ConstructionHeuristic



if __name__ == "__main__":
	print("\n---- HEURISTIC ----")
	ch = ConstructionHeuristic("./InstanceGenerator/InstanceFiles/6nodes/6-3-2-1_a.pkl")
	ch.add_car_moves_to_employees()
	rr = RandomRemoval(solution=ch.assigned_car_moves, num_first_stage_tasks=ch.world_instance.first_stage_tasks,
					   neighborhood_size=2)
	# rr.to_string()
	gi = GreedyInsertion(destroyed_solution_object=rr, unused_car_moves=ch.unused_car_moves,
						 parking_nodes=ch.parking_nodes, world_instance=ch.world_instance)
	print("Local Search")
	ls = InterSwap(gi.repaired_solution, gi.feasibility_checker)
	ls.to_string()
