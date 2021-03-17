def get_first_stage_solution_list_from_dict(first_stage_solution):
    first_stage_solution_list = []
    for k, v in first_stage_solution.items():
        for i in range(len(v)):
            first_stage_solution_list.append(v[i])
    return first_stage_solution_list


def get_second_stage_solution_dict(input_solution, num_first_stage_tasks):
    second_stage_solution = {}
    for k, v in input_solution.items():
        second_stage_solution[k] = []
        for s in range(len(input_solution[k])):
            second_stage_solution[k].append([])
            for i in range(num_first_stage_tasks, len(input_solution[k][s])):
                second_stage_solution[k][s].append(input_solution[k][s][i])
    return second_stage_solution


def get_second_stage_solution_list_from_dict(second_stage_solution_dict, num_scenarios):
    second_stage_solution = [[] for _ in range(num_scenarios)]
    for k, v in second_stage_solution_dict.items():
        for s in range(num_scenarios):
            for i in range(len(v[s])):
                second_stage_solution[s].append(v[s][i])
    return second_stage_solution

