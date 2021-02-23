def read_config(config_name: str):
    with open(config_name, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

cf = read_config('../Gurobi/tests/6nodes/6-3-0-1_a.yaml')