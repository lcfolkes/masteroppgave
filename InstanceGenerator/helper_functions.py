import yaml

def read_config(config_name: str):
    with open(config_name, 'r') as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

read_config('world_constants_config.yaml')