import yaml


def read_config():
    with open("instance_config.yaml", 'r') as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

read_config()