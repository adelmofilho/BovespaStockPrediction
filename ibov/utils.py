import json


def load_config(file="config.json"):

    with open(file, "r") as f:
        config = json.load(f)

    return config
