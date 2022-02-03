import os
import argparse
import yaml
import logging


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def main(config_path, datasource):
    config = read_params(config_path)


if __name__ == "main":
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument("--config", default=default_config_path,
                      help="Path to config file")
    args.add_argument("--datasource", default=None, help="Path to data file")

    parsed_args = args.parse_args()
    main(config_path=parsed_args.config, datasource=parsed_args.datasource)
