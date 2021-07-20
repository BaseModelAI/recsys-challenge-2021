import argparse
import random
import logging
import os
import yaml


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    return parser


def split_validation(config):
    log.info("Split validation into training and testing parts")
    valid_lines = []
    training_lines = []
    with open(config['validation_part'], encoding="utf-8") as f:
        for line in f:
            if random.random() <= config['validation_percentage']:
                valid_lines.append(line)
            else:
                training_lines.append(line)

    with open(os.path.join(config['working_dir'], 'validation_for_training'), 'w') as f:
        for line in training_lines:
            f.write(line)

    with open(os.path.join(config['working_dir'], 'validation_holdout'), 'w') as f:
        for line in valid_lines:
            f.write(line)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    with open(params.config_file) as f:
        config = yaml.load(f)
    split_validation(config)
