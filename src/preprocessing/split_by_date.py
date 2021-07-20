
"""
Reading all training parts from directory provided in configuration file.
Splitting them into new files for each day separately.
For example, all training examples from 4th February will be saved into `preprocessing/train_set_all_04-02-21`
"""

from datetime import datetime
from collections import defaultdict
from typing import List
import argparse
import os
import pickle
import math
import logging
import yaml
from read_dataset_utils import all_features_to_idx, labels_to_idx


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    return parser


def split(config: dict, parts_filenames_chunks: List[List[str]]):
    all_languages = set()
    days = set()
    for chunk_idx, filenames in enumerate(parts_filenames_chunks):
        train_set = defaultdict(list)
        for filename in filenames:
            log.info(f'Processing {filename}')
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    features = line.split("\x01")
                    all_languages.add(features[all_features_to_idx['language']])
                    timestamp = int(features[all_features_to_idx['tweet_timestamp']])
                    if any(t != '' for t in [features[f] for f in list(labels_to_idx.values())]):
                        reaction_timestamp = int(max([int(features[f]) for f in list(labels_to_idx.values()) if features[f] != '']))
                        timestamp = reaction_timestamp

                    train_set[datetime.utcfromtimestamp(timestamp).date()].append({'line': line, 'timestamp': timestamp})

        for key in train_set.keys():
            days.add(key)
            with open(f"""{config['working_dir']}/train_set_{chunk_idx}_{key.strftime('%d-%m-%y')}""", 'wb') as handle:
                pickle.dump(train_set[key], handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
    return all_languages, days


def merge(config: dict, days: set, parts_filenames_chunks: List[List[str]]):
    for day in days:
        log.info(f"Merging date: {day}")
        train_day_all = []
        for chunk_idx in range(len(parts_filenames_chunks)):
            chunk_filename = f"""{config['working_dir']}/train_set_{chunk_idx}_{day.strftime('%d-%m-%y')}"""
            with open(chunk_filename, 'rb') as handle:
                train_day_all += pickle.load(handle)
            os.remove(chunk_filename)
        with open(f"""{config['working_dir']}/train_set_all_{day.strftime('%d-%m-%y')}""", 'wb') as handle:
            pickle.dump(train_day_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


def split_training_parts(config: dict):
    os.makedirs(config['working_dir'], exist_ok=True)

    part_files = [os.path.join(config['recsys_data'], f) for f in os.listdir(config['recsys_data']) if 'part' in f][:config['max_n_parts']]
    log.info(f"Number of used training parts: {len(part_files)}")

    parts_filenames_chunks = [part_files[i * config['max_n_parts_in_memory']:(i+1) * config['max_n_parts_in_memory']] for i in range(math.ceil(len(part_files) / config['max_n_parts_in_memory']))]
    all_languages, days = split(config, parts_filenames_chunks)
    merge(config, days, parts_filenames_chunks)

    language2id = {l: i for i, l in enumerate(all_languages)}
    with open(f"{config['working_dir']}/language2id", 'wb') as handle:
        pickle.dump(language2id, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    log.info("Splitting parts by date")
    with open(params.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    split_training_parts(config)
