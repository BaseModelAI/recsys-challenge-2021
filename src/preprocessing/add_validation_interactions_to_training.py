import argparse
import os
import logging
import pickle
import yaml
from utils import update_interactions_datapoint


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    return parser


def add_validation_interactions(config):
    log.info("Adding validation interactions to training data")
    with open(f"""{config['working_dir']}/validation_interaction_counter_all""", 'rb') as handle:
        interaction_counter_all = pickle.load(handle)

    with open(f"""{config['working_dir']}/validation_num_enaged_all""", 'rb') as handle:
        num_enaged_all = pickle.load(handle)

    with open(f"""{config['working_dir']}/validation_num_enaging_all""", 'rb') as handle:
        num_enaging_all = pickle.load(handle)

    with open(f"""{config['working_dir']}/validation_user_hashtag_all""", 'rb') as handle:
        user_hashtag_all = pickle.load(handle)

    with open(f"""{config['working_dir']}/validation_interaction_counter_lang_all""", 'rb') as handle:
        interaction_counter_lang_all = pickle.load(handle)

    with open(f"""{config['working_dir']}/validation_twit_positive""", 'rb') as handle:
        twit_positive = pickle.load(handle)

    with open(f"""{config['working_dir']}/validation_interaction_counter_all_NOT""", 'rb') as handle:
        interaction_counter_all_negative = pickle.load(handle)

    with open(f"""{config['working_dir']}/validation_num_enaged_all_NOT""", 'rb') as handle:
        num_enaged_all_negative = pickle.load(handle)

    with open(f"""{config['working_dir']}/validation_num_enaging_all_NOT""", 'rb') as handle:
        num_enaging_all_negative = pickle.load(handle)

    with open(f"""{config['working_dir']}/validation_twit_negative""", 'rb') as handle:
        twit_negative = pickle.load(handle)


    filenames = [os.path.join(config['working_dir'], 'validation_split_holdout')]
    filenames += [os.path.join(config['working_dir'], f) for f in os.listdir(config['working_dir']) if 'train_set_all' in f]
    for filename in filenames:
        log.info(f"Processing {filename}")

        # loading data from current day
        with open(filename, 'rb') as handle:
            train_day = pickle.load(handle)

        for i, example in enumerate(train_day):
            interactions_valid, interactions_valid_negatives = update_interactions_datapoint(example, num_enaged_all, num_enaging_all, interaction_counter_all, interaction_counter_lang_all,
                                    user_hashtag_all, twit_positive, interaction_counter_all_negative, num_enaged_all_negative,
                                    num_enaging_all_negative, twit_negative)

            train_day[i]['interactions_valid'] = interactions_valid

            train_day[i]['interactions_valid_negatives'] = interactions_valid_negatives

        with open(filename, 'wb') as handle:
            pickle.dump(train_day, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    with open(params.config_file) as f:
        config = yaml.load(f)
    add_validation_interactions(config)
