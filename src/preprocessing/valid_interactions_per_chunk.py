import argparse
import logging
import pickle
import yaml
from collections import Counter
from sipHash64 import sipHash64
from utils import compute_validation_interactions_datapoint, update_interactions_datapoint


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    # parser.add_argument("--num-chunks", type=str, default='config.yaml', help='Number of validation chunks.')
    return parser


def add_interactions_per_chunk(config):
    log.info("Processing validation chunks")
    with open(f"""{config['working_dir']}/validation_split_train_with_interactions""", 'rb') as handle:
        valid_set_for_training = pickle.load(handle)

    for fake_day_training in range(config['num_validation_chunks']):
        log.info(f"Processing chunk id: {fake_day_training}")
        interaction_counter_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
        num_enaged_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
        num_enaging_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
        user_hashtag_all =  {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
        interaction_counter_lang_all =  {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
        twit_positive = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}

        twit_negative = Counter()
        interaction_counter_all_negative = Counter()
        num_enaged_all_negative = Counter()
        num_enaging_all_negative = Counter()

        for line_idx in range(len(valid_set_for_training)):
            fake_day = sipHash64(str(line_idx)) % config['num_validation_chunks']
            if fake_day != fake_day_training:
                compute_validation_interactions_datapoint(valid_set_for_training[line_idx], interaction_counter_all, num_enaged_all, num_enaging_all, user_hashtag_all,
                                    interaction_counter_lang_all, twit_positive, interaction_counter_all_negative,
                                    num_enaged_all_negative, num_enaging_all_negative, twit_negative)


        valid_for_training_with_valid_interactions = []

        for line_idx in range(len(valid_set_for_training)):
            fake_day = sipHash64(str(line_idx)) % config['num_validation_chunks']
            if fake_day == fake_day_training:
                interactions_valid, interactions_valid_negatives = update_interactions_datapoint(valid_set_for_training[line_idx], num_enaged_all, num_enaging_all, interaction_counter_all,
                                                                                            interaction_counter_lang_all, user_hashtag_all, twit_positive, interaction_counter_all_negative,
                                                                                            num_enaged_all_negative, num_enaging_all_negative, twit_negative)

                current_example = valid_set_for_training[line_idx]
                current_example['interactions_valid'] = interactions_valid
                current_example['interactions_valid_negatives'] = interactions_valid_negatives
                valid_for_training_with_valid_interactions.append(current_example)

        with open(f"{config['working_dir']}/validation_train_with_valid_interactions_{fake_day_training}", 'wb') as handle:
            pickle.dump(valid_for_training_with_valid_interactions, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    with open(params.config_file) as f:
        config = yaml.load(f)
    add_interactions_per_chunk(config)
