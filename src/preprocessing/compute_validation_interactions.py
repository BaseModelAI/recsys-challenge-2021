import argparse
import pickle
import logging
import yaml
from sipHash64 import sipHash64
from collections import Counter
from utils import compute_validation_interactions_datapoint


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    return parser


def compute_validation_interactions(config):
    log.info("Computting validation interactions")

    with open(f"""{config['working_dir']}/validation_split_train_with_interactions""", 'rb') as handle:
        valid_set_for_training = pickle.load(handle)

    interaction_counter_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
    num_enaged_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
    num_enaging_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
    user_hashtag_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
    interaction_counter_lang_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
    twit_positive = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}

    interaction_counter_all_negative = Counter()
    num_enaged_all_negative = Counter()
    num_enaging_all_negative = Counter()
    twit_negative = Counter()

    for example in valid_set_for_training:
        compute_validation_interactions_datapoint(example, interaction_counter_all, num_enaged_all, num_enaging_all, user_hashtag_all,
                                    interaction_counter_lang_all, twit_positive, interaction_counter_all_negative,
                                    num_enaged_all_negative, num_enaging_all_negative, twit_negative)

    with open(f"""{config['working_dir']}/validation_interaction_counter_all""", 'wb') as handle:
        pickle.dump(interaction_counter_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"""{config['working_dir']}/validation_num_enaged_all""", 'wb') as handle:
        pickle.dump(num_enaged_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"""{config['working_dir']}/validation_num_enaging_all""", 'wb') as handle:
        pickle.dump(num_enaging_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"""{config['working_dir']}/validation_user_hashtag_all""", 'wb') as handle:
        pickle.dump(user_hashtag_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"""{config['working_dir']}/validation_interaction_counter_lang_all""", 'wb') as handle:
        pickle.dump(interaction_counter_lang_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"""{config['working_dir']}/validation_twit_positive""", 'wb') as handle:
        pickle.dump(twit_positive, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"""{config['working_dir']}/validation_interaction_counter_all_NOT""", 'wb') as handle:
        pickle.dump(interaction_counter_all_negative, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"""{config['working_dir']}/validation_num_enaged_all_NOT""", 'wb') as handle:
        pickle.dump(num_enaged_all_negative, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"""{config['working_dir']}/validation_num_enaging_all_NOT""", 'wb') as handle:
        pickle.dump(num_enaging_all_negative, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"""{config['working_dir']}/validation_twit_negative""", 'wb') as handle:
        pickle.dump(twit_negative, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    with open(params.config_file) as f:
        config = yaml.load(f)
    compute_validation_interactions(config)
