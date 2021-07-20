import argparse
import pickle
import os
import logging
import yaml
from sipHash64 import sipHash64
from read_dataset_utils import all_features_to_idx
from utils import get_similar_interactions


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    return parser


def get_training_interactions(line, interaction_users, num_enaged, num_enaging, interaction_counter_lang, user_hashtag, auth2similar_followers):
    features = line.split("\x01")

    engaged_with_user_id_orig = features[all_features_to_idx['engaged_with_user_id']]
    enaging_user_id_orig = features[all_features_to_idx['enaging_user_id']]
    key = sipHash64(f'{engaged_with_user_id_orig}_{enaging_user_id_orig}')

    hashtags = features[all_features_to_idx['hashtags']].split()
    key_hashtags = [sipHash64(f'{h}_{enaging_user_id_orig}') for h in hashtags]
    language = features[all_features_to_idx['language']]
    key_language = sipHash64(f'{language}_{enaging_user_id_orig}')


    engaged_with_user_id = sipHash64(engaged_with_user_id_orig)
    enaging_user_id = sipHash64(enaging_user_id_orig)


    interaction_counter_lang_like = interaction_counter_lang['like'][key_language]
    interaction_counter_lang_reply = interaction_counter_lang['reply'][key_language]
    interaction_counter_lang_retweet = interaction_counter_lang['retweet'][key_language]
    interaction_counter_lang_retweet_with_comment = interaction_counter_lang['retweet_with_comment'][key_language]

    user_hashtag_like = 0
    user_hashtag_reply = 0
    user_hashtag_retweet = 0
    user_hashtag_retweet_with_comment = 0

    for key_hashtag_current in key_hashtags:
        user_hashtag_like += user_hashtag['like'][key_hashtag_current]
        user_hashtag_reply += user_hashtag['reply'][key_hashtag_current]
        user_hashtag_retweet += user_hashtag['retweet'][key_hashtag_current]
        user_hashtag_retweet_with_comment += user_hashtag['retweet_with_comment'][key_hashtag_current]


    similar_followers = get_similar_interactions(auth2similar_followers, engaged_with_user_id_orig, enaging_user_id_orig, interaction_users)


    interactions = [interaction_users['like'][key],
                    interaction_users['reply'][key],
                    interaction_users['retweet'][key],
                    interaction_users['retweet_with_comment'][key],
                    num_enaged['like'][engaged_with_user_id],
                    num_enaged['reply'][engaged_with_user_id],
                    num_enaged['retweet'][engaged_with_user_id],
                    num_enaged['retweet_with_comment'][engaged_with_user_id],
                    num_enaging['like'][enaging_user_id],
                    num_enaging['reply'][enaging_user_id],
                    num_enaging['retweet'][enaging_user_id],
                    num_enaging['retweet_with_comment'][enaging_user_id],
                    interaction_counter_lang_like,
                    interaction_counter_lang_reply,
                    interaction_counter_lang_retweet,
                    interaction_counter_lang_retweet_with_comment,
                    user_hashtag_like,
                    user_hashtag_reply,
                    user_hashtag_retweet,
                    user_hashtag_retweet_with_comment] + similar_followers
    return interactions


def add_interactions(config):
    log.info("Adding training interactions to valid")
    with open(f"""{config['working_dir']}/author2similar_follow.pkl""", 'rb') as handle:
        auth2similar_followers = pickle.load(handle)

    with open(f"""{config['working_dir']}/interaction_users""", 'rb') as handle:
        interaction_users = pickle.load(handle)

    with open(f"""{config['working_dir']}/num_enaging""", 'rb') as handle:
        num_enaging = pickle.load(handle)

    with open(f"""{config['working_dir']}/num_enaged""", 'rb') as handle:
        num_enaged = pickle.load(handle)

    with open(f"""{config['working_dir']}/interaction_counter_lang""", 'rb') as handle:
        interaction_counter_lang = pickle.load(handle)

    with open(f"""{config['working_dir']}/user_hashtag""", 'rb') as handle:
        user_hashtag = pickle.load(handle)

    data = []

    with open(os.path.join(config['working_dir'], 'validation_for_training'), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            interactions = get_training_interactions(line, interaction_users, num_enaged, num_enaging, interaction_counter_lang, user_hashtag, auth2similar_followers)
            data.append({'line': line,
                        'interactions': interactions})
    with open(f"""{config['working_dir']}/validation_split_train_with_interactions""", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    data = []

    with open(os.path.join(config['working_dir'], 'validation_holdout'), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            interactions = get_training_interactions(line, interaction_users, num_enaged, num_enaging, interaction_counter_lang, user_hashtag, auth2similar_followers)
            data.append({'line': line,
                        'interactions': interactions})

    with open(f"""{config['working_dir']}/validation_split_holdout""", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    with open(params.config_file) as f:
        config = yaml.load(f)
    add_interactions(config)
