import argparse
import os
import pickle
import yaml
import logging
from collections import Counter
from sipHash64 import sipHash64
from read_dataset_utils import all_features_to_idx, labels_to_idx
from utils import get_similar_interactions


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    return parser


def process_example(features):
    engaged_with_user_id = features[all_features_to_idx['engaged_with_user_id']]
    enaging_user_id = features[all_features_to_idx['enaging_user_id']]

    # compute hash of `engaged_with_user_id` and `enaging_user_id` pairs
    key = sipHash64(f'{engaged_with_user_id}_{enaging_user_id}')

    hashtags = features[all_features_to_idx['hashtags']].split()
    # compute hash of hashtag and  `enaging_user_id`
    key_hashtags = [sipHash64(f'{h}_{enaging_user_id}') for h in hashtags]

    language = features[all_features_to_idx['language']]
    # compute hash of language and  `enaging_user_id`
    key_language = sipHash64(f'{language}_{enaging_user_id}')

    # compute hashes of users
    engaged_with_user_id = sipHash64(engaged_with_user_id)
    enaging_user_id = sipHash64(enaging_user_id)
    return engaged_with_user_id, enaging_user_id, key, key_hashtags, key_language


def compute_interactions(config):
    def increase_reaction(relation, key, engaged_with_user_id, enaging_user_id, key_hashtags, key_language):
        interaction_counter[relation][key] += 1
        num_enaged[relation][engaged_with_user_id] += 1
        num_enaging[relation][enaging_user_id] += 1

        interaction_counter_all[relation][key] += 1
        num_enaged_all[relation][engaged_with_user_id] += 1
        num_enaging_all[relation][enaging_user_id] += 1

        for key_hashtag_current in key_hashtags:
            user_hashtag_all[relation][key_hashtag_current] +=1
            user_hashtag[relation][key_hashtag_current] +=1


        interaction_counter_lang_all[relation][key_language] += 1
        interaction_counter_lang[relation][key_language] += 1

    # filenames with training data sepearated by day - calculated in `splt_by_date.py`
    filenames = [os.path.join(config['working_dir'], f) for f in os.listdir(config['working_dir']) if 'train_set' in f]

    # number of all enaged-enaging users interactions per each relation
    interaction_counter_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}

    # number of all interactions of enaged users per each relation
    num_enaged_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}

    # number of all interactions of enaging users per each relation
    num_enaging_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}

    # number of all interactions with hashtags per each relation
    user_hashtag_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}

    # number of all interactions of enaging users with twitts from each language
    interaction_counter_lang_all = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}


    for filename in filenames:
        log.info(f"Processing {filename}")
        # interactions only for data from current day
        interaction_counter = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
        num_enaged = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
        num_enaging = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
        user_hashtag = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}
        interaction_counter_lang = {'like': Counter(), 'reply': Counter(), 'retweet': Counter(), 'retweet_with_comment': Counter()}

        # loading data from current day
        with open(filename, 'rb') as handle:
            train_day = pickle.load(handle)

        for example in train_day:
            line = example['line']
            features = line.split("\x01")
            engaged_with_user_id, enaging_user_id, key, key_hashtags, key_language = process_example(features)

            if features[labels_to_idx[f'like_timestamp']] != '':
                increase_reaction('like', key, engaged_with_user_id, enaging_user_id, key_hashtags, key_language)

            if features[labels_to_idx[f'reply_timestamp']] != '':
                increase_reaction('reply', key, engaged_with_user_id, enaging_user_id, key_hashtags, key_language)


            if features[labels_to_idx[f'retweet_timestamp']] != '':
                increase_reaction('retweet', key, engaged_with_user_id, enaging_user_id, key_hashtags, key_language)

            if features[labels_to_idx[f'retweet_with_comment_timestamp']] != '':
                increase_reaction('retweet_with_comment', key, engaged_with_user_id, enaging_user_id, key_hashtags, key_language)


        file_suffix = filename.split('_')[-1]

        with open(f"""{config['working_dir']}/interaction_users_day_{file_suffix}""", 'wb') as handle:
            pickle.dump(interaction_counter, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"""{config['working_dir']}/num_enaged_day_{file_suffix}""", 'wb') as handle:
            pickle.dump(num_enaged, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"""{config['working_dir']}/num_enaging_day_{file_suffix}""", 'wb') as handle:
            pickle.dump(num_enaging, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"""{config['working_dir']}/user_hashtag_day_{file_suffix}""", 'wb') as handle:
            pickle.dump(user_hashtag, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"""{config['working_dir']}/interaction_counter_lang_day_{file_suffix}""", 'wb') as handle:
            pickle.dump(interaction_counter_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open(f"""{config['working_dir']}/interaction_users""", 'wb') as handle:
        pickle.dump(interaction_counter_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"""{config['working_dir']}/num_enaging""", 'wb') as handle:
        pickle.dump(num_enaging_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"""{config['working_dir']}/num_enaged""", 'wb') as handle:
        pickle.dump(num_enaged_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"""{config['working_dir']}/user_hashtag""", 'wb') as handle:
        pickle.dump(user_hashtag_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"""{config['working_dir']}/interaction_counter_lang""", 'wb') as handle:
        pickle.dump(interaction_counter_lang_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_interactions(config):
    log.info("Saving interactions")
    with open(f"""{config['working_dir']}/author2similar_follow.pkl""", 'rb') as handle:
        auth2similar_followers = pickle.load(handle)

    with open(f"""{config['working_dir']}/interaction_users""", 'rb') as handle:
        interaction_counter_all = pickle.load(handle)

    with open(f"""{config['working_dir']}/num_enaging""", 'rb') as handle:
        num_enaging_all = pickle.load(handle)

    with open(f"""{config['working_dir']}/num_enaged""", 'rb') as handle:
        num_enaged_all = pickle.load(handle)

    with open(f"""{config['working_dir']}/interaction_counter_lang""", 'rb') as handle:
        interaction_counter_lang_all = pickle.load(handle)

    with open(f"""{config['working_dir']}/user_hashtag""", 'rb') as handle:
            user_hashtag_all = pickle.load(handle)

    filenames = [os.path.join(config['working_dir'], f) for f in os.listdir(config['working_dir']) if 'train_set_all' in f]
    for filename in filenames:
        log.info(f"Processing {filename}")

        # loading data from current day
        with open(filename, 'rb') as handle:
            train_day = pickle.load(handle)

        file_suffix = filename.split('_')[-1]

        with open(f"""{config['working_dir']}/interaction_users_day_{file_suffix}""", 'rb') as handle:
            interaction_counter = pickle.load(handle)
        with open(f"""{config['working_dir']}/num_enaged_day_{file_suffix}""", 'rb') as handle:
            num_enaged = pickle.load(handle)
        with open(f"""{config['working_dir']}/num_enaging_day_{file_suffix}""", 'rb') as handle:
            num_enaging = pickle.load(handle)

        with open(f"""{config['working_dir']}/user_hashtag_day_{file_suffix}""", 'rb') as handle:
            user_hashtag = pickle.load(handle)
        with open(f"""{config['working_dir']}/interaction_counter_lang_day_{file_suffix}""", 'rb') as handle:
            interaction_counter_lang = pickle.load(handle)

        interactions_without_current_day = {
            'like': interaction_counter_all['like'] - interaction_counter['like'],
            'reply': interaction_counter_all['reply'] - interaction_counter['reply'],
            'retweet': interaction_counter_all['retweet'] - interaction_counter['retweet'],
            'retweet_with_comment': interaction_counter_all['retweet_with_comment'] - interaction_counter['retweet_with_comment']
        }

        for i, example in enumerate(train_day):
            line = example['line']
            features = line.split("\x01")

            engaged_with_user_id_orig = features[all_features_to_idx['engaged_with_user_id']]
            enaging_user_id_orig = features[all_features_to_idx['enaging_user_id']]

            engaged_with_user_id, enaging_user_id, key, key_hashtags, key_language = process_example(features)

            num_enaged_like = num_enaged_all['like'][engaged_with_user_id] - num_enaged['like'][engaged_with_user_id]
            num_enaged_reply = num_enaged_all['reply'][engaged_with_user_id] - num_enaged['reply'][engaged_with_user_id]
            num_enaged_retweet = num_enaged_all['retweet'][engaged_with_user_id] - num_enaged['retweet'][engaged_with_user_id]
            num_enaged_retweet_with_comment = num_enaged_all['retweet_with_comment'][engaged_with_user_id] - num_enaged['retweet_with_comment'][engaged_with_user_id]

            num_enaging_like = num_enaging_all['like'][enaging_user_id] - num_enaging['like'][enaging_user_id]
            num_enaging_reply = num_enaging_all['reply'][enaging_user_id] - num_enaging['reply'][enaging_user_id]
            num_enaging_retweet = num_enaging_all['retweet'][enaging_user_id] - num_enaging['retweet'][enaging_user_id]
            num_enaging_retweet_with_comment = num_enaging_all['retweet_with_comment'][enaging_user_id] - num_enaging['retweet_with_comment'][enaging_user_id]

            interaction_users_like = interactions_without_current_day['like'][key]
            interaction_users_reply = interactions_without_current_day['reply'][key]
            interaction_users_retweet = interactions_without_current_day['retweet'][key]
            interaction_users_retweet_with_comment = interactions_without_current_day['retweet_with_comment'][key]

            similar_followers = get_similar_interactions(auth2similar_followers, engaged_with_user_id_orig, enaging_user_id_orig, interactions_without_current_day)

            interaction_counter_lang_like = interaction_counter_lang_all['like'][key_language] - interaction_counter_lang['like'][key_language]
            interaction_counter_lang_reply = interaction_counter_lang_all['reply'][key_language] - interaction_counter_lang['reply'][key_language]
            interaction_counter_lang_retweet = interaction_counter_lang_all['retweet'][key_language] - interaction_counter_lang['retweet'][key_language]
            interaction_counter_lang_retweet_with_comment = interaction_counter_lang_all['retweet_with_comment'][key_language] - interaction_counter_lang['retweet_with_comment'][key_language]
            user_hashtag_like = 0
            user_hashtag_reply = 0
            user_hashtag_retweet = 0
            user_hashtag_retweet_with_comment = 0

            for key_hashtag_current in key_hashtags:
                user_hashtag_like += (user_hashtag_all['like'][key_hashtag_current] - user_hashtag['like'][key_hashtag_current])
                user_hashtag_reply += (user_hashtag_all['reply'][key_hashtag_current] - user_hashtag['reply'][key_hashtag_current])
                user_hashtag_retweet += (user_hashtag_all['retweet'][key_hashtag_current] - user_hashtag['retweet'][key_hashtag_current])
                user_hashtag_retweet_with_comment += (user_hashtag_all['retweet_with_comment'][key_hashtag_current] - user_hashtag['retweet_with_comment'][key_hashtag_current])


            train_day[i]['interactions'] = [interaction_users_like, interaction_users_reply, interaction_users_retweet, interaction_users_retweet_with_comment,
                                            num_enaged_like, num_enaged_reply, num_enaged_retweet, num_enaged_retweet_with_comment,
                                            num_enaging_like, num_enaging_reply, num_enaging_retweet, num_enaging_retweet_with_comment,
                                            interaction_counter_lang_like, interaction_counter_lang_reply, interaction_counter_lang_retweet, interaction_counter_lang_retweet_with_comment,
                                            user_hashtag_like, user_hashtag_reply, user_hashtag_retweet, user_hashtag_retweet_with_comment
                                        ] + similar_followers

        with open(filename, 'wb') as handle:
            pickle.dump(train_day, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    with open(params.config_file) as f:
        config = yaml.load(f)
    compute_interactions(config)
    save_interactions(config)
