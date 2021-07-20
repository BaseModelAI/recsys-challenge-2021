"""
Finding similar users to each one based on their followers
"""

import argparse
import os
import pickle
import logging
import yaml
from read_dataset_utils import all_features_to_idx
from collections import defaultdict
from tqdm import tqdm
from SetSimilaritySearch import SearchIndex


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    return parser


def compute_most_similar(authors2followers, threshold, topN):
    log.info("Computting authors similarity")
    author_list = list(authors2followers.keys())
    index = SearchIndex(list(authors2followers.values()), similarity_func_name="jaccard", similarity_threshold=threshold)

    auth2similar = defaultdict(list)

    for idx, auth in tqdm(enumerate(author_list)):
        most_similar = index.query(authors2followers[auth])
        most_similar = sorted(most_similar, key = lambda x: x[1], reverse=True)

        for s in most_similar[:topN]:
            if s[0] != idx:
                auth2similar[auth].append((author_list[s[0]], s[1]))
    return auth2similar


def find_similar_authors(config):
    filenames = [os.path.join(config['working_dir'], f) for f in os.listdir(config['working_dir']) if 'train_set' in f]

    auth2follow_all = defaultdict(set)

    for filename in filenames:
        log.info(f"Processing {filename}")
        with open(filename, 'rb') as handle:
            train_day = pickle.load(handle)

        for i in range(len(train_day)):
            line = train_day[i]['line']
            features = line.split("\x01")

            engaged_with_user_id = features[all_features_to_idx['engaged_with_user_id']]
            enaging_user_id = features[all_features_to_idx['enaging_user_id']]

            # users who follow
            if features[all_features_to_idx[f'engagee_follows_engager']] == 'true' :
                auth2follow_all[engaged_with_user_id].add(enaging_user_id)


    auth2similar_follow = compute_most_similar(auth2follow_all, config['authors_similarity_threshold'], config['authors_similarity_top_N'])

    with open(f"""{config['working_dir']}/author2similar_follow.pkl""", 'wb') as handle:
        pickle.dump(auth2similar_follow, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    with open(params.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    find_similar_authors(config)
