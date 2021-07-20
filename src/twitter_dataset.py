import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from datetime import datetime
from read_dataset_utils import all_features_to_idx, labels_to_idx
from fourier_feature_encoding import encode_scalar_column


class TwitterDataset(Dataset):
    def __init__(self, data, subword_codes, language2id, sketch_width, sketch_depth, labels_available):
        self.data = data
        self.labels_available = labels_available
        self.subword_codes = subword_codes
        self.language2id = language2id
        self.sketch_width = sketch_width
        self.sketch_depth = sketch_depth

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]['line'].split("\x01")
        num_tokens = len(features[all_features_to_idx['text_ tokens']].split('\t'))
        subwords = features[all_features_to_idx['text_ tokens']].split('\t')

        # calcualte tweet tokens sketch
        subword_sketch = np.zeros([self.sketch_width * self.sketch_depth])
        for sw in subwords[1:-1]:
            if sw in self.subword_codes.keys():
                subword_sketch[self.subword_codes[sw]] += 1
        subword_sketch = normalize(subword_sketch.reshape(-1, self.sketch_width), 'l2').reshape((self.sketch_width * self.sketch_depth,))


        interactions = self.data[idx]['interactions'] + self.data[idx]['interactions_valid'] + self.data[idx]['interactions_valid_negatives']
        num_hashtags = len(features[all_features_to_idx['hashtags']].split())

        engaged_with_user_follower_count = int(features[all_features_to_idx['engaged_with_user_follower_count']])
        engaged_with_user_following_count = int(features[all_features_to_idx['engaged_with_user_following_count']])
        enaging_user_follower_count = int(features[all_features_to_idx['enaging_user_follower_count']])
        enaging_user_following_count = int(features[all_features_to_idx['enaging_user_following_count']])

        engaged_with_user_is_verified = 1 if features[all_features_to_idx['engaged_with_user_is_verified']] == 'true' else 0
        enaging_user_is_verified = 1 if features[all_features_to_idx['enaging_user_is_verified']] == 'true' else 0

        if features[all_features_to_idx['language']] in self.language2id:
            language = self.language2id[features[all_features_to_idx['language']]]
        else:
            language = len(self.language2id)

        is_hashtag = 1 if features[all_features_to_idx['hashtags']] != '' else 0
        media = features[all_features_to_idx['present_media']]
        media = set(media.split('\t'))
        is_gif = 1 if 'GIF' in media else 0
        is_photo = 1 if 'Photo' in media else 0
        is_video = 1 if 'Video' in media else 0
        is_link = 1 if features[all_features_to_idx['present_links']] != '' else 0
        is_domain = 1 if features[all_features_to_idx['present_domains']] != '' else 0
        tweet_type = features[all_features_to_idx['tweet_type']]
        is_quote = 1 if tweet_type == 'Quote' else 0
        is_retweet = 1 if tweet_type == 'Retweet' else 0
        is_top_level = 1 if tweet_type == 'TopLevel' else 0


        twitt_date = datetime.fromtimestamp(int(features[all_features_to_idx['tweet_timestamp']]))
        enaging_account_creation_diff = (twitt_date - datetime.fromtimestamp(int(features[all_features_to_idx['enaging_user_account_creation']]))).days
        enaged_account_creation_diff = (twitt_date - datetime.fromtimestamp(int(features[all_features_to_idx['engaged_with_user_account_creation']]))).days

        twitt_hour = twitt_date.hour
        twitt_day = twitt_date.day-1 # indexing days from 0


        if self.labels_available:
            label_like = 1 if features[labels_to_idx['like_timestamp']] != '' else 0
            label_retweet = 1 if features[labels_to_idx['retweet_timestamp']] != '' else 0
            label_retweet_with_comment = 1 if features[labels_to_idx['retweet_with_comment_timestamp']] != '' else 0
            label_reply = 1 if features[labels_to_idx['reply_timestamp']] != '' else 0
            labels = torch.LongTensor([label_like, label_retweet, label_retweet_with_comment, label_reply])


        features_interactions = encode_scalar_column(np.array(interactions +
                                    [num_hashtags, num_tokens, engaged_with_user_follower_count, engaged_with_user_following_count,
                                    enaging_user_follower_count, enaging_user_following_count, enaging_account_creation_diff,
                                    enaged_account_creation_diff])).flatten()
        engagee_follows_engager = 1 if features[all_features_to_idx['engagee_follows_engager']] == 'true' else 0

        result = {
            'input': np.concatenate((subword_sketch,
                                     features_interactions,
                                     [engagee_follows_engager, engaged_with_user_is_verified, enaging_user_is_verified,
                                     is_hashtag, is_gif, is_photo, is_video, is_link, is_domain, is_quote, is_retweet, is_top_level])),
            'language': language,
            'twitt_hour': twitt_hour,
            'twitt_day': twitt_day
        }

        if self.labels_available:
            result['labels'] = labels
        return result