from sipHash64 import sipHash64
from read_dataset_utils import all_features_to_idx, labels_to_idx


def get_similar_interactions(auth2similar, engaged_with_user_id_orig, enaging_user_id_orig, interactions):
    similar_authors = auth2similar[engaged_with_user_id_orig]
    interaction_users_similar_like = 0
    interaction_users_similar_reply = 0
    interaction_users_similar_retweet = 0
    interaction_users_similar_retweet_with_comment = 0

    score_all = 0
    for sa in similar_authors:
        author = sa[0]
        score = sa[1]
        key_sim = sipHash64(f'{author}_{enaging_user_id_orig}')
        interaction_users_similar_like += interactions['like'][key_sim] * score
        interaction_users_similar_reply += interactions['reply'][key_sim] * score
        interaction_users_similar_retweet += interactions['retweet'][key_sim] * score
        interaction_users_similar_retweet_with_comment += interactions['retweet_with_comment'][key_sim] * score
        score_all += score

    return [interaction_users_similar_like, interaction_users_similar_reply, interaction_users_similar_retweet, interaction_users_similar_retweet_with_comment, score_all]


def compute_validation_interactions_datapoint(datapoint, interaction_counter_all, num_enaged_all, num_enaging_all, user_hashtag_all,
                                    interaction_counter_lang_all, twit_positive, interaction_counter_all_negative,
                                    num_enaged_all_negative, num_enaging_all_negative, twit_negative):

    def increase_reaction(relation, key, engaged_with_user_id, enaging_user_id, key_hashtags, key_language):
        interaction_counter_all[relation][key] += 1
        num_enaged_all[relation][engaged_with_user_id] += 1
        num_enaging_all[relation][enaging_user_id] += 1

        for key_hashtag_current in key_hashtags:
            user_hashtag_all[relation][key_hashtag_current] +=1
        interaction_counter_lang_all[relation][key_language] += 1

    line = datapoint['line']
    features = line.split("\x01")
    tweet_id = features[all_features_to_idx['tweet_id']]

    engaged_with_user_id = features[all_features_to_idx['engaged_with_user_id']]
    enaging_user_id = features[all_features_to_idx['enaging_user_id']]

    key = sipHash64(f'{engaged_with_user_id}_{enaging_user_id}')

    hashtags = features[all_features_to_idx['hashtags']].split()
    key_hashtags = [sipHash64(f'{h}_{enaging_user_id}') for h in hashtags]

    language = features[all_features_to_idx['language']]
    key_language = sipHash64(f'{language}_{enaging_user_id}')

    engaged_with_user_id = sipHash64(engaged_with_user_id)
    enaging_user_id = sipHash64(enaging_user_id)

    if features[labels_to_idx[f'like_timestamp']] != '':
        increase_reaction('like', key, engaged_with_user_id, enaging_user_id, key_hashtags, key_language)
        twit_positive['like'][tweet_id] += 1

    if features[labels_to_idx[f'reply_timestamp']] != '':
        increase_reaction('reply', key, engaged_with_user_id, enaging_user_id, key_hashtags, key_language)
        twit_positive['reply'][tweet_id] += 1

    if features[labels_to_idx[f'retweet_timestamp']] != '':
        increase_reaction('retweet', key, engaged_with_user_id, enaging_user_id, key_hashtags, key_language)
        twit_positive['retweet'][tweet_id] += 1

    if features[labels_to_idx[f'retweet_with_comment_timestamp']] != '':
        increase_reaction('retweet_with_comment', key, engaged_with_user_id, enaging_user_id, key_hashtags, key_language)
        twit_positive['retweet_with_comment'][tweet_id] += 1

    if features[labels_to_idx[f'like_timestamp']] == '' and features[labels_to_idx[f'reply_timestamp']] == '' and \
            features[labels_to_idx[f'retweet_timestamp']] == '' and features[labels_to_idx[f'retweet_with_comment_timestamp']] == '':

        interaction_counter_all_negative[key] += 1
        num_enaged_all_negative[engaged_with_user_id] += 1
        num_enaging_all_negative[enaging_user_id] += 1
        twit_negative[tweet_id] += 1


def update_interactions_datapoint(datapoint, num_enaged_all, num_enaging_all, interaction_counter_all, interaction_counter_lang_all,
                                    user_hashtag_all, twit_positive, interaction_counter_all_negative, num_enaged_all_negative,
                                    num_enaging_all_negative, twit_negative):
    line = datapoint['line']
    features = line.split("\x01")

    tweet_id = features[all_features_to_idx['tweet_id']]

    engaged_with_user_id_orig = features[all_features_to_idx['engaged_with_user_id']]
    enaging_user_id_orig = features[all_features_to_idx['enaging_user_id']]

    key = sipHash64(f'{engaged_with_user_id_orig}_{enaging_user_id_orig}')
    key2 = sipHash64(f'{enaging_user_id_orig}_{engaged_with_user_id_orig}')

    hashtags = features[all_features_to_idx['hashtags']].split()
    key_hashtags = [sipHash64(f'{h}_{enaging_user_id_orig}') for h in hashtags]
    language = features[all_features_to_idx['language']]
    key_language = sipHash64(f'{language}_{enaging_user_id_orig}')

    engaged_with_user_id = sipHash64(engaged_with_user_id_orig)
    enaging_user_id = sipHash64(enaging_user_id_orig)

    num_enaged_like = num_enaged_all['like'][engaged_with_user_id]
    num_enaged_reply = num_enaged_all['reply'][engaged_with_user_id]
    num_enaged_retweet = num_enaged_all['retweet'][engaged_with_user_id]
    num_enaged_retweet_with_comment = num_enaged_all['retweet_with_comment'][engaged_with_user_id]


    num_enaging_like = num_enaging_all['like'][enaging_user_id]
    num_enaging_reply = num_enaging_all['reply'][enaging_user_id]
    num_enaging_retweet = num_enaging_all['retweet'][enaging_user_id]
    num_enaging_retweet_with_comment = num_enaging_all['retweet_with_comment'][enaging_user_id]

    interaction_users_like = interaction_counter_all['like'][key]
    interaction_users_reply = interaction_counter_all['reply'][key]
    interaction_users_retweet = interaction_counter_all['retweet'][key]
    interaction_users_retweet_with_comment = interaction_counter_all['retweet_with_comment'][key]

    num_enaged_like_reverse = num_enaged_all['like'][enaging_user_id]
    num_enaged_reply_reverse = num_enaged_all['reply'][enaging_user_id]
    num_enaged_retweet_reverse = num_enaged_all['retweet'][enaging_user_id]
    num_enaged_retweet_with_comment_reverse = num_enaged_all['retweet_with_comment'][enaging_user_id]


    num_enaging_like_reverse = num_enaging_all['like'][engaged_with_user_id]
    num_enaging_reply_reverse = num_enaging_all['reply'][engaged_with_user_id]
    num_enaging_retweet_reverse = num_enaging_all['retweet'][engaged_with_user_id]
    num_enaging_retweet_with_comment_reverse = num_enaging_all['retweet_with_comment'][engaged_with_user_id]

    interaction_users_like_reverse = interaction_counter_all['like'][key2]
    interaction_users_reply_reverse = interaction_counter_all['reply'][key2]
    interaction_users_retweet_reverse = interaction_counter_all['retweet'][key2]
    interaction_users_retweet_with_comment_reverse = interaction_counter_all['retweet_with_comment'][key2]


    interaction_counter_lang_like = interaction_counter_lang_all['like'][key_language]
    interaction_counter_lang_reply = interaction_counter_lang_all['reply'][key_language]
    interaction_counter_lang_retweet = interaction_counter_lang_all['retweet'][key_language]
    interaction_counter_lang_retweet_with_comment = interaction_counter_lang_all['retweet_with_comment'][key_language]
    user_hashtag_like = 0
    user_hashtag_reply = 0
    user_hashtag_retweet = 0
    user_hashtag_retweet_with_comment = 0
    for key_hashtag_current in key_hashtags:
        user_hashtag_like += user_hashtag_all['like'][key_hashtag_current]
        user_hashtag_reply += user_hashtag_all['reply'][key_hashtag_current]
        user_hashtag_retweet += user_hashtag_all['retweet'][key_hashtag_current]
        user_hashtag_retweet_with_comment += user_hashtag_all['retweet_with_comment'][key_hashtag_current]


    interactions_valid = [interaction_users_like, interaction_users_reply, interaction_users_retweet, interaction_users_retweet_with_comment,
                                                num_enaged_like, num_enaged_reply, num_enaged_retweet, num_enaged_retweet_with_comment,
                                                num_enaging_like, num_enaging_reply, num_enaging_retweet, num_enaging_retweet_with_comment,
                                            interaction_counter_lang_like, interaction_counter_lang_reply, interaction_counter_lang_retweet, interaction_counter_lang_retweet_with_comment,
                                            user_hashtag_like, user_hashtag_reply, user_hashtag_retweet, user_hashtag_retweet_with_comment,
                                                    twit_positive['like'][tweet_id], twit_positive['reply'][tweet_id],
                                                    twit_positive['retweet'][tweet_id],twit_positive['retweet_with_comment'][tweet_id]
                                        ]

    interactions_valid_negatives = [interaction_users_like_reverse, interaction_users_reply_reverse, interaction_users_retweet_reverse, interaction_users_retweet_with_comment_reverse,
                                            num_enaged_like_reverse, num_enaged_reply_reverse, num_enaged_retweet_reverse, num_enaged_retweet_with_comment_reverse,
                                            num_enaging_like_reverse, num_enaging_reply_reverse, num_enaging_retweet_reverse, num_enaging_retweet_with_comment_reverse,
                                                interaction_counter_all_negative[key], num_enaged_all_negative[engaged_with_user_id], num_enaging_all_negative[enaging_user_id],
                                                twit_negative[tweet_id]
                                        ]
    return interactions_valid, interactions_valid_negatives