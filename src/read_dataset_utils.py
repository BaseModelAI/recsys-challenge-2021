all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
               "enaging_user_account_creation", "engagee_follows_engager"]


all_features_to_idx = dict(zip(all_features, range(len(all_features))))


labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23}