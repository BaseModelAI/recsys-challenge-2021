# Synerise at ACM Twitter RecSys Challenge 2021


Implementation of our 2nd place solution to [Twitter RecSys Challenge 2021](https://recsys-twitter.com/). The goal of the competition was to predict user engagement with 1 billion tweets selected by Twitter. An additional challenge was the test phase - the models were evaluated in a very constrained test environment - just 1 CPU core, with no GPUs available, and a time limit of 24h for all predictions which gives about 6ms per single tweet prediction.

The challenge focuses on a real-world task of tweet engagement prediction in a dynamic environment. It considers predicting four different engagement types: Likes, Retweet, Quote, and Replies.

## Approach

## Getting Started
1. Register and download training and validation set from [competition webstie](https://recsys-twitter.com/data/show-downloads)

2. Setup a configuration file `config.yaml`:
    * `working_dir` - path where all preprocess files will be saved
    * `recsys_data` - path to directory with uncompressed training data parts
    * `validation_part` - path to validation uncompressed part
    * `max_n_parts` - maximum number of training parts that are taken into training. Limit it for speedup training.
    * `max_n_parts_in_memory` - number of training parts that are loaded into memory at the same time. Limiting it allows to limit RAM usage
    * `authors_similarity_top_N` - denotes maximum number of similar users to current one
    * `authors_similarity_threshold` - users similarity threshold
    * `validation_percentage` - percentage of validation set used for testing, the other part is used for finetuning
    * `num_validation_chunks` - number of validation chunks
    * `sketch_width` - sketch width for tweet sketch
    * `sketch_depth` - sketch depth for tweet sketch

3. Finetune DistilBERT and precompute token sketches

```
    python bert_finetuning.py
```
BERT checkpoints will be saved periodically. You can run sketch computation on any chosen checkpoint:

Prepare sketches of tokens from checkpoint model that was trained with the above script. Change `/distilbert_checkpoints/checkpoint-1000` to the most updated model path.

```
    python prepare_token_embeddings.py --checkpoint-path ./distilbert_checkpoints/checkpoint-1000
```

Apply EMDE to compute sketches
```
    python emde.py
```

Steps 2 and 3 can be run simultaneously

4. Preprocess dataset and compute interactions of users:
```
    python interactions_extraction.py
```

5. Train model
```
    python train.py
```
