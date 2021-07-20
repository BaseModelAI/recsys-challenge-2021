import os
import logging
import argparse
import random
import pickle as pickle
import json
import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model.net import Model
from model.trainer import Trainer
from twitter_dataset import TwitterDataset


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    parser.add_argument("--hidden-size", type=int, default=1500, help='Hidden size of neural network')
    parser.add_argument("--batch-size", type=int, default=256, help='Batch size')
    parser.add_argument("--learning-rate", type=int, default=1e-4, help='Start learning rate')
    parser.add_argument("--decay", type=float, default=0.96, help='Learning rate decay')
    parser.add_argument("--validation-decay", type=float, default=0.5, help='Learning rate decay for training on validation part')
    parser.add_argument("--num-validation-epochs", type=int, default=1, help='Number of epochs for training on validation part')
    return parser


def train(params):
    log.info("Loading data")
    with open(params.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(f"{config['working_dir']}/language2id", 'rb') as handle:
        language2id = pickle.load(handle)

    subword_codes_name = f"{config['working_dir']}/subwords_bert.json"
    with open(subword_codes_name) as f:
        subword_codes = json.load(f)

    with open(os.path.join(config['working_dir'], 'validation_split_holdout'), 'rb') as handle:
        valid_day = pickle.load(handle)

    valid_dataset = TwitterDataset(valid_day, subword_codes, language2id, config['sketch_width'], config['sketch_depth'], labels_available=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, num_workers=15, shuffle=False)


    EMBEDDING_SIZE = 20 # size of embeddings for day, language and hour
    NUM_CATEGORICAL_FEATURES = 12 # number of categortical features created in TwitterDataset
    NUM_NUMERICAL_FEATURES = 73 # number of numerical features created in TwitterDataset

    net = Model(params.hidden_size, len(language2id), EMBEDDING_SIZE, NUM_CATEGORICAL_FEATURES, NUM_NUMERICAL_FEATURES, config['sketch_depth'], config['sketch_width'])
    lr = params.learning_rate
    model = Trainer(net, lr)
    trainer = pl.Trainer(gpus=1,  max_epochs=1, logger=False, checkpoint_callback=False)

    log.info("Start training")
    filenames = [os.path.join(config['working_dir'], f) for f in os.listdir(config['working_dir']) if 'train_set_all' in f]
    for training_filename in filenames:
        log.info(f"Training on {training_filename}")
        model.learning_rate = lr
        with open(training_filename, 'rb') as handle:
            traininig_part = pickle.load(handle)

        train_dataset = TwitterDataset(traininig_part, subword_codes, language2id, config['sketch_width'], config['sketch_depth'], labels_available=True)
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=15, shuffle=True)
        trainer.fit(model, train_loader)
        lr = lr * params.decay

    trainer.test(model, valid_loader)

    train_day = []
    for chunk_idx in range(config['num_validation_chunks']):
        with open(os.path.join(config['working_dir'], f"validation_train_with_valid_interactions_{chunk_idx}"), 'rb') as handle:
            train_day_current = pickle.load(handle)
        train_day += train_day_current
    random.shuffle(train_day)

    log.info("Finetunning on validation set")
    lr = params.learning_rate
    train_dataset = TwitterDataset(train_day, subword_codes, language2id, config['sketch_width'], config['sketch_depth'], labels_available=True)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=15, shuffle=True)

    for _ in range(params.num_validation_epochs):
        model.learning_rate = lr
        trainer.fit(model, train_loader)
        lr = lr * params.validation_decay

    trainer.test(model, valid_loader)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    train(params)
