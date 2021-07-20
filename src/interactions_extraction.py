import argparse
import logging
import yaml
from preprocessing.split_by_date import split_training_parts
from preprocessing.authors_similarity import find_similar_authors
from preprocessing.compute_interactions import compute_interactions, save_interactions
from preprocessing.split_validation import split_validation
from preprocessing.add_training_interactions_to_valid import add_interactions
from preprocessing.compute_validation_interactions import compute_validation_interactions
from preprocessing.add_validation_interactions_to_training import add_validation_interactions
from preprocessing.valid_interactions_per_chunk import add_interactions_per_chunk


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    return parser


def interactions_extraction(params):
    with open(params.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    split_training_parts(config)
    find_similar_authors(config)
    compute_interactions(config)
    save_interactions(config)
    split_validation(config)
    add_interactions(config)
    compute_validation_interactions(config)
    add_validation_interactions(config)
    add_interactions_per_chunk(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    interactions_extraction(params)
