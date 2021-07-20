import torch
import os
import argparse
import logging
import yaml
import numpy as np
from collections import defaultdict
from transformers import DistilBertForMaskedLM
from tqdm import tqdm
from read_dataset_utils import all_features_to_idx


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    parser.add_argument("--checkpoint-path", type=str, required=True, help='Path to bert checkpoint model')
    parser.add_argument("--max-N-parts", type=int, default=1,
                            help='Maximum number of training parts that are taken to bert finetuning.')
    return parser


def prepare_sketches(params):
    BERT_EMBEDDING_SIZE = 768
    with open(params.config_file) as f:
        config = yaml.load(f)

    model = DistilBertForMaskedLM.from_pretrained(params.checkpoint_path).cuda()
    filenames = [os.path.join(config['recsys_data'], f) for f in os.listdir(config['recsys_data']) if 'part' in f][:params.max_N_parts]
    log.info(f"Used input filenames: {filenames}")

    bert_tokens = defaultdict(lambda: np.zeros(BERT_EMBEDDING_SIZE))
    token_occurences = defaultdict(int)

    for filename in filenames:
        with open(filename, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                features = line.split("\x01")

                tokens = features[all_features_to_idx['text_ tokens']].split('\t')
                tokens = list(map(int, tokens))

                embeddings = model(torch.tensor(tokens).unsqueeze(0).cuda(), output_hidden_states=True).hidden_states
                embeddings = embeddings[-1].squeeze()

                for k, v in zip(tokens, embeddings):
                    v = v.to('cpu').detach().numpy()
                    bert_tokens[k] += v
                    token_occurences[k] += 1

    for k, v in bert_tokens.items():
        v /= token_occurences[k]
        bert_tokens[k] = v

    log.info(f"Number of tokens: {len(bert_tokens)}")
    os.makedirs(os.path.join(config['working_dir']), exist_ok=True)
    with open(os.path.join(config['working_dir'], 'subwords_bert_pretrained.txt'), 'w') as f:
        f.write('control line\n')
        for k, v in bert_tokens.items():
            f.write(str(k) + ' ' + '-1' + ' ')
            f.write(' '.join(map(str, v)))
            f.write('\n')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    log.info("Finetuning bert model on tweets")
    prepare_sketches(params)
