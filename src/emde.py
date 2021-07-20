import json
import os
import logging
import argparse
import yaml
import numpy as np
from coders import DLSH


log = logging.getLogger(__name__)



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    return parser


def load_emb(fname):
    ids = []
    vecs = []
    with open(fname) as f:
        next(f)
        for line in f:
            line = line.rstrip('\n')
            line = line.split()
            emb_len = len(line)-2
            idx = line[0]
            vec = np.array(list(map(float, line[-emb_len:])), dtype=np.float32)
            ids.append(idx)
            vecs.append(vec)
    vecs = np.vstack(vecs)
    return ids, vecs


def cs_lsh(embeddings, sketch_width, sketch_depth):
    coder = DLSH(sketch_depth, sketch_width)
    coder.fit(embeddings)
    return coder.transform_to_absolute_codes(embeddings)


def process_file(fname, func):
    codedict = {}
    ids, embeddings = load_emb(fname)
    codes = func(embeddings)
    for i in range(len(ids)):
        codedict[ids[i]] = list(map(int, list(codes[i])))
    return codedict


def encode(config):
    method = lambda x: cs_lsh(x, config['sketch_width'], config['sketch_depth'])
    input_fname = os.path.join(config['working_dir'], 'subwords_bert_pretrained.txt')
    codedict = process_file(input_fname, method)
    fname = os.path.join(config['working_dir'], 'subwords_bert.json')
    with open(fname, 'w') as f:
        json.dump(codedict,f)
    log.info(f"Wrote results to: {fname}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    with open(params.config_file) as f:
        config = yaml.load(f)
    encode(config)

