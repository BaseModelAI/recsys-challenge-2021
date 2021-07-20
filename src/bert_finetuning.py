import os
import argparse
import logging
import yaml
from transformers import DistilBertForMaskedLM, DistilBertTokenizer, DataCollatorForLanguageModeling
import numpy as np
from transformers import Trainer
from transformers import TrainingArguments
from read_dataset_utils import all_features_to_idx
from datasets import Dataset
import random


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config.yaml', help='Configuration file.')
    parser.add_argument("--max-N-parts", type=int, default=1,
                            help='Maximum number of training parts that are taken to bert finetuning.')
    parser.add_argument("--num-epochs", type=int, default=1,  help='Number of epochs.')
    parser.add_argument("--max_num_tokens", type=int, default=100,  help='Maximum number of tokens from tweet.')
    parser.add_argument("--batch-size", type=int, default=16,  help='Batch size.')
    parser.add_argument("--save-steps", type=int, default=100_000,  help=' Every N Steps model checkpoint is saved')
    return parser


def update_data(data, tokens, max_num_tokens):
    data['text'].append('')
    data['input_ids'].append(tokens)
    data['attention_mask'].append(np.array(np.array(tokens) != np.zeros(max_num_tokens), dtype='int'))


def finetune_bert(params):
    with open(params.config_file) as f:
        config = yaml.load(f)

    VALIDATION_PERCENTAGE = 0.02
    BERT_CONFIG_NAME = 'distilbert-base-multilingual-cased'
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_CONFIG_NAME)
    model = DistilBertForMaskedLM.from_pretrained(BERT_CONFIG_NAME).cuda()


    filenames = [os.path.join(config['recsys_data'], f) for f in os.listdir(config['recsys_data']) if 'part' in f][:params.max_N_parts]
    log.info(f"Used input filenames: {filenames}")

    data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            pad_to_multiple_of=8
    )


    training_args = TrainingArguments(output_dir = "./distilbert_checkpoints",
                                    logging_dir='./logs',
                                    logging_steps=0,
                                    warmup_steps=1000,
                                    weight_decay=0.01,
                                    save_strategy = "steps",
                                    evaluation_strategy="steps",
                                    eval_steps = params.save_steps,
                                    save_steps = params.save_steps,
                                    per_device_train_batch_size = params.batch_size,
                                    num_train_epochs = params.num_epochs,
                                    do_predict= True,
                                    dataloader_num_workers=8,
                                    )

    data_train = {"text": [], "input_ids": [], "attention_mask": []}
    data_valid = {"text": [], "input_ids": [], "attention_mask": []}

    for filename in filenames:
        log.info(f'Processing {filename}')
        with open(filename, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                features = line.split("\x01")
                tokens = features[all_features_to_idx['text_ tokens']].split('\t')
                tokens = list(map(int, tokens))
                tokens = tokens[:params.max_num_tokens] + [0]*(params.max_num_tokens-len(tokens))

                if random.uniform(0, 1) > VALIDATION_PERCENTAGE:
                    update_data(data_train, tokens, params.max_num_tokens)
                else:
                    update_data(data_valid, tokens, params.max_num_tokens)

        dataset_train = Dataset.from_dict(data_train)
        dataset_valid = Dataset.from_dict(data_valid)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid,
            data_collator=data_collator
        )
        trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    log.info("Finetuning bert model on tweets")
    finetune_bert(params)
