import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.metrics import AveragePrecision


class Trainer(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = nn.BCEWithLogitsLoss()
        self.train_AP  = {
            'like' :AveragePrecision(),
            'reply' :AveragePrecision(),
            'retweet' :AveragePrecision(),
            'retweet_with_comment' :AveragePrecision()
        }
        self.val_AP = {
            'like' :AveragePrecision(),
            'reply' :AveragePrecision(),
            'retweet' :AveragePrecision(),
            'retweet_with_comment' :AveragePrecision()
        }

    def forward(self, batch):
        return self.model(batch['input'].float(), batch['language'], batch['twitt_hour'], batch['twitt_day'])

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(output, batch['labels'].float())
        self.train_AP['like'](output[:,0], batch['labels'][:,0].long())
        self.train_AP['retweet'](output[:,1], batch['labels'][:,1].long())
        self.train_AP['retweet_with_comment'](output[:,2], batch['labels'][:,2].long())
        self.train_AP['reply'](output[:,3], batch['labels'][:,3].long())

        return loss

    def validation_step(self, batch, batch_idx: int):
        output = self(batch)
        loss = self.loss(output, batch['labels'].float())
        self.val_AP['like'](output[:,0], batch['labels'][:,0].long())
        self.val_AP['retweet'](output[:,1], batch['labels'][:,1].long())
        self.val_AP['retweet_with_comment'](output[:,2], batch['labels'][:,2].long())
        self.val_AP['reply'](output[:,3], batch['labels'][:,3].long())
        return loss

    def test_epoch_end(self, out):
        print(f"""Validation AP like: {self.val_AP['like'].compute().item()}""")
        print(f"""Validation AP retweet: {self.val_AP['retweet'].compute().item()}""")
        print(f"""Validation AP retweet_with_comment: {self.val_AP['retweet_with_comment'].compute().item()}""")
        print(f"""Validation AP reply: {self.val_AP['reply'].compute().item()}""")

        self.val_AP['like'].reset()
        self.val_AP['retweet'].reset()
        self.val_AP['retweet_with_comment'].reset()
        self.val_AP['reply'].reset()

    def test_step(self, batch, batch_idx: int):
        output = self(batch)
        loss = self.loss(output, batch['labels'].float())
        self.val_AP['like'](output[:,0], batch['labels'][:,0].long())
        self.val_AP['retweet'](output[:,1], batch['labels'][:,1].long())
        self.val_AP['retweet_with_comment'](output[:,2], batch['labels'][:,2].long())
        self.val_AP['reply'](output[:,3], batch['labels'][:,3].long())
        return loss

    def validation_epoch_end(self, out):
        print(f"""Validation AP like: {self.val_AP['like'].compute().item()}""")
        print(f"""Validation AP retweet: {self.val_AP['retweet'].compute().item()}""")
        print(f"""Validation AP retweet_with_comment: {self.val_AP['retweet_with_comment'].compute().item()}""")
        print(f"""Validation AP reply: {self.val_AP['reply'].compute().item()}""")

        self.val_AP['like'].reset()
        self.val_AP['retweet'].reset()
        self.val_AP['retweet_with_comment'].reset()
        self.val_AP['reply'].reset()

    def training_epoch_end(self, losses):
        print(f"""Training loss: {np.mean([i['loss'].item() for i in losses])}""")
        print(f"""Training AP like: {self.train_AP['like'].compute().item()}""")
        print(f"""Training AP retweet: {self.train_AP['retweet'].compute().item()}""")
        print(f"""Training AP retweet_with_comment: {self.train_AP['retweet_with_comment'].compute().item()}""")
        print(f"""Training AP reply: {self.train_AP['reply'].compute().item()}""")

        self.train_AP['like'].reset()
        self.train_AP['retweet'].reset()
        self.train_AP['retweet_with_comment'].reset()
        self.train_AP['reply'].reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        return optimizer
