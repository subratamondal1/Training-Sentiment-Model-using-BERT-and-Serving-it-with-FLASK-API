import torch
import config
import engine
import dataset
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from model import BERTBaseUncased
from transformers import get_linear_schedule_with_warmup

def run():
    df = pd.read_csv(config.TRAINING_FILE).fillna("none")
    df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)

    train, valid = train_test_split(
        df, 
        test_size = 0.2, 
        random_state = 44,
        stratify = df["sentiment"].values # same ratio for all the target classes
        )
    
    train = train.reset_index(drop=True) # resets the index from 0 to len(train)
    valid = valid.reset_index(drop=True) # resets the index from 0 to len(valid)

    train_dataset = dataset.BERTDataset(
        review = train["review"].values,
        target = train["sentiment"].values
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE,
        num_workers = config.TRAIN_NUM_WORKERS
    )

    valid_dataset = dataset.BERTDataset(
        review = valid["review"].values,
        target = valid["sentiment"].values
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config.VAL_BATCH_SIZE,
        num_workers = config.VALID_NUM_WORKERS
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, 
        num_training_steps = num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_dataloader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_dataloader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy
        
if __name__ == "__main__":
    run()
