from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import torch
from collections import OrderedDict, defaultdict
import os
import random
from itertools import chain
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import time
import pickle as pkl
from typing import List, Dict, Any


# set up the directory
tmp_model_dir = 'tmp_models'
if not os.path.exists(tmp_model_dir):
    os.mkdir(tmp_model_dir)

NUM_FOLD = 4 # split the data into 4 folds. train on three to predict on the fourth
bsize = 16 # batch size to fine-tune the Roberta model
NUM_STEPS = 2000 # number of fine-tuning steps
max_length = 128 # max length of the input text
DEBUG = False

# hyperparameters for debugging
if DEBUG:
    NUM_STEPS = 300
    NUM_FOLD = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
pretrain_model = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
lsm = torch.nn.LogSoftmax(dim=-1)

# create cross validation folds
# where each fold is represented by a set of training and test A and B samples
# Usually A_samples are the research split of Corpus A and B_samples are the research split of Corpus B
# K is the number of folds, usually set to 4
def cv(A_samples: List[str], B_samples: List[str], K: int) -> List[Dict[str, List[str]]]:
    return [
        {
            "train_A": [p for i, p in enumerate(A_samples) if i % K != k],
            "train_B": [n for i, n in enumerate(B_samples) if i % K != k],
            "test_A": [p for i, p in enumerate(A_samples) if i % K == k],
            "test_B": [n for i, n in enumerate(B_samples) if i % K == k],
        }
        for k in range(K)
    ]

# fine-tune a Roberta model on the training samples from the cross validation fold
# return the model
def train(cv_dict: Dict[str, List[str]]) -> AutoModelForSequenceClassification:
    train_data_dicts = list(
        chain(
            [{"input": x, "label": 1} for x in cv_dict["train_A"]],
            [{"input": x, "label": 0} for x in cv_dict["train_B"]],
        )
    )

    model = AutoModelForSequenceClassification.from_pretrained(pretrain_model).to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, 400, NUM_STEPS)
    model.train()

    for step in tqdm.trange(NUM_STEPS):
        random.shuffle(train_data_dicts)
        input_texts = [d["input"] for d in train_data_dicts[:bsize]]
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        labels = torch.tensor([d["label"] for d in train_data_dicts[:bsize]]).to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        if step % 2 == 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return model


# evaluate the model on the test sample
# return the logits for each sample
# the shape of the logits is (num_samples, 2)
# where the first column is the logit for the B class and the second column is the logit for the A class
def evaluate(texts: List[str], model: AutoModelForSequenceClassification) -> np.ndarray:
    model.eval()
    all_logits, all_highlights = [], []
    cur_start = 0
    while cur_start < len(texts):
        texts_ = texts[cur_start : cur_start + bsize]
        inputs = tokenizer(
            texts_,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        logits = model(**inputs).logits
        logits = lsm(logits.detach().cpu()).numpy().tolist()
        all_logits.extend(logits)
        cur_start += bsize
    assert len(all_logits) == len(texts)

    return np.array(all_logits)


# train the model on the training samples from the cross validation fold
# and then evaluate the model on the test samples from the cross validation fold
def train_and_eval(cv_dict: Dict[str, List[str]]) -> Dict[str, Any]:
    model = train(cv_dict)
    A_eval_logits = evaluate(cv_dict["test_A"], model)
    B_eval_logits = evaluate(cv_dict["test_B"], model)

    all_logits_A = np.concatenate((A_eval_logits, B_eval_logits), axis=0)[:,1]
    all_labels = np.concatenate((np.ones(len(A_eval_logits)), np.zeros(len(B_eval_logits))), axis=0)

    auc = roc_auc_score(all_labels, all_logits_A)

    return {
            "test_A_scores": A_eval_logits[:, 1],
            "test_B_scores": B_eval_logits[:, 0],
            "auc_roc": auc,
            'model': model
        }

# A_samples are usually the research split of the Corpus A
# B_samples are usually the research split of the Corpus B
def return_extreme_values(A_samples: List[str], B_samples: List[str]):
    A_sample2score, B_sample2score = {}, {}
    text2model_path = {}
    clf_scores = {}

    for fold_idx, cv_dict in enumerate(cv(A_samples, B_samples, NUM_FOLD)):
        train_and_eval_result = train_and_eval(cv_dict)
        model = train_and_eval_result['model']
        model_tmp_path = os.path.join(tmp_model_dir, f"model_{fold_idx}_{int(time.time())}.pt")
        for A_sample, score in zip(cv_dict["test_A"], train_and_eval_result["test_A_scores"]):
            A_sample2score[A_sample] = score
            text2model_path[A_sample] = model_tmp_path
        for B_sample, score in zip(cv_dict["test_B"], train_and_eval_result["test_B_scores"]):
            B_sample2score[B_sample] = score
            text2model_path[B_sample] = model_tmp_path
        clf_scores[model_tmp_path] = train_and_eval_result["auc_roc"]
        print(f"fold {fold_idx} done, auc: {train_and_eval_result['auc_roc']}")

    return {
        'clf_scores': clf_scores, # a mapping from model path to the AUC score for each fold, useful to tell how easy it is to separate the two corpora
        'A_sample2score': A_sample2score, # a mapping from the A sample to how representative each sample is for the A corpus
        'B_sample2score': B_sample2score, # a mapping from the B sample to how representative each sample is for the B corpus
        'sorted_A_samples': sorted(A_sample2score.keys(), key=A_sample2score.get, reverse=True), # sorted A samples by their scores
        'sorted_B_samples': sorted(B_sample2score.keys(), key=B_sample2score.get, reverse=True) # sorted B samples by their scores
    }


if __name__ == "__main__":
    example_problem = pkl.load(open('example_problem.pkl', 'rb'))
    A_samples, B_samples = example_problem['split']['research']['A_samples'], example_problem['split']['research']['B_samples']
    extreme_values = return_extreme_values(A_samples, B_samples)

    print('======== Most representative A samples:')
    for sample in extreme_values['sorted_A_samples'][:5]:
        print(sample)
    
    print('======== Most representative B samples:')
    for sample in extreme_values['sorted_B_samples'][:5]:
        print(sample)

    print('Average AUC score for the 4 folds:', np.mean(list(extreme_values['clf_scores'].values())))


    