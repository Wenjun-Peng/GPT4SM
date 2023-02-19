import os
import math
import json
import wandb
import random
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import Tuple
from scipy import stats

import torch
from torch import nn
from torch.utils.data import DataLoader

import datasets
from datasets import load_dataset, DatasetDict
from dataset.emb_cache import load_gpt_embeds
from dataset.load_utils import load_func
from dataset.collator import SiameseBERTCollator, SiameseGPTCollator, BERTwithGPTEmbCollator
from model.siameseBERT import VanillaBERT, ContrastiveBERT, AdditiveBERT, AdditiveBERTwithGPTEmb, BERTwithGPTEmbToken, BERTwithGPTRegCopy, BERTwithGPTRegCopyEmb, BERTwith0Epoch
from model.siameseGPT import VanillaGPT, ContrastiveGPT, ResHeadGPT, CosGPT
from metrics import calc_ndcg
import evaluate

from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    SchedulerType,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
)
import hashlib

from sklearn.metrics import ndcg_score, roc_auc_score, accuracy_score

logger = get_logger(__name__)


MODEL_TYPE = {
    'ContrastiveBERT': [ContrastiveBERT, SiameseBERTCollator, 'cls'],
    'VanillaBERT': [VanillaBERT, SiameseBERTCollator, 'cls'],
    'AdditiveBERT': [AdditiveBERT, SiameseBERTCollator, 'cls'],
    'AdditiveBERTwithGPTEmb': [AdditiveBERTwithGPTEmb, BERTwithGPTEmbCollator, 'cls'],
    'BERTwithGPTEmbToken': [BERTwithGPTEmbToken, BERTwithGPTEmbCollator, 'cls'],
    'BERTwithGPTReg': [BERTwithGPTRegCopy, BERTwithGPTEmbCollator, 'reg'],
    'BERTwithGPTCopy': [BERTwithGPTRegCopy, BERTwithGPTEmbCollator, 'copy'],
    'BERTwithGPTRegEmb': [BERTwithGPTRegCopyEmb, BERTwithGPTEmbCollator, 'reg'],
    'BERTwithGPTCopyEmb': [BERTwithGPTRegCopyEmb, BERTwithGPTEmbCollator, 'copy'],
    'BERTwith0Epoch': [BERTwith0Epoch, BERTwithGPTEmbCollator, 'cls'],
    'CosGPT': [CosGPT, BERTwithGPTEmbCollator, 'cls'],
    'ContrastiveGPT': [ContrastiveGPT, BERTwithGPTEmbCollator, 'cls'],
    'VanillaGPT': [VanillaGPT, BERTwithGPTEmbCollator, 'cls'],
    'ResHeadGPT': [ResHeadGPT, BERTwithGPTEmbCollator, 'cls']
}

DATA_INFO = {
    "ads": {
        "dataset_name": "ads",
        "dataset_config_name": None,
        "text1": "query",
        "text2": "keywords",
        "idx": "idx",
        "remove": ["label", "query", "keywords", "idx"],
    },

    "blue": {
        "dataset_name": "blue",
        "dataset_config_name": None,
        "text1": "query",
        "text2": "title",
        "idx": "idx",
        "remove": ["label", "query", "title", "idx"],
    },
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )

    parser.add_argument(
        "--job_name", type=str, default=None, help="The job name used for wandb logging"
    )
    
    # GPT3 configuration
    parser.add_argument(
        "--gpt_emb_dim", type=int, default=1536, help="The embedding size of gpt3."
    )
    parser.add_argument(
        "--gpt_emb_train_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 train set.",
    )
    parser.add_argument(
        "--gpt_emb_test_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 test set.",
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="The train file of mind train set.",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="The test file of mind train set.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        help="Base model or model identifier from models.",
        required=True,
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--cross_gpt_emb",
        action="store_true",
        help="Whether to enable cross gpt emb concat.",
    )

    parser.add_argument(
        "--additive_pooling",
        action="store_true",
        help="Whether to enable cross gpt emb concat.",
    )

    parser.add_argument(
        "--res_head",
        action="store_true",
        help="Whether to enable cross gpt emb concat.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--cls_hidden_size",
        type=int,
        default=256,
        help="The dimention of transform hidden layer.",
    )

    parser.add_argument(
        "--match_hidden_size",
        type=int,
        default=3072,
        help="The dimention of transform hidden layer.",
    )

    parser.add_argument(
        "--match_output_size",
        type=int,
        default=1536,
        help="The dimention of transform hidden layer.",
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="The dropout rate of transformation layer.",
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=1,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--bce_weight",
        type=float,
        default=1,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--temperature",
        type=int,
        default=None,
        help="Training temperature.",
    )

    parser.add_argument(
        "--use_snippet",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    parser.add_argument(
        "--data_name", type=str, default="ads", help="dataset name for training."
    )

    parser.add_argument(
        "--project_name", type=str, default=None, help="project name for training."
    )
    args = parser.parse_args()

    return args

def eval_relevance(
    args,
    model,
    total_loss,
    epoch,
    completed_steps,
    train_dataloader,
    eval_dataloader,
    accelerator
):
    model.eval()

    tokenizer = eval_dataloader.collate_fn.tokenizer
    queries = []
    ce_target = []
    ce_logits = []
    encoded_text1 = []
    encoded_text2 = []
    ce_losses = []
    contrastive_losses = []
    # Compute clean to target and to gpt distance
    for step, batch in tqdm(enumerate(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)
            ce_logits.append(outputs.ce_logits.cpu())
            encoded_text1.append(outputs.encoded_text1.cpu())
            encoded_text2.append(outputs.encoded_text2.cpu())
            ce_target.append(batch['labels'].cpu())
            
            str_query = [tokenizer.decode(q, skip_special_tokens=True) for q in batch['text1']['input_ids']]
            queries.extend(str_query)

            if outputs.ce_loss is not None:
                ce_losses.append(outputs.ce_loss.cpu())
            
            if outputs.contrastive_loss is not None:
                contrastive_losses.append(outputs.contrastive_loss.cpu())
    
    ce_logits = torch.cat(ce_logits, dim=0)
    ce_target = torch.cat(ce_target, dim=0)

    if len(ce_logits.size()) == 1:
        ce_logits = torch.sigmoid(ce_logits)
        preds = torch.round(ce_logits)
    else:
        ce_logits = torch.softmax(ce_logits, dim=-1)
        _, preds = torch.max(ce_logits, dim=-1)

    # print(preds)
    # print(ce_target)

    if len(ce_losses) > 0:
        ce_losses = torch.stack(ce_losses, dim=0)
    else:
        ce_losses = torch.zeros((1, 1))

    if len(contrastive_losses) > 0:
        contrastive_losses = torch.stack(contrastive_losses, dim=0)
    else:
        contrastive_losses = torch.zeros((1, 1))
    encoded_text1 = torch.cat(encoded_text1, dim=0)
    encoded_text2 = torch.cat(encoded_text2, dim=0)
    # relevance_score = torch.matmul(encoded_text1, encoded_text2.t())
    data4ndcg = {
        "Query": queries, 
        "Label": ce_target.numpy().tolist(), 
        "Score": preds.numpy().tolist(),
    }
    data4ndcg = pd.DataFrame(data4ndcg)
    dcg_dict = calc_ndcg(data4ndcg)

    if args.model_type != 'CosGPT':
        auc = roc_auc_score(ce_target, ce_logits, multi_class='ovr')
        acc = accuracy_score(ce_target, preds)
    else:
        auc = 0
        acc = 0

    # ndcg = ndcg_score(relevance_target, relevance_score)
    results = {
        'AUC': auc,
        'ACC': acc,
    }

    for key in dcg_dict:
        results[key] = dcg_dict[key]

    logger.info(
        f"epoch {epoch}: {results}, train_loss: {total_loss.item() / len(train_dataloader)}"
    )

    if args.with_tracking:
        accelerator.log(
            {
                "metrics": results,
                "train_loss": total_loss.item() / len(train_dataloader),
                "bce_loss": ce_losses.mean().item(),
                "contrastive_loss": contrastive_losses.mean().item(),
            },
            step=completed_steps,
            log_kwargs={"wandb": {"commit": False}},
        )
    return results


def train_relevance(
    args,
    model,
    train_dataset,
    train_dataloader,
    eval_dataloader,
    accelerator,
    learning_rate,
    gradient_accumulation_steps,
    max_train_steps,
    num_train_epochs,
    num_warmup_steps,
    completed_steps=0,
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
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


    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running copier training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            
            # check unused params
            # for n, p in model.named_parameters():
            #     if p.grad is None:
            #         print(f'{n} has no grad')

            if (
                step % gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= max_train_steps:
                break

        eval_metric = eval_relevance(
            args,
            model,
            total_loss,
            epoch,
            completed_steps,
            train_dataloader,
            eval_dataloader,
            accelerator
        )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        output_dir = args.output_dir
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(all_results, f)

    return completed_steps, eval_metric


def main():
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir, kwargs_handlers=[ddp_kwargs])
        if args.with_tracking
        else Accelerator(kwargs_handlers=[ddp_kwargs])
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(args)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load raw dataset
    raw_datasets = load_func(
        train_tsv_path=args.train_file,
        test_tsv_path=args.test_file,
        name=args.data_name,
        use_snippet=args.use_snippet
    )

    label_list = list(set(raw_datasets["test"]["label"]))
    num_labels = len(label_list)
    print(label_list)
    # binary class can just output logits
    if len(label_list) == 2:
        num_labels = 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
    )
    config.num_labels = num_labels
    
    model_class, collator_class, reg_copy_mode = MODEL_TYPE[args.model_type]
    model = model_class.from_pretrained(config=config, args=args)
    model.reg_copy_mode = reg_copy_mode

    data_collator = collator_class(tokenizer)

    # Preprocess Dataset
    emb_caches = load_gpt_embeds(
        args,
        args.gpt_emb_train_file,
        args.gpt_emb_test_file,
    )

    emb_caches.open()

    padding = "max_length" if args.pad_to_max_length else False
    def process_func(examples, key):
        text1 = examples[DATA_INFO[args.data_name]["text1"]]
        text2 = examples[DATA_INFO[args.data_name]["text2"]]
        
        idx_name = DATA_INFO[args.data_name]["idx"]
        if idx_name == "md5":
            idx_byte = hashlib.md5(
                examples[DATA_INFO[args.data_name]["text"]].encode("utf-8")
            ).digest()
            idx = int.from_bytes(idx_byte, "big")
        else:
            idx = examples[idx_name]
        
        
        tok_text1 = tokenizer(
            text1, padding=padding, max_length=args.max_length, truncation=True
        )

        tok_text2 = tokenizer(
            text2, padding=padding, max_length=args.max_length, truncation=True
        )
        # print(idx)
        # check_emb = torch.as_tensor(emb_caches[key][idx][1])
        # new_check_emb = check_emb / torch.norm(check_emb, p=2, dim=0, keepdim=True)
        # print(check_emb)
        # print(new_check_emb)
        result = {}
        result["text1_emb"] = emb_caches[key][idx][0]
        result["text2_emb"] = emb_caches[key][idx][1]
        result["text1"] = tok_text1
        result["text2"] = tok_text2
        result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = DatasetDict(
            {
                k: dataset.map(
                    partial(process_func, key=k),
                    remove_columns=DATA_INFO[args.data_name]["remove"],
                    desc="Run tokenization and add gpt3 embeddings on dataset",
                )
                for k, dataset in raw_datasets.items()
            }
        )
    
    logger.info(processed_datasets)
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    # for index in random.sample(range(len(train_dataset)), 1):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value

        init_kwargs = None
        if args.job_name is not None:
            init_kwargs = {"wandb": {"name": args.job_name}}

        if args.project_name is not None:
            project_name = args.project_name
        else:
            project_name = args.data_name + '_' + args.model_type

        accelerator.init_trackers(
            project_name,
            experiment_config,
            init_kwargs=init_kwargs,
        )

    if args.model_type != 'CosGPT':
        train_relevance(
            args,
            model,
            train_dataset,
            train_dataloader,
            eval_dataloader,
            accelerator,
            args.learning_rate,
            args.gradient_accumulation_steps,
            args.max_train_steps,
            args.num_train_epochs,
            args.num_warmup_steps,
            completed_steps=0,
        )
    
    else:
        total_loss = torch.zeros(1)
        epoch=0
        completed_steps=1
        eval_metric = eval_relevance(
            args,
            model,
            total_loss,
            epoch,
            completed_steps,
            train_dataloader,
            eval_dataloader,
            accelerator
        )
    
    if 'copy' in args.model_type.lower():
        model.reg_copy_mode = 'cls'
        cls_train_epochs = 3
        train_relevance(
            args,
            model,
            train_dataset,
            train_dataloader,
            eval_dataloader,
            accelerator,
            args.learning_rate,
            args.gradient_accumulation_steps,
            args.max_train_steps,
            cls_train_epochs,
            args.num_warmup_steps,
            completed_steps=0,
        )
        
    if accelerator.is_main_process:
        if args.with_tracking and args.report_to != "wandb":
            accelerator.end_training()


if __name__ == '__main__':
    main()
