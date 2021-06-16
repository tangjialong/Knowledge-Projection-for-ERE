# -*- coding:utf-8 -*-
import os
import logging
import numpy as np
import argparse
import math
from tqdm import tqdm, trange
from sklearn.metrics import classification_report

import torch
from torch.nn import CrossEntropyLoss
from pytorch_transformers import BertTokenizer, BertForSequenceClassification

from base.bert_utils import prepare_optimizer, prepare_model, prepare_env
from base.bert_utils import save_model, load_model
from base.bert_utils import p_r_f1, log_format
from base.bert_utils import MultiTaskBertForSequenceClassification
from base.data_loader import get_data_loader, get_processor
from base.io_utils import convert_examples_to_features

logger = logging.getLogger(__name__)

def compute_metrics(preds, labels, average='macro'):
    assert len(preds) == len(labels)
    return p_r_f1(preds, labels, average)

def eval_model(model, eval_data_loader, label_list, eval_batch_size, output_dir, device, prefix=""):
    num_labels = len(label_list)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_data_loader.dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    model.eval()
    preds = []
    labels = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_data_loader, desc="Evaluating"):
        labels += [label_ids]

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            if 'IERE' in prefix:
                logits = model(input_ids, segment_ids, input_mask, labels=None, task='IERE')
            elif 'IDRR' in prefix:
                logits = model(input_ids, segment_ids, input_mask, labels=None, task='IDRR')
            if isinstance(logits, tuple):
                logits = logits[0]

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    labels = torch.cat(labels)

    logger.info('\n' + classification_report(labels.numpy(), preds, target_names=label_list))
    result = compute_metrics(preds, labels.numpy(), 'macro')
    logger.info(result)
    
    return result

def train_model(model,
                tokenizer,
                IERE_train_data_loader,
                IDRR_train_data_loader,
                IERE_eval_data_loader,
                IDRR_eval_data_loader,
                IERE_label_list,
                IDRR_label_list,
                args,
                optimizer,
                env_option
                ):
    device = env_option['device']

    IERE_num_labels = len(IERE_label_list)
    IDRR_num_labels = len(IDRR_label_list)
    logger.info("***** Running training *****")
    logger.info("  IERE Num examples = %d", len(IERE_train_data_loader.dataset))
    logger.info("  IDRR Num examples = %d", len(IDRR_train_data_loader.dataset))
    logger.info("  Batch size = %d", args.train_batch_size)

    loss_fct = CrossEntropyLoss()

    IERE_best_score = 0.
    IDRR_best_score = 0.
    IERE_early_stop = 0
    IDRR_early_stop = 0
    for epoch_index in trange(int(args.num_train_epochs), desc="Epoch"):
        if IDRR_early_stop < 3:
            # IDRR
            for step_index, batch in enumerate(tqdm(IDRR_train_data_loader, desc="Step")):
                model.train()

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None, task='IDRR')
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = loss_fct(logits.view(-1, IDRR_num_labels), label_ids.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.zero_grad()

            valid_result = eval_model(model, IDRR_eval_data_loader, IDRR_label_list, args.eval_batch_size, args.IDRR_output_dir, device,
                                    prefix="IDRR epoch_%d_dev_" % epoch_index)
            logger.info(log_format(valid_result, prefix="IDRR Valid at Epoch %s" % epoch_index))
            if valid_result['score'] > IDRR_best_score:
                save_model(model, tokenizer=tokenizer, output_dir=args.IDRR_output_dir)
                IDRR_best_score = valid_result['score']
                IDRR_early_stop = 0
            else:
                IDRR_early_stop += 1
            logger.info("IDRR_early_stop is %d" % IDRR_early_stop)

        if IERE_early_stop < 3:
            # IERE
            for step_index, batch in enumerate(tqdm(IERE_train_data_loader, desc="Step")):
                model.train()

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None, task='IERE')
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = loss_fct(logits.view(-1, IERE_num_labels), label_ids.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.zero_grad()
                
            valid_result = eval_model(model, IERE_eval_data_loader, IERE_label_list, args.eval_batch_size, args.IERE_output_dir, device,
                                    prefix="IERE epoch_%d_dev_" % epoch_index)
            logger.info(log_format(valid_result, prefix="IERE Valid at Epoch %s" % epoch_index))
            if valid_result['score'] > IERE_best_score:
                save_model(model, tokenizer=tokenizer, output_dir=args.IERE_output_dir)
                IERE_best_score = valid_result['score']
                IERE_early_stop = 0
            else:
                IERE_early_stop += 1
            logger.info("IERE_early_stop is %d" % IERE_early_stop)

        if IERE_early_stop >= 3 and IDRR_early_stop >=3:
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--processor",
                        default='Both',
                        type=str,
                        help="Processor for Data folder, Both")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir.")
    parser.add_argument("--bert_model", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--IERE_output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")    
    parser.add_argument("--IDRR_output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")    
    parser.add_argument("--train_mode",
                        default='Multi',
                        type=str,
                        help="The mode of training. e.g. Multi")

    # Other parameters
    parser.add_argument("--device",
                        default=None,
                        type=str,
                        help="Device Str")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. "
                             "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs",
                        default=100.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    
    if "uncased" in args.bert_model:
        assert args.do_lower_case is True
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print("Set CUDA_VISIBLE_DEVICES as %s" % args.device)

    env_option = prepare_env(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")
    
    if os.path.exists(args.IERE_output_dir) and os.listdir(args.IERE_output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.IERE_output_dir))
    if not os.path.exists(args.IERE_output_dir):
        os.makedirs(args.IERE_output_dir)
    
    if os.path.exists(args.IDRR_output_dir) and os.listdir(args.IDRR_output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.IDRR_output_dir))
    if not os.path.exists(args.IDRR_output_dir):
        os.makedirs(args.IDRR_output_dir)
    
    processor = get_processor(args.processor)
    (IERE_label_list, IDRR_label_list) = processor.get_labels()
    IERE_num_labels = len(IERE_label_list)
    IDRR_num_labels = len(IDRR_label_list) 

    logger.info("Load Bert ...")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    logger.info(args.bert_model)
    if args.train_mode == 'Multi':
        logger.info("Multi Training ...")
        model = MultiTaskBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.bert_model,
                                                                       task1_num_labels=IERE_num_labels, task2_num_labels=IDRR_num_labels)
    else:
        raise NotImplementedError('%s not implement' % args.train_mode)
    model = prepare_model(model, env_option)
   
    if args.do_train:
        (IERE_valid_examples, IDRR_valid_examples) = processor.get_dev_examples(args.data_dir)
        IERE_valid_features_to_bert = convert_examples_to_features(examples=IERE_valid_examples,
                                                                   label_list=IERE_label_list,
                                                                   max_seq_length=args.max_seq_length,
                                                                   tokenizer=tokenizer,
                                                                   output_mode="classification")
        IDRR_valid_features_to_bert = convert_examples_to_features(examples=IDRR_valid_examples,
                                                                   label_list=IDRR_label_list,
                                                                   max_seq_length=args.max_seq_length,
                                                                   tokenizer=tokenizer,
                                                                   output_mode="classification")
        
        IERE_valid_data_loader = get_data_loader(IERE_valid_features_to_bert, args.train_batch_size, args.eval_batch_size, is_train=False)
        IDRR_valid_data_loader = get_data_loader(IDRR_valid_features_to_bert, args.train_batch_size, args.eval_batch_size, is_train=False)
        
        (IERE_train_examples, IDRR_train_examples) = processor.get_train_examples(args.data_dir)


        IERE_train_features_to_bert = convert_examples_to_features(examples=IERE_train_examples,
                                                                   label_list=IERE_label_list,
                                                                   max_seq_length=args.max_seq_length,
                                                                   tokenizer=tokenizer,
                                                                   output_mode="classification")
        IDRR_train_features_to_bert = convert_examples_to_features(examples=IDRR_train_examples,
                                                                   label_list=IDRR_label_list,
                                                                   max_seq_length=args.max_seq_length,
                                                                   tokenizer=tokenizer,
                                                                   output_mode="classification")

        IERE_train_data_loader = get_data_loader(IERE_train_features_to_bert, args.train_batch_size, args.eval_batch_size, is_train=True)
        IDRR_train_data_loader = get_data_loader(IDRR_train_features_to_bert, args.train_batch_size, args.eval_batch_size, is_train=True)
        
        model, optimizer = prepare_optimizer(model, args.learning_rate, args.adam_epsilon)
        
        train_model(model=model,
                    tokenizer=tokenizer,
                    IERE_train_data_loader=IERE_train_data_loader,
                    IDRR_train_data_loader=IDRR_train_data_loader,
                    IERE_eval_data_loader=IERE_valid_data_loader,
                    IDRR_eval_data_loader=IDRR_valid_data_loader,
                    IERE_label_list=IERE_label_list,
                    IDRR_label_list=IDRR_label_list,
                    args=args,
                    optimizer=optimizer,
                    env_option=env_option)

    if args.do_eval or args.do_train:
        (IERE_test_examples, IDRR_test_examples) = processor.get_test_examples(args.data_dir)
        IERE_test_features_to_bert = convert_examples_to_features(examples=IERE_test_examples,
                                                                  label_list=IERE_label_list,
                                                                  max_seq_length=args.max_seq_length,
                                                                  tokenizer=tokenizer,
                                                                  output_mode="classification")
        IDRR_test_features_to_bert = convert_examples_to_features(examples=IDRR_test_examples,
                                                                  label_list=IDRR_label_list,
                                                                  max_seq_length=args.max_seq_length,
                                                                  tokenizer=tokenizer,
                                                                  output_mode="classification")
        
        IERE_test_data_loader = get_data_loader(IERE_test_features_to_bert, args.train_batch_size, args.eval_batch_size, is_train=False)
        IDRR_test_data_loader = get_data_loader(IDRR_test_features_to_bert, args.train_batch_size, args.eval_batch_size, is_train=False)

        # IERE
        logger.info("Load model from %s" % args.IERE_output_dir)
        if args.train_mode == 'Multi':
            logger.info("Multi Training ...")
            model = MultiTaskBertForSequenceClassification.from_pretrained(args.IERE_output_dir,
                                                                           task1_num_labels=IERE_num_labels, task2_num_labels=IDRR_num_labels)
        else:
            raise NotImplementedError('%s not implement' % args.train_mode)
        tokenizer = BertTokenizer.from_pretrained(args.IERE_output_dir, do_lower_case=args.do_lower_case)
        model.to(env_option['device'])
        test_result = eval_model(model=model,
                                 eval_data_loader=IERE_test_data_loader,
                                 label_list=IERE_label_list,
                                 eval_batch_size=args.eval_batch_size, 
                                 output_dir=args.IERE_output_dir,
                                 device=env_option['device'],
                                 prefix="IERE final_test",
                                 )
        print(test_result)

        # IDRR
        logger.info("Load model from %s" % args.IDRR_output_dir)
        if args.train_mode == 'Multi':
            logger.info("Multi Training ...")
            model = MultiTaskBertForSequenceClassification.from_pretrained(args.IDRR_output_dir,
                                                                           task1_num_labels=IERE_num_labels, task2_num_labels=IDRR_num_labels)
        else:
            raise NotImplementedError('%s not implement' % args.train_mode)
        tokenizer = BertTokenizer.from_pretrained(args.IDRR_output_dir, do_lower_case=args.do_lower_case)
        model.to(env_option['device'])
        test_result = eval_model(model=model,
                                 eval_data_loader=IDRR_test_data_loader,
                                 label_list=IDRR_label_list,
                                 eval_batch_size=args.eval_batch_size, 
                                 output_dir=args.IDRR_output_dir,
                                 device=env_option['device'],
                                 prefix="IDRR final_test",
                                 )
        print(test_result)