# -*- coding:utf-8 -*-
import os
import logging
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import classification_report

import torch
from torch.nn import CrossEntropyLoss
from pytorch_transformers import BertTokenizer, BertForSequenceClassification

from base.bert_utils import add_parser_arguments
from base.bert_utils import prepare_optimizer, prepare_model, prepare_env
from base.bert_utils import save_model, load_model
from base.bert_utils import p_r_f1, log_format
from base.bert_utils import FourBertForSequenceClassification
from base.data_loader import get_data_loader, get_processor
from base.io_utils import convert_examples_to_features

logger = logging.getLogger(__name__)

def compute_metrics(preds, labels, average='macro'):
    assert len(preds) == len(labels)
    return p_r_f1(preds, labels, average)

def eval_model(model, eval_data_loader, label_list, eval_batch_size, output_dir, device, only_four=False, up_bound=False, prefix=""):

    num_labels = len(label_list)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_data_loader.dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    model.eval()
    preds = []
    labels = []

    if not up_bound:
        auxs = []
        pred_auxs = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_data_loader, desc="Evaluating"):
        labels += [label_ids]

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        if num_labels == 14: # IERE
            four_ids = torch.where(label_ids>=3, torch.ones_like(label_ids), torch.zeros_like(label_ids)) + \
                                    torch.where(label_ids>=6, torch.ones_like(label_ids), torch.zeros_like(label_ids)) + \
                                    torch.where(label_ids>=8, torch.ones_like(label_ids), torch.zeros_like(label_ids))
        elif num_labels == 11: # IDRR
            four_ids = torch.where(label_ids>=2, torch.ones_like(label_ids), torch.zeros_like(label_ids)) + \
                                    torch.where(label_ids>=4, torch.ones_like(label_ids), torch.zeros_like(label_ids)) + \
                                    torch.where(label_ids>=6, torch.ones_like(label_ids), torch.zeros_like(label_ids))
        
        with torch.no_grad():
            if up_bound:
                logits, aux_logits = model(input_ids, segment_ids, input_mask, labels=four_ids)
            else:
                logits, aux_logits = model(input_ids, segment_ids, input_mask, labels=None)
            if isinstance(logits, tuple):
                logits = logits[0]

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
        
        if not up_bound:
            auxs += [four_ids.cpu()]

            if len(pred_auxs) == 0:
                pred_auxs.append(aux_logits.detach().cpu().numpy())
            else:
                pred_auxs[0] = np.append(pred_auxs[0], aux_logits.detach().cpu().numpy(), axis=0)
                
    if not only_four:
        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        labels = torch.cat(labels)
        
        logger.info('\n' + classification_report(labels.numpy(), preds, target_names=label_list))
        result = compute_metrics(preds, labels.numpy(), 'macro')
        logger.info(result)

    if not up_bound:
        pred_auxs = pred_auxs[0]
        pred_auxs = np.argmax(pred_auxs, axis=1)
        auxs = torch.cat(auxs)

        logger.info('\n' + classification_report(auxs.numpy(), pred_auxs, target_names=['Temporal', 'Contingency', 'Comparison', 'Expansion']))
        auxresult = compute_metrics(pred_auxs, auxs.numpy(), 'macro')
        logger.info(auxresult)
        
        if only_four:
            return auxresult

    return result

def train_model(model,
                tokenizer,
                train_data_loader,
                eval_data_loader,
                label_list,
                args,
                optimizer,
                env_option,
                only_four=False,
                up_bound=False,
                lamda = 0.5,
                ):
    device = env_option['device']

    num_labels = len(label_list)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_loader.dataset))
    logger.info("  Batch size = %d", args.train_batch_size)

    loss_fct = CrossEntropyLoss()
    aux_loss_fct = CrossEntropyLoss()
    
    best_score = 0.
    early_stop = 0
    for epoch_index in trange(int(args.num_train_epochs), desc="Epoch"):
        for step_index, batch in enumerate(tqdm(train_data_loader, desc="Step")):
            model.train()

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            if num_labels == 14: # IERE
                four_ids = torch.where(label_ids>=3, torch.ones_like(label_ids), torch.zeros_like(label_ids)) + \
                                       torch.where(label_ids>=6, torch.ones_like(label_ids), torch.zeros_like(label_ids)) + \
                                       torch.where(label_ids>=8, torch.ones_like(label_ids), torch.zeros_like(label_ids))
            elif num_labels == 11: # IDRR
                four_ids = torch.where(label_ids>=2, torch.ones_like(label_ids), torch.zeros_like(label_ids)) + \
                                       torch.where(label_ids>=4, torch.ones_like(label_ids), torch.zeros_like(label_ids)) + \
                                       torch.where(label_ids>=6, torch.ones_like(label_ids), torch.zeros_like(label_ids))
            # define a new function to compute loss values for both output_modes
            if up_bound:
                logits, aux_logits = model(input_ids, segment_ids, input_mask, labels=four_ids)
            else:
                logits, aux_logits = model(input_ids, segment_ids, input_mask, labels=None)
            
            if isinstance(logits, tuple):
                logits = logits[0]
            
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            if not up_bound:
                aux_loss = aux_loss_fct(aux_logits.view(-1, 4), four_ids.view(-1))
                if only_four:
                    loss = aux_loss
                else:
                    loss = lamda * loss + (1-lamda) * aux_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()

        valid_result = eval_model(model, eval_data_loader, label_list, args.eval_batch_size, args.output_dir, device, only_four, up_bound, 
                                  prefix="epoch_%d_dev" % epoch_index)
        logger.info(log_format(valid_result, prefix="Valid at Epoch %s" % epoch_index))
        if valid_result['score'] > best_score:
            save_model(model, tokenizer=tokenizer, output_dir=args.output_dir)
            best_score = valid_result['score']
            early_stop = 0
        else:
            early_stop += 1
        logger.info("early_stop is %d" % early_stop)
        if early_stop >= 3:
            return

if __name__ == "__main__":
    args = add_parser_arguments()

    if args.only_four:
        assert not args.up_bound 

    env_option = prepare_env(args.seed)
    
    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    processor = get_processor(args.processor)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    logger.info("Load Bert ...")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    logger.info(args.bert_model)
    if args.train_mode == 'Four':
        logger.info("Four Training ...")
        model = FourBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.bert_model,
                                                                  num_labels=num_labels, coarse_num_labels=4)
    else:
        raise NotImplementedError('%s not implement' % args.train_mode)

    model = prepare_model(model, env_option)

    if args.do_train:
        valid_features_to_bert = convert_examples_to_features(examples=processor.get_dev_examples(args.data_dir),
                                                              label_list=label_list,
                                                              max_seq_length=args.max_seq_length,
                                                              tokenizer=tokenizer,
                                                              output_mode="classification")
        valid_data_loader = get_data_loader(valid_features_to_bert, args.train_batch_size, args.eval_batch_size, is_train=False)

        
        train_features_to_bert = convert_examples_to_features(examples=processor.get_train_examples(args.data_dir),
                                                              label_list=label_list,
                                                              max_seq_length=args.max_seq_length,
                                                              tokenizer=tokenizer,
                                                              output_mode="classification")
        train_data_loader = get_data_loader(train_features_to_bert, args.train_batch_size, args.eval_batch_size, is_train=True)
        
        model, optimizer = prepare_optimizer(model, args.learning_rate, args.adam_epsilon)

        train_model(model=model,
                    tokenizer=tokenizer,
                    train_data_loader=train_data_loader,
                    eval_data_loader=valid_data_loader,
                    label_list=label_list,
                    args=args,
                    optimizer=optimizer,
                    env_option=env_option,
                    only_four=args.only_four,
                    up_bound=args.up_bound,
                    lamda=args.lamda)

    if args.do_eval or args.do_train:
        model, tokenizer = load_model(train_mode=args.train_mode, 
                                      output_dir=args.output_dir,
                                      num_labels=num_labels,
                                      do_lower_case=args.do_lower_case,
                                      device=env_option['device'])

        test_features_to_bert = convert_examples_to_features(examples=processor.get_test_examples(args.data_dir),
                                                             label_list=label_list,
                                                             max_seq_length=args.max_seq_length,
                                                             tokenizer=tokenizer,
                                                             output_mode="classification")
        test_data_loader = get_data_loader(test_features_to_bert, args.train_batch_size, args.eval_batch_size, is_train=False)

        test_result = eval_model(model=model,
                                 eval_data_loader=test_data_loader,
                                 label_list=label_list,
                                 eval_batch_size=args.eval_batch_size, 
                                 output_dir=args.output_dir,
                                 device=env_option['device'],
                                 only_four=args.only_four,
                                 up_bound=args.up_bound,
                                 prefix="final_test"
                                 )
        print(test_result)