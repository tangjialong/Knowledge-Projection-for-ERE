#!/usr/bin/env python
# -*- coding:utf-8 -*- 
import os
import csv
import sys
import logging
import numpy as np

import torch
from torch.utils.data import TensorDataset, RandomSampler, DistributedSampler, DataLoader, SequentialSampler

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_file(self, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                line = line.strip().split('\t')
                if len(line) >= 3:
                    lines.append(line)
        return lines
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the train, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class IEREProcessor(DataProcessor):
    def get_labels(self):
        """See base class."""
        return ['Precedence', 'Succession', 'Synchronous', 
                'Reason', 'Result', 'Condition', 
                'Contrast', 'Concession',
                'Conjunction', 'Instantiation', 'Restatement', 'Alternative', 'ChosenAlternative', 'Exception']

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, 'train')))
        return self._create_examples(self._read_file(os.path.join(data_dir, 'train-all')), 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_file(os.path.join(data_dir, 'dev')), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_file(os.path.join(data_dir, "test")), "test")

class IDRRProcessor(DataProcessor):
    def get_labels(self):
        """See base class."""
        return ['Temporal.Asynchronous', 'Temporal.Synchrony', 
                'Contingency.Cause', 'Contingency.Pragmatic cause',
                'Comparison.Contrast', 'Comparison.Concession',
                'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement', 'Expansion.Alternative', 'Expansion.List']

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, 'train')))
        return self._create_examples(self._read_file(os.path.join(data_dir, 'train')), 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_file(os.path.join(data_dir, 'dev')), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_file(os.path.join(data_dir, "test")), "test")

class BothProcessor(DataProcessor):
    def get_labels(self):
        """See base class."""
        return (['Precedence', 'Succession', 'Synchronous', 
                'Reason', 'Result', 'Condition', 
                'Contrast', 'Concession',
                'Conjunction', 'Instantiation', 'Restatement', 'Alternative', 'ChosenAlternative', 'Exception'],
                ['Temporal.Asynchronous', 'Temporal.Synchrony', 
                'Contingency.Cause', 'Contingency.Pragmatic cause',
                'Comparison.Contrast', 'Comparison.Concession',
                'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement', 'Expansion.Alternative', 'Expansion.List'])

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, 'train-IERE')))
        train_set_IERE = self._create_examples(self._read_file(os.path.join(data_dir, 'train-IERE')), 'train-IERE')
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, 'train-IDRR')))
        train_set_IDRR = self._create_examples(self._read_file(os.path.join(data_dir, 'train-IDRR')), 'train-IDRR')
        return (train_set_IERE, train_set_IDRR)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return (self._create_examples(self._read_file(os.path.join(data_dir, 'dev-IERE')), "dev-IERE"), 
                self._create_examples(self._read_file(os.path.join(data_dir, 'dev-IDRR')), "dev-IDRR"))
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return (self._create_examples(self._read_file(os.path.join(data_dir, "test-IERE")), "test-IERE"),
                self._create_examples(self._read_file(os.path.join(data_dir, "test-IDRR")), "test-IDRR"))

def get_data_loader(features_to_bert, train_batch_size, eval_batch_size, is_train=False):
    all_input_ids = torch.tensor([f.input_ids for f in features_to_bert], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features_to_bert], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features_to_bert], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features_to_bert], dtype=torch.long)

    if is_train:
        train_features = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_features)
        return DataLoader(train_features, sampler=train_sampler, batch_size=train_batch_size)
    else:
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        return DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

def get_processor(processor_name='IERE'):
    if processor_name == 'IERE':
        return IEREProcessor()
    elif processor_name == 'IDRR':
        return IDRRProcessor()
    elif processor_name == 'Both':
        return BothProcessor()
    else:
        raise NotImplementedError('%s not implement' % processor_name)