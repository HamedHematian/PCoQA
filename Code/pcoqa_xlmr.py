# -*- coding: utf-8 -*-
import numpy as np
import torch
import json
import pickle
import shutil
import unicodedata
from tqdm import tqdm
from copy import deepcopy
import transformers
import sys
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertModel, BertForQuestionAnswering, AutoTokenizer
from transformers import AutoTokenizer, AutoModel
import os
from collections import defaultdict, namedtuple
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam, RMSprop
from copy import deepcopy
import random
import pickle as pk
import gdown
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch.backends.cudnn
import torch.cuda
SEED = 12345
def set_determenistic_mode(SEED, disable_cudnn=False):
  torch.manual_seed(SEED)                       # Seed the RNG for all devices (both CPU and CUDA).
  random.seed(SEED)                             # Set python seed for custom operators.
  rs = RandomState(MT19937(SeedSequence(SEED))) # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
  np.random.seed(SEED)
  torch.cuda.manual_seed_all(SEED)              # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

  if not disable_cudnn:
    torch.backends.cudnn.benchmark = False    # Causes cuDNN to deterministically select an algorithm,
                                              # possibly at the cost of reduced performance
                                              # (the algorithm itself may be nondeterministic).
    torch.backends.cudnn.deterministic = True # Causes cuDNN to use a deterministic convolution algorithm,
                                              # but may slow down performance.
                                              # It will not guarantee that your training process is deterministic
                                              # if you are using other libraries that may use nondeterministic algorithms
  else:
    torch.backends.cudnn.enabled = False # Controls whether cuDNN is enabled or not.
                                         # If you want to enable cuDNN, set it to True.
set_determenistic_mode(SEED)
def seed_worker(worker_id):
    worker_seed = SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(SEED)

gdown.download(id="1tzaAHUkIkGzhbZpCIKmOYYQKvvqoRfsO")
gdown.download(id="16wPRHP2AC5WI2m7Y_fEWEOUYl7ynMKrb")
gdown.download(id="1qYU3601tCOI-MTQut8Nb7mSZMDLXuASY")

with open('PCoQA_Train.pk', 'rb') as f:
    train_data = pk.load(f)
with open('PCoQA_Eval.pk', 'rb') as f:
    eval_data = pk.load(f)
with open('PCoQA_Test.pk', 'rb') as f:
    test_data = pk.load(f)


os.makedirs('examples')
os.makedirs('features')
os.makedirs('examples/train')
os.makedirs('examples/eval')
os.makedirs('examples/test')
os.makedirs('features/train')
os.makedirs('features/eval')
os.makedirs('features/test')

def read_file(filename):
  with open(filename, 'r') as f:
    return json.load(f)

def load_data(filename):
  with open(filename, 'rb') as f:
    x = pickle.load(f)
  return x

def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

# load tokenizer and model
model_path_or_name = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
model = AutoModel.from_pretrained(model_path_or_name)

"""# Official Evaluation Code"""

import json, string, re
from collections import Counter, defaultdict


def is_overlapping(x1, x2, y1, y2):
  return max(x1, y1) <= min(x2, y2)

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def compute_span_overlap(pred_span, gt_span, text):
  if gt_span == 'غیرقابل‌پاسخ':
    if pred_span == 'غیرقابل‌پاسخ':
      return 'Exact match', 1.0
    return 'No overlap', 0.
  fscore = f1_score(pred_span, gt_span)
  pred_start = text.find(pred_span)
  gt_start = text.find(gt_span)

  if pred_start == -1 or gt_start == -1:
    return 'Span indexing error', fscore

  pred_end = pred_start + len(pred_span)
  gt_end = gt_start + len(gt_span)

  fscore = f1_score(pred_span, gt_span)
  overlap = is_overlapping(pred_start, pred_end, gt_start, gt_end)

  if exact_match_score(pred_span, gt_span):
    return 'Exact match', fscore
  if overlap:
    return 'Partial overlap', fscore
  else:
    return 'No overlap', fscore

def exact_match_score(prediction, ground_truth):
  return (normalize_answer(prediction) == normalize_answer(ground_truth))

def display_counter(title, c, c2=None):
  print(title)
  for key, _ in c.most_common():
    if c2:
      print('%s: %d / %d, %.1f%%, F1: %.1f' % (
        key, c[key], sum(c.values()), c[key] * 100. / sum(c.values()), sum(c2[key]) * 100. / len(c2[key])))
    else:
      print('%s: %d / %d, %.1f%%' % (key, c[key], sum(c.values()), c[key] * 100. / sum(c.values())))

def leave_one_out_max(prediction, ground_truths, article):
  if len(ground_truths) == 1:
    return metric_max_over_ground_truths(prediction, ground_truths, article)[1]
  else:
    t_f1 = []
    # leave out one ref every time
    for i in range(len(ground_truths)):
      idxes = list(range(len(ground_truths)))
      idxes.pop(i)
      refs = [ground_truths[z] for z in idxes]
      t_f1.append(metric_max_over_ground_truths(prediction, refs, article)[1])
  return 1.0 * sum(t_f1) / len(t_f1)


def metric_max_over_ground_truths(prediction, ground_truths, article):
  scores_for_ground_truths = []
  for ground_truth in ground_truths:
    score = compute_span_overlap(prediction, ground_truth, article)
    scores_for_ground_truths.append(score)
  return max(scores_for_ground_truths, key=lambda x: x[1])


def handle_cannot(refs):
  num_cannot = 0
  num_spans = 0
  for ref in refs:
    if ref == 'غیرقابل‌پاسخ':
      num_cannot += 1
    else:
      num_spans += 1
  if num_cannot >= num_spans:
    refs = ['CANNOTANSWER']
  else:
    refs = [x for x in refs if x != 'غیرقابل‌پاسخ']
  return refs


def leave_one_out(refs):
  if len(refs) == 1:
    return 1.
  splits = []
  for r in refs:
    splits.append(r.split())
  t_f1 = 0.0
  for i in range(len(refs)):
    m_f1 = 0
    for j in range(len(refs)):
      if i == j:
        continue
      f1_ij = f1_score(refs[i], refs[j])
      if f1_ij > m_f1:
        m_f1 = f1_ij
    t_f1 += m_f1
  return t_f1 / len(refs)






def eval_fn(val_results, model_results, verbose):
  span_overlap_stats = Counter()
  sentence_overlap = 0.
  para_overlap = 0.
  total_qs = 0.
  f1_stats = defaultdict(list)
  unfiltered_f1s = []
  total_dials = 0.
  unanswerables = []
  for p in val_results:
    for par in p['paragraphs']:
      did = par['id']
      qa_list = par['qas']
      good_dial = 1.
      for qa in qa_list:
        q_idx = qa['id']
        val_spans = [anss['text'] for anss in qa['answers']]
        val_spans = handle_cannot(val_spans)
        hf1 = leave_one_out(val_spans)

        if did not in model_results or q_idx not in model_results[did]:
          # print(did, q_idx, 'no prediction for this dialogue id')
          good_dial = 0
          f1_stats['NO ANSWER'].append(0.0)
          if val_spans == ['غیرقابل‌پاسخ']:
            unanswerables.append(0.0)
          total_qs += 1
          unfiltered_f1s.append(0.0)
          if hf1 >= .4:
            human_f1.append(hf1)
          continue

        pred_span, pred_yesno, pred_followup = model_results[did][q_idx]

        max_overlap, _ = metric_max_over_ground_truths( \
          pred_span, val_spans, par['context'])
        max_f1 = leave_one_out_max( \
          pred_span, val_spans, par['context'])
        unfiltered_f1s.append(max_f1)

        # dont eval on low agreement instances
        if hf1 < .4:
          continue

        human_f1.append(hf1)

        if val_spans == ['غیرقابل‌پاسخ']:
          unanswerables.append(max_f1)
        if verbose:
          print("-" * 20)
          print(pred_span)
          print(val_spans)
          print(max_f1)
          print("-" * 20)
        if max_f1 >= hf1:
          HEQ += 1.
        else:
          good_dial = 0.
        span_overlap_stats[max_overlap] += 1
        f1_stats[max_overlap].append(max_f1)
        total_qs += 1.
      DHEQ += good_dial
      total_dials += 1


  DHEQ_score = 100.0 * DHEQ / total_dials
  HEQ_score = 100.0 * HEQ / total_qs
  all_f1s = sum(f1_stats.values(), [])
  overall_f1 = 100.0 * sum(all_f1s) / len(all_f1s)
  unfiltered_f1 = 100.0 * sum(unfiltered_f1s) / len(unfiltered_f1s)
  unanswerable_score = (100.0 * sum(unanswerables) / len(unanswerables))
  metric_json = {"unfiltered_f1": unfiltered_f1, "f1": overall_f1, "HEQ": HEQ_score, "DHEQ": DHEQ_score, "yes/no": yesno_score, "followup": followup_score, "unanswerable_acc": unanswerable_score}
  if verbose:
    print("=======================")
    display_counter('Overlap Stats', span_overlap_stats, f1_stats)
  print("=======================")
  print('Overall F1: %.1f' % overall_f1)
  with open('val_report.txt', 'a') as f:
    f.write('Overall F1: %.1f' % overall_f1)

  print('Unfiltered F1 ({0:d} questions): {1:.1f}'.format(len(unfiltered_f1s), unfiltered_f1))
  print('Accuracy On Unanswerable Questions: {0:.1f} %% ({1:d} questions)'.format(unanswerable_score, len(unanswerables)))
  print('Human F1: %.1f' % (100.0 * sum(human_f1) / len(human_f1)))
  print('Model F1 >= Human F1 (Questions): %d / %d, %.1f%%' % (HEQ, total_qs, 100.0 * HEQ / total_qs))
  print('Model F1 >= Human F1 (Dialogs): %d / %d, %.1f%%' % (DHEQ, total_dials, 100.0 * DHEQ / total_dials))
  print("=======================")
  output_string = 'Overall F1: %.1f\n' % overall_f1
  output_string += 'Yes/No Accuracy : %.1f\n' % yesno_score
  output_string += 'Followup Accuracy : %.1f\n' % followup_score
  output_string += 'Unfiltered F1 ({0:d} questions): {1:.1f}\n'.format(len(unfiltered_f1s), unfiltered_f1)
  output_string += 'Accuracy On Unanswerable Questions: {0:.1f} %% ({1:d} questions)\n'.format(unanswerable_score, len(unanswerables))
  output_string += 'Human F1: %.1f\n' % (100.0 * sum(human_f1) / len(human_f1))
  output_string += 'Model F1 >= Human F1 (Questions): %d / %d, %.1f%%\n' % (HEQ, total_qs, 100.0 * HEQ / total_qs)
  output_string += 'Model F1 >= Human F1 (Dialogs): %d / %d, %.1f%%' % (DHEQ, total_dials, 100.0 * DHEQ / total_dials)

  # save_prediction(epoch, train_step, output_string)

  return metric_json

def run_eval(mode, predicted_obj):
  new_eval_data = dict()
  mean_f1s_ = [[] for _ in range(30)]
  if mode == 'eval':
    main_data = eval_data
  elif mode == 'test':
    main_data = test_data

  for data in main_data:
    new_eval_data[str(data['id'])] = data


  res = {
      'question': [],
      'pred': [],
      'orig': [],
      'f1': [],
      'heq-q': []
  }

  f1s = []
  dialog_f1s = []
  dialog_hfs = []
  unans_score = []
  heq_q = []
  heq_d = []
  heq_m = []
  EM = []

  dialogs_f1s = dict()
  dialogs_hfs = dict()

  results = []
  for q_idx, model_answer in predicted_obj.answers.items():
    d_id, q_num = q_idx.split('#')[0], int(q_idx.split('#')[1])
    d_id, q_num = q_idx.split('#')[0], int(q_idx.split('#')[1])

    if d_id not in dialogs_f1s.keys():
      dialogs_f1s[d_id] = []
      dialogs_hfs[d_id] = []

    qa = new_eval_data[d_id]['qas'][q_num]
    answers_num = len(qa['answers'])
    orig_answers = [qa['answers'][qidx]['text'] for qidx in range(answers_num)]
    if 'غیرقابل‌پاسخ' in orig_answers:
      if model_answer.startswith('غیرقابل'):
        unans_score.append(1.0)
      else:
        unans_score.append(0.0)

    res['question'].append(qa['question'])
    res['pred'].append(model_answer)
    res['orig'].append(orig_answers)

    context = new_eval_data[d_id]['article']
    hf = qa['hf']
    f1s_ = [compute_span_overlap(model_answer, orig_answer, context)[1] for orig_answer in orig_answers]
    max_f1 = max(f1s_)
    dialog_hfs.append(hf)
    dialog_f1s.append(max_f1)
    f1s.append(max_f1)
    mean_f1s_[q_num].append(max_f1)
    
    if int(max_f1) == 1:
      EM.append(1.)
    else:
      EM.append(0.)
    
    if max_f1 >= hf:
      heq_q.append(1)
    else:
      heq_q.append(0)
    
    res['f1'].append(max_f1)
    res['heq-q'].append(heq_q)

    dialogs_f1s[d_id].append(max_f1)
    dialogs_hfs[d_id].append(hf)


  for key in dialogs_f1s.keys():
    dialog_f1s = dialogs_f1s[key]
    dialog_hfs = dialogs_hfs[key]

    heq_d.append(all(x >= y for x, y in zip(dialog_f1s, dialog_hfs)))
    heq_m.append(sum(dialog_f1s) >= sum(dialog_hfs))
    


  f1_score_ = sum(f1s) / len(f1s)
  heq_q_score_ = sum(heq_q) / len(heq_q)
  heq_m_score_ = sum(heq_m) / len(heq_m)
  heq_d_score_ = sum(heq_d) / len(heq_d)
  unans_score_ = sum(unans_score) / len(unans_score)
  EM_score_ = sum(EM) / len(EM)


  mean_f1s = [sum(mean_f1) / (1e-8 + len(mean_f1)) for mean_f1 in mean_f1s_]
  print(f'f1 score is {f1_score_}')
  print(f'HEQ-Q score is {heq_q_score_}')
  print(f'HEQ-M score is {heq_m_score_}')
  print(f'HEQ-D score is {heq_d_score_}')
  print(f'EM score is {EM_score_}')
  print(f'Unanswerable score is {unans_score_}')
  print(mean_f1s)
  return f1_score_, heq_q_score_, heq_d_score_, mean_f1s, res

"""# Preprocess Data"""

class CQA_DATA:

  def __init__(self,
               question,
               context,
               history,
               answer,
               qid,
               q_num,
               answer_start,
               answer_end,
               is_answerable):

    self.question = question
    self.context = context
    self.answer = answer
    self.history = history
    self.qid = qid
    self.q_num = q_num
    self.answer_start = answer_start
    self.answer_end = answer_end
    self.is_answerable = is_answerable
    self.cleaned_context = None
    self.cleaned_answer = {
        'text': self.answer,
        'start': self.answer_start,
        'end': self.answer_end
    }
    self.cleaned_context = context

  def __repr__(self):
    repr = ''
    repr += 'context -> ' + self.context[:100] + '\n'
    repr += 'question ->' + self.question + '\n'
    repr += 'question id ->' + str(self.qid) + '\n'
    repr += 'turn_number ->' + str(self.turn_number) + '\n'
    repr += 'answer ->' + self.answers[0]['text'] + '\n'
    return repr

class Feature:

  def __init__(self,
               qid,
               question_part,
               input_ids,
               attention_mask,
               offset_mappings,
               max_context_dict,
               start,
               end,
               is_answerable,
               context,
               cleaned_context,
               context_start,
               context_end,
               example_start_char,
               example_end_char,
               example_answer):

    self.qid = qid
    self.question_part = question_part
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.offset_mappings = offset_mappings
    self.max_context_dict = max_context_dict
    self.start = start
    self.end = end
    self.is_answerable = is_answerable
    self.context = context
    self.cleaned_context = cleaned_context
    self.context_start = context_start
    self.context_end = context_end
    self.example_start_char = example_start_char
    self.example_end_char = example_end_char
    self.example_answer = example_answer

  def __repr__(self):
    repr = ''
    repr += 'qid --> ' + str(self.qid) + '\n'
    repr += 'quesion part --> ' + str(self.question_part) + '\n'
    repr += 'answer part --> ' + str(self.start) + ' ' + str(self.end) + '\n'
    return repr

"""# Examples"""

def make_examples(data, data_type, num_sample, clean_samples=True):
  examples = []
  each_file_size = 1000
  example_file_index = 0
  data_dir = f'examples/{data_type}/'

  for dialog_num, dialog in enumerate(tqdm(data, leave=False, position=0)):
    dialog_history = []
    dialog_container = []
    dialog_id = dialog['id']
    title = dialog['title']
    context = dialog['article']
    dialog_len = len(dialog['qas'])

    for q_num, qa in enumerate(dialog['qas']):
      history = []
      question = qa['rewritten_question']
      answer = qa['answers'][0]
      #if num_sample == 5555:
        #print(answer['text'])
        #print('++++++++++++++++++++++++++++')

      is_answerable = False if answer['text'] == 'غیرقابل‌پاسخ' else True

      if not q_num == 0:
        history = deepcopy(dialog_history)

      qid = f'{dialog_id}#{q_num}'
      cqa_example = CQA_DATA(question=question,
                             context=context,
                             history=history,
                             answer=answer['text'],
                             qid=qid,
                             q_num=q_num,
                             answer_start=answer['start'],
                             answer_end=answer['end'],
                             is_answerable=is_answerable)

      examples.append(cqa_example)
      dialog_history.append([cqa_example.question, cqa_example.answer])

    if (dialog_num + 1) % each_file_size == 0:
      filename = f'{data_type}_examples_' + str(example_file_index) + '.bin'
      save_data(examples, os.path.join(data_dir, filename))
      example_file_index += 1
      examples = []

  if examples != []:
    filename = f'{data_type}_examples_' + str(example_file_index) + '.bin'
    save_data(examples, os.path.join(data_dir, filename))

"""# Features"""

def make_features(data_type):
  data_dir = f'examples/{data_type}/'
  example_files = os.listdir(data_dir)
  example_files = [os.path.join(data_dir, example_file) for example_file in example_files]
  features_list = []
  features_dir = f'features/{data_type}/'
  current_file_index = 0
  max_history_to_consider = int(sys.argv[2])

  for file_index, filename in enumerate(example_files):
    examples = load_data(filename)
    for example in tqdm(examples, leave=False, position=0):
      example_features = []
      concatenated_question = []

      # concat history
      for hist in example.history[-max_history_to_consider:]:
        concatenated_question.append(hist[0])

      # append current question to concatenated question
      concatenated_question.append(example.question)

      # make string out of concatenated question
      concatenated_question = ' '.join(concatenated_question)

      # tokenize current feature
      text_tokens = tokenizer(
          concatenated_question,
          example.cleaned_context,
          max_length=model.config.max_position_embeddings - 2,
          padding='max_length',
          truncation='only_second',
          return_overflowing_tokens=True,
          return_offsets_mapping=True,
          stride=128)

      # find start and end of context
      for idx in range(len(text_tokens['input_ids'])):
        found_start = False
        found_end = False
        context_start = 0
        context_end = 511
        max_context_dict = {}

        for token_idx, token in enumerate(text_tokens['offset_mapping'][idx][1:]):
          if token[0] == 0 and token[1] == 0:
            context_start = token_idx + 3
            break

        for token_idx, token in enumerate(text_tokens['offset_mapping'][idx][context_start:]):
          if token[0] == 0 and token[1] == 0:
            context_end = token_idx + context_start - 1
            break

        chunk_offset_mapping = text_tokens['offset_mapping'][idx]
        for context_idx, data in enumerate(chunk_offset_mapping[context_start: context_end + 1]):
          max_context_dict[f'({data[0]},{data[1]})'] = min(context_idx, context_end - context_idx) + (context_end - context_start + 1) * .01

        # find and mark current question answer
        marker_ids = np.zeros(shape=(model.config.max_position_embeddings,), dtype=np.int64)
        last_token = None
        for token_idx, token in enumerate(chunk_offset_mapping[context_start: context_end + 1]):
          if token[0] == example.cleaned_answer['start'] and not found_start:
            found_start = True
            start = token_idx + context_start

          elif last_token and last_token[0] < example.cleaned_answer['start'] and token[0] > example.cleaned_answer['start']:
            found_start = True
            start = (token_idx - 1) + context_start

          if token[1] == example.cleaned_answer['end'] and not found_end:
            found_end = True
            end = token_idx + context_start

          elif last_token and last_token[1] < example.cleaned_answer['end'] and token[1] > example.cleaned_answer['end'] and last_token:
            found_end = True
            end = token_idx + context_start
          last_token = token

        # add feature to features list
        if end < start and found_start and found_end:
          assert False, 'start and end do not match'

        # since there is no prediction we throw the example out (only when training)
        # if ((not found_start) or (not found_end)) and data_type == 'train':
        #   continue

        if ((not found_start) or (not found_end)) and data_type == 'train':
          continue
        elif ((not found_start) or (not found_end)) and data_type != 'train':
          start, end = 0, 0

        # plausibility check
        if found_start or found_end:
          answer = example.cleaned_answer['text'].strip()
          generated_answer = example.cleaned_context[chunk_offset_mapping[start][0]: chunk_offset_mapping[end][1]]
          if answer.find(generated_answer) == -1:
            pass
            #print(generated_answer)
            #print(answer)
            #print(' ' in answer)
            #print(example.answer)
            #print('------------------------------')


        # mark history answers
        example_features.append(Feature(example.qid,
                                          idx,
                                          text_tokens['input_ids'][idx],
                                          text_tokens['attention_mask'][idx],
                                          text_tokens['offset_mapping'][idx],
                                          max_context_dict,
                                          start,
                                          end,
                                          example.is_answerable,
                                          example.context,
                                          example.cleaned_context,
                                          context_start,
                                          context_end,
                                          example.answer_start,
                                          example.answer_end,
                                          example.answer))
      # create max context mask
      for feature_1 in example_features:
        max_context_mask = {}
        for key in list(feature_1.max_context_dict.keys()):
          max_context_mask[key] = True
          for feature_2 in example_features:
            if key in feature_2.max_context_dict:
              if feature_1.max_context_dict[key] < feature_2.max_context_dict[key]:
                max_context_mask[key] = False
        feature_1.max_context_mask = max_context_mask

        found_start = found_end = False
        start_mask = end_mask = 0
        # now compute span mask
        for key_idx, (key, value) in enumerate(feature_1.max_context_mask.items()):
          if key_idx == 0 and value:
            found_start = True
          elif value and not found_start:
            start_mask = key_idx
            found_start = True
          elif not value and found_start and not found_end:
            end_mask = key_idx
            found_end = True
          elif key_idx == len(feature_1.max_context_mask) - 1 and value and not found_end:
            end_mask = key_idx + 1
        feature_1.mask_span = [context_start + start_mask, context_start + end_mask]
      features_list.extend(example_features)

    filename = f'{data_type}_features_' + str(file_index) + '.bin'
    save_data(features_list, os.path.join(features_dir, filename))
    features_list = []

make_examples(train_data, 'train', 1000000)
make_examples(eval_data, 'eval', 5555)
make_examples(test_data, 'test', 5555)
make_features('train')
make_features('eval')
make_features('test')

"""# DataLoader"""

class DataManager:

  def __init__(self, current_file, current_index, data_dir, batch_size, shuffle=True):
    self.files = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_')[2].split('.')[0]))
    self.files = list(map(lambda x: os.path.join(data_dir, x), self.files))
    self.shuffle = shuffle
    self.data_len = 0
    for filename in self.files:
      self.data_len += len(load_data(filename))
    self.batch_size = batch_size
    self.reset_datamanager(current_file, current_index)


  def reset_datamanager(self, current_file_index, current_index):
    self.current_index = current_index
    self.current_file_index = current_file_index
    self.features = self.load_data_file(self.files[self.current_file_index])

  def load_data_file(self, filename):
    if self.shuffle:
      data = load_data(filename)
      random.shuffle(data)
      return data
    else:
      return load_data(filename)

  def next(self):
    temp = self.features[self.current_index:self.current_index + self.batch_size]
    self.temp = temp
    self.current_index += self.batch_size
    if self.current_index >= len(self.features):
      self.current_index = 0
      self.current_file_index += 1
      if self.current_file_index == len(self.files):
        self.reset_datamanager(current_file_index=0, current_index=0)
        return temp, True
      else:
        self.features = self.load_data_file(self.files[self.current_file_index])
    return temp, False

class DataLoader:

  def __init__(self, current_file, current_index, batch_size, shuffle=True, mode='train'):
    data_type = mode
    self.batch_size = batch_size
    self.data_manager = DataManager(current_file, current_index, f'features/{data_type}/', batch_size, shuffle)

  def __iter__(self):
    self.stop_iteration = False
    return self

  def __len__(self):
    return int(self.data_manager.data_len // self.batch_size)

  def reset_dataloader(self, current_file, current_index):
    self.data_manager.reset_datamanager(current_file, current_index)

  def features_2_tensor(self, features):
    x = dict()
    x['input_ids'] = torch.LongTensor([feature.input_ids for feature in features])
    x['attention_mask'] = torch.LongTensor([feature.attention_mask for feature in features])
    #x['token_type_ids'] = torch.LongTensor([feature.token_type_ids for feature in features])
    x['start_positions'] = torch.cat([torch.tensor([feature.start]) for feature in features]).view(-1)
    x['end_positions'] = torch.cat([torch.tensor([feature.end]) for feature in features]).view(-1)
    x['features'] = features
    return x

  def __next__(self):
    if self.stop_iteration:
      raise StopIteration
    features, self.stop_iteration = self.data_manager.next()
    return self.features_2_tensor(features)

"""# Utils"""

feature_output = namedtuple(
    'feature_output',
        ['start_logit', 'end_logit', 'feature'])

PrelimPrediction = namedtuple(
    "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "qid"]
    )

NbestPrediction = namedtuple(
    "NbestPrediction", ["text", "start_logit", "end_logit"]
)

Answer = namedtuple(
    'Answer', ['qid', 'answer']
)

def to_numpy(tensor):
  return tensor.detach().cpu().numpy()

class EvalProcessOutput:
  def __init__(self, n_best_size=5, answer_max_len=50, answerability_threshold=0.0):
    self.answers = defaultdict(list)
    self.examples_output = []
    self.n_best_size = n_best_size
    self.answer_max_len = answer_max_len
    self.answerability_threshold = answerability_threshold
    self.ps = []


  def process_feature_output(self, start_logits, end_logits, features):
    for start_logit, end_logit, feature in zip(start_logits, end_logits, features):
      self.examples_output.append(
          feature_output(start_logit, end_logit, feature)
      )

  def stack_features(self):
    examples_list = defaultdict(list)
    for feature_out in self.examples_output:
      examples_list[feature_out.feature.qid].append(feature_out)
    return examples_list


  def process_output(self):
    self.extract_answers()
    self.get_predictions()

  def get_predictions(self):
    dialogs = defaultdict(list)
    self.dialogs_answers = defaultdict(list)
    for example_qid, answer in self.answers.items():
      dialog_id = example_qid.split('#')[0]
      dialogs[dialog_id].append(Answer(example_qid, answer))

    self.digs = dialogs
    for dialog_id, dialog in dialogs.items():
      dialog = sorted(dialog, key=lambda x: int(x.qid.split('#')[1]))
      max_dialog_len = int(dialog[-1].qid.split('#')[1]) + 1
      self.dialogs_answers[dialog_id] = ['' for i in range(max_dialog_len)]
      for example in dialog:
        example_turn = int(example.qid.split('#')[1])
        self.dialogs_answers[dialog_id][example_turn] = example.answer


  def extract_answers(self):
    examples_list = self.stack_features()
    for example_qid, example in examples_list.items():
      null_score = np.inf
      prelim_predictions = []
      self.example = example
      for feature_index, feature_output in enumerate(example):
        feature_null_score = feature_output.start_logit[0] + feature_output.end_logit[0]

        if feature_null_score < null_score:
          null_score = feature_null_score
          null_feature_index = feature_index
          null_start_logit = feature_output.start_logit[0]
          null_end_logit = feature_output.end_logit[0]

        start_indexes = self.get_best_indexes(feature_output.start_logit)
        end_indexes = self.get_best_indexes(feature_output.end_logit)

        for start_index in start_indexes:
          for end_index in end_indexes:
            if start_index > feature_output.feature.context_end:
              continue
            # if end_index > feature_output.feature.context_end:
            #   continue
            # if start_index < feature_output.feature.context_start:
            #   continue
            if end_index < feature_output.feature.context_start:
              continue
            if start_index < feature_output.feature.mask_span[0]:
              continue
            if end_index - start_index + 1 > self.answer_max_len:
              continue
            if end_index <= start_index:
              continue

            prelim_predictions.append(
                PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=feature_output.start_logit[start_index],
                  end_logit=feature_output.end_logit[end_index],
                  qid=example_qid
            )
                )
      # append a null one for handling CANNOTANSWER
      prelim_predictions.append(
        PrelimPrediction(
          feature_index=null_feature_index,
          start_index=0,
          end_index=0,
          start_logit=null_start_logit,
          end_logit=null_end_logit,
          qid=example_qid
      )
        )
      prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
      self.t = prelim_predictions
      # print(ff)
      best_pred = prelim_predictions[0]
      is_answerable = null_score - (best_pred.start_logit + best_pred.end_logit) <= self.answerability_threshold
      if is_answerable:
        feature = example[best_pred.feature_index].feature
        start_char = feature.offset_mappings[best_pred.start_index][0]
        end_char = feature.offset_mappings[best_pred.end_index][1]
        answer = feature.cleaned_context[start_char: end_char + 1]
        # answer = self.improve_answer_quality(answer)
      else:
        answer = 'غیرقابل‌پاسخ'

      self.answers[example_qid] = answer


  def get_best_indexes(self, logits):
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= self.n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes



"""# Model"""

class Persian_Model(nn.Module):

  def __init__(self, bert, device):
    super(Persian_Model, self).__init__()
    self.transformer = bert
    self.start_end_head = nn.Linear(self.transformer.config.hidden_size, 2)
    nn.init.normal_(self.start_end_head.weight, mean=.0, std=.02)
    self.device = device

  def forward(self, x):
    for key in x:
      x[key] = x[key].to(device)
    # transformer output
    transformer_output = self.transformer(**x)
    start_end_logits = self.start_end_head(transformer_output.last_hidden_state)
    start_logits, end_logits = start_end_logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)
    return start_logits, end_logits

"""# Saving Settings"""

is_pretrained = sys.argv[1]

if is_pretrained in ['parsquad', 'quac']:
  pretrain_parsquad_id = '1-carGHsxRtjl23CEOOaVCQzOPLMqgBKA'
  pretrain_quac_id = '1-zKrKtyIGtJhzBlSw56_z8ZfjDcJlkB6'
  if is_pretrained == 'parsquad':
    gdown.download(id=pretrain_parsquad_id)
    target_dir = 'ParSQuAD'
    pretrained_weights = torch.load('checkpoint_3_0_0')
  if is_pretrained == 'quac':
    gdown.download(id=pretrain_quac_id)
    target_dir = 'QuAC'
    pretrained_weights = torch.load('checkpoint_3_0_0')
else:
  target_dir = 'None'

hist_num_str = str(sys.argv[2])

drive_prefix = f'Xlmr_FineTune_{target_dir}_{hist_num_str}/'
drive_checkpoint_dir = 'Checkpoint/'
drive_log_dir = 'Log/'
checkpoint_dir = os.path.join(drive_prefix, drive_checkpoint_dir)
log_dir = os.path.join(drive_prefix, drive_log_dir)

meta_log_file = os.path.join(drive_prefix, drive_log_dir, 'test.txt')
prediction_file_prefix = os.path.join(drive_prefix, drive_log_dir, 'prediction_')
loss_log_file = os.path.join(drive_prefix, drive_log_dir, 'loss.txt')
mean_f1_file = os.path.join(drive_prefix, drive_log_dir, 'mean_f1.txt')
eval_res_file = os.path.join(drive_prefix, drive_log_dir, 'eval_result.json')
test_res_file = os.path.join(drive_prefix, drive_log_dir, 'test_result.json')

if not os.path.exists(drive_prefix):
  os.mkdir(drive_prefix)
  print('crated saved dir')
if not os.path.exists(checkpoint_dir):
  os.mkdir(checkpoint_dir)
if not os.path.exists(log_dir):
  os.mkdir(log_dir)

with open(meta_log_file, 'w') as f:
  pass
# check if drive is accessible
try:
   with open(os.path.join(drive_prefix, drive_log_dir, 'test.txt'), 'r') as f:
      pass
except:
  print('No Access to Drive')
  exit()


def print_loss(loss_collection, epoch, step):
  txt = f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step}/{int(len(train_dataloader) / accumulation_steps)}] | Loss {round(sum(loss_collection) / len(loss_collection), 4)}'
  print(txt)

def save_loss(loss_collection, epoch, step):
  txt = f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step}/{int(len(train_dataloader) / accumulation_steps)}] | Loss {round(sum(loss_collection) / len(loss_collection), 4)}'
  with open(loss_log_file, 'a') as f:
    f.write(txt)
    f.write('\n')



checkpoint_available = False


def save_prediction(epoch, step, prediction_log):
  with open(os.path.join(drive_prefix, drive_log_dir, 'prediction.txt'), 'a') as f:
    f.write(f'--------- EPOCH {epoch} STEP {step} ---------\n')
    f.write(prediction_log)
    f.write('\n')
    f.write('\n')

def save_checkpoint(filename):
  checkpoint_config = {
  'model_dict': model_p.state_dict()}
  torch.save(checkpoint_config, filename)

def load_checkpoint():
    checkpoint_config = torch.load(current_checkpoint)
    return checkpoint_config['model_dict']

"""# Train loop"""

epochs = 0
lr = 5e-6
beta_1 = .9
beta_2 = .999
O = []
eps = 1e-6
batch_size = 5
accumulation_steps = 1
accumulation_counter = 0

best_eval_f1 = 0
best_eval_checkpoint = None

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_p = Persian_Model(deepcopy(model), device).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)

if is_pretrained in ['parsquad', 'quac']:
  model_p.load_state_dict(pretrained_weights['model_dict'])

loss_collection = []
train_dataloader = DataLoader(current_file=0, current_index=0, batch_size=batch_size, shuffle=True, mode='train')
eval_dataloader = DataLoader(current_file=0, current_index=0, batch_size=1, shuffle=False, mode='eval')
test_dataloader = DataLoader(current_file=0, current_index=0, batch_size=1, shuffle=False, mode='test')

each_step_log = 300
start_epoch = 0
start_step = 0
current_file = 0
current_index = 0
k = 3
weight_decay = float(sys.argv[2])
f1_list = []

optimization_steps = int(epochs * len(train_dataloader) / accumulation_steps)
warmup_ratio = .1
warmup_steps = int(optimization_steps * warmup_ratio)

optimizer = AdamW(model_p.parameters(), lr=lr, betas=(beta_1,beta_2), eps=eps, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=optimization_steps)

# laod checkpoint if available
if checkpoint_available:
  print('loading checkpoint')
  model_p_dict = load_checkpoint()
  # load state dicts
  model_p.load_state_dict(model_p_dict)
  optimizer.load_state_dict(optimizer_dict)
  scheduler.load_state_dict(scheduler_dict)

current_file_index_ = current_file
train_dataloader.reset_dataloader(current_file, current_index)
model_p.train()
for epoch in range(start_epoch, epochs):
  train_step = 0
  acc_loss = 0
  log_step = 0

  for data in train_dataloader:
    start_positions = data.pop('start_positions').to(device)
    end_positions = data.pop('end_positions').to(device)
    features = data.pop('features')
    start_logits, end_logits = model_p(data)
    loss = (loss_fn(start_logits, start_positions) + loss_fn(end_logits, end_positions)) / 2
    loss = loss / accumulation_steps
    acc_loss += loss.item()
    loss.backward()

    if len(loss_collection) % each_step_log == 0 and len(loss_collection) != 0:
      print_loss(loss_collection, epoch, log_step + 1)
      save_loss(loss_collection, epoch, log_step + 1)
      loss_collection = []


    accumulation_counter += 1
    if accumulation_counter % accumulation_steps == 0:
      loss_collection.append(acc_loss)
      acc_loss = 0
      log_step += 1
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
      torch.cuda.empty_cache()
      accumulation_counter = 0

    train_step += 1

  model_p.eval()
  print('-------------------- Evaluation --------------------')
  eval_p = EvalProcessOutput()
  with torch.no_grad():
    for step, data in enumerate(eval_dataloader):
      start_positions = data.pop('start_positions')
      end_positions = data.pop('end_positions')
      features = data.pop('features')
      start_logits, end_logits = model_p(data)
      eval_p.process_feature_output(to_numpy(start_logits),
                                    to_numpy(end_logits),
                                    features)

  eval_p.process_output()
  f1, heq_q, heq_d, mean_f1s, res = run_eval('eval', eval_p)
  f1_list.append(f1)
  if f1 > best_eval_f1:
    print('saving model...')
    best_eval_checkpoint = f'best_model_{epoch}.pt'
    save_checkpoint(best_eval_checkpoint)
    print('model saved successfully')
    best_eval_f1 = f1
    with open(eval_res_file, 'w') as f:
      json.dump(res, f)
      
  print('Best Eval F1', best_eval_f1)
  early_stop = all([f1_list[-k] > i for i in f1_list[-k+1:]]) if epoch + 1 >= k else False
  if early_stop:
    print('Early Stopping')
    break
  model_p.train()

checkpoint_config = torch.load('drive/MyDrive/XLMR_FineTune_None_2/Checkpoint/best_model_4.pt')

model_p.load_state_dict(checkpoint_config['model_dict'])
model_p.eval()
print('-------------------- Eval Time --------------------')
eval_p = EvalProcessOutput()
with torch.no_grad():
  for step, data in enumerate(eval_dataloader):
    start_positions = data.pop('start_positions')
    end_positions = data.pop('end_positions')
    features = data.pop('features')
    start_logits, end_logits = model_p(data)
    eval_p.process_feature_output(to_numpy(start_logits),
                                  to_numpy(end_logits),
                                  features)

eval_p.process_output()
f1, heq_q, heq_d, mean_f1s, res = run_eval('eval', eval_p)
with open(eval_res_file, 'w') as f:
  json.dump(res, f)




model_p.load_state_dict(checkpoint_config['model_dict'])
model_p.eval()
print('-------------------- Test Time --------------------')
test_p = EvalProcessOutput()
with torch.no_grad():
  for step, data in enumerate(test_dataloader):
    start_positions = data.pop('start_positions')
    end_positions = data.pop('end_positions')
    features = data.pop('features')
    start_logits, end_logits = model_p(data)
    test_p.process_feature_output(to_numpy(start_logits),
                                  to_numpy(end_logits),
                                  features)

test_p.process_output()
f1, heq_q, heq_d, mean_f1s, res = run_eval('test', test_p)
with open(test_res_file, 'w') as f:
  json.dump(res, f)
