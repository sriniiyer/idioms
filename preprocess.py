import argparse
import torch
import random
import re
import json
import time
import numpy as np
from Tree import TNTTree
from subword_nmt import apply_bpe, learn_bpe
from collections import Counter

def rhs(rule):
  return rule.split('-->', 1)[1]

def lhs(rule):
  return rule.split('-->', 1)[0]

# parents is a dict that stores each rules' parent 
def getChildrenFromProd(rules, index, node, nodepos, parent, parents, parentpos):
  left, right = lhs(rules[index]), rhs(rules[index])
  assert (left == node)
  # Means node is actually a terminal without an expansion rule
  parents[index] = parent
  parentpos[index] = nodepos + 2
  parent = index
  for i, r in enumerate(right.split('___')):
    if left in CDDataset.pre_terminal_symbols or (not r.endswith('_NT')): #terminal, ignore it
      pass
    else:
      # r can be a terminal rule
      index = getChildrenFromProd(rules, index + 1, r, i, parent, parents, parentpos)
  return index

# for concode
def isGetter(codeToks):
  return re.search(r"function \( \) \{ return concodeclass_[a-zA-Z0-9_]+ ; \}", ' '.join(codeToks)) != None

def isSetter(codeToks):
  return re.search(r"function \( .* \) \{ concodeclass_[a-zA-Z0-9_]+ = .* ; \}", ' '.join(codeToks)) != None

def split_camel(identifier):
  matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
  return [m.group(0).lower() for m in matches]

def split_underscore(identifier):
  return [x for x in identifier.strip('_').split('_') if x != '']

def split_tokens(identifier, toks):
  for tok in toks:
    identifier = identifier.replace(tok, ' ' + tok + ' ')
  return identifier.split(' ')

def processToken(identifier, vocabs):
  toks = split_tokens(identifier, ['<', '>', '[', ']'])
  toks = [y for x in toks for y in split_underscore(x)]
  toks = [y for x in toks for y in split_camel(x)]
  if 'bpe' in vocabs:
    return vocabs['bpe'].segment_tokens(toks)
  else:
    return toks

def split_camel_case(identifier, vocabs):
  if 'use_new_split' in vocabs and vocabs['use_new_split']:
    return processToken(identifier, vocabs)

  # hack to return characters
  matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
  camel_split = [m.group(0).lower() for m in matches]

  if 'bpe' in vocabs:
    return vocabs['bpe'].segment_tokens(camel_split)
  else:
    return camel_split

def combine_dicts(d1, d2):
  comb = d1
  for k in d2:
    if k not in comb:
      comb[k] = d2[k]
    else:
      for val in d2[k]:
        if val not in comb[k]:
          comb[k].append(val)

  return comb

def expandBatchOneHot(batch, pad, width=None):
  vocab_size = batch.max() + 1 if width == None else width
  new_batch = np.full((batch.size(0), batch.size(1), vocab_size), 0) # This is a mask.
  for i in range(0, batch.size(0)):
    for j in range(0, batch.size(1)):
      if batch.dim() == 3:
        for k in range(0, batch.size(2)):
          if batch[i][j][k] != pad: # pad is 0. Ignore it
            new_batch[i][j][batch[i][j][k]] = 1.0
      elif batch.dim() == 2:
        if batch[i][j] != pad:
          new_batch[i][j][batch[i][j]] = 1.0
  return torch.FloatTensor(new_batch)

def make_batch_elem_into_tensor(batch, entry, pad):
  seq_len = max(len(elem[entry]) for elem in batch)
  torch_batch = np.full((len(batch), seq_len), pad) #torch.LongTensor(seq_len, len(batch)).fill_(pad)
  for i in range(0, len(batch)):
    for j in range(0, len(batch[i][entry])):
      torch_batch[i][j] = batch[i][entry][j]
  return torch.LongTensor(torch_batch)

def make_batch_char_elem_into_tensor(batch, entry, pad, maxl=None, minl=None):
  max_char_length = min(maxl, max(len(chars) for elem in batch for chars in elem[entry]))
  max_char_length = max(max_char_length, minl)
  torch_batch = np.full((len(batch), max_char_length, max(len(elem[entry]) for elem in batch)), pad)

  for i in range(0, len(batch)):
    for j in range(0, len(batch[i][entry])):
      for k in range(0, min(max_char_length, len(batch[i][entry][j]))):
        torch_batch[i][k][j] = batch[i][entry][j][k]
  return torch.LongTensor(torch_batch)

class Vocab():

  def addSymbol(self, sym):
    self.stoi[sym] = len(self.itos)
    self.itos.append(sym)

  def __init__(self, elements, prune, max_num, start=True, stop=True, pad=True, unk=True, rule=False, bpe=-1):
    self.start = start
    self.stop = stop
    self.codes = None
    vocab = Counter()
    self.max_num = max_num
    self.itos = []
    self.stoi = {}
    if pad:
      self.addSymbol('<blank>')
    if unk:
      self.addSymbol('<unk>')
    if start:
      self.addSymbol('<s>')
    if stop:
      self.addSymbol('</s>')
    self.rule = rule
    if rule: # Adding these for both ATIS and CONCODE. Extra things in the vocab are ok.
      for pre_terminal in CDDataset.pre_terminal_symbols:
        self.addSymbol(CDDataset._unk_rule_from_Nt(pre_terminal))

    if bpe >= 0:
      self.codes = learn_bpe.learn_bpe(elements, bpe, 0) #  last is min freq
      b = apply_bpe.BPE(self.codes)
      elements = b.segment_tokens(elements)

    for w in elements:
      vocab[w] += 1
    if bpe >= 0:
      print('Vocab size {}'.format(len(vocab)))

    # prune low frequency words
    max_vocab = self.max_num if not rule else 100000000000
    for (w, f) in vocab.most_common(max_vocab):
      if ( (rule == False and f > prune) or (rule == True and not CDDataset._is_terminal_rule(w)) or (rule == True and CDDataset._is_terminal_rule(w) and len(self.itos) < self.max_num)  or w.endswith("_concodeNT")):
        word = w.replace('concodeclass_', '').replace('concodefunc_', '')
        self.itos.append(word)
        self.stoi[word] = len(self.itos) - 1
      else: #map everything else to unk
        if rule:
          # We need the right kind of UNK rule here
          mapped_to_known_unk = False
          for pre_terminal in CDDataset.pre_terminal_symbols:
            if pre_terminal in w:
              self.stoi[w] = self.stoi[CDDataset._unk_rule_from_Nt(pre_terminal)]
              mapped_to_known_unk = True
              break

          if not mapped_to_known_unk:
            # An unk type we dont know about. Investigate.
            import ipdb; ipdb.set_trace()
            # For next_rules, we cannot have any other type of unk
            self.stoi[w] = self.stoi['<unk>']
        else:
          self.stoi[w] = self.stoi['<unk>']

  def __len__(self):
    return len(self.itos)

  def addStartOrEnd(self, words):
     return (['<s>'] if self.start else []) + words + (["</s>"] if self.stop else [])

  # the char parameter is only for recursion
  def to_num(self, words, char=0, start=True, stop=True):
      # will be 2 dimensional if its char 
      if char > 0:
        ret = [self.to_num(list(word), char=char - 1) for word in words]
      else:
        start_sym = [self.stoi['<s>']] if self.start and start else []
        stop_sym = [self.stoi['</s>']] if self.stop and stop else []
        if self.rule:
          ret = start_sym
          try:
            for w in words:
              ret += [self.stoi[w] if w in self.stoi else self.stoi[CDDataset._unk_rule_from_Nt(CDDataset.getAnonRule(w))]]
          except:
            import ipdb; ipdb.set_trace()
          ret += stop_sym
        else:
          ret = start_sym
          for w in words:
            try:
              ret += [self.stoi[w] if w in self.stoi else self.stoi['<unk>']]
            except:
              import ipdb; ipdb.set_trace()
          ret += stop_sym
      return ret

class Dataset():
  def compute_batches(self, batch_size, vocabs, max_chars, rank, num_gpus, decoder_type, randomize=True, trunc=-1, no_filter=False):
    timer = time.process_time()

    self.batches = []
    curr_batch = []
    total = 0
    for i in range(rank, len(self.examples), num_gpus):
      if not no_filter and decoder_type in ["concode", "prod"] and len(self.examples[i]['next_rules']) > 200:
        continue
      total += 1
      curr_batch.append(self.examples[i])
      if len(curr_batch) == batch_size or i == (len(self.examples) - 1) or i == trunc:
        self.batches.append(self.make_batch_into_tensor(curr_batch, vocabs, max_chars))
        curr_batch = []
      if i == trunc:
        break

    if randomize:
      random.shuffle(self.batches)
    print('Computed batched in :' + str(time.process_time() - timer) + ' secs')
    return total

class CDDataset(Dataset):
  # Misnomer. These are pre_terminal symbols that can trigger copy actions. So, Variable_NT isnt here, because we dont want to copy variables from the NL
  pre_terminal_symbols = TNTTree.pre_terminal_symbols

  @staticmethod
  def _is_terminal_rule(rule):
    return lhs(rule) in CDDataset.pre_terminal_symbols

  @staticmethod
  def _unk_rule_from_Nt(Nt):
    return Nt + '--><' + Nt.lower()  + '_unk>'

  @staticmethod
  def _unk_rule(rule):
    for pre_terminal in CDDataset.pre_terminal_symbols:
      if CDDataset._unk_rule_from_Nt(pre_terminal) == rule:
        return True
    return False

  @staticmethod
  def getAnonRule(rule):
    if lhs(rule) in CDDataset.pre_terminal_symbols:
      return lhs(rule)
    else:
      return rule

  def __init__(self, dataFile, opt, test=False, trunc=-1, shuffle=False, test_tgt_min_seq_length=0):
    self.examples = []
    self.rhs = {}
    self.opt = opt
    dataset = json.loads(open(dataFile, 'r').read())
    if shuffle:
      random.seed(1123)
      random.shuffle(dataset)

    max_code = max([len(js['code']) for js in dataset])
    print('Maximum code toks: ' + str(max_code))
    length_filtered = 0
    for js in dataset:
      length_correct = (test and len(js['code']) >= test_tgt_min_seq_length) or (not test and len(js['seq2seq']) <= opt.src_seq_length and len(js['code']) <= opt.tgt_seq_length)
      if length_correct:

        # Important: This should be done after copy!
        js['rules_with_tag'] = []
        for i in range(0, len(js['rules'])):
          js['rules_with_tag'].append(js['rules'][i])
          js['rules'][i] = js['rules'][i].replace('concodeclass_', '').replace('concodefunc_', '')


        nonTerminals = [rule.split('-->')[0] for rule in js['rules']]
        prevRules = [CDDataset.getAnonRule(x) for x in js['rules']]

        parents = {}
        parentpos = {}
        children = {}
        parentRules = []
        parentposvec = []
        getChildrenFromProd(js['rules'], 0, {"atis": "LogicalForm_NT", "concode": "MemberDeclaration_NT", "sql": "Statement_NT"}[opt.dataset], -2, -1, parents, parentpos) # -2 coz we are adding 2 to the nodepos. Becuase it starts with lhs and then <sep>
        for i in range(0, len(js['rules'])):
          if i > 0: # When i == 0, the parent will be <s>, and it will be appended by the vocab[prev_rules]
            parentRules.append(CDDataset.getAnonRule(js['rules'][parents[i]]))

          if parents[i] not in children:
            children[parents[i]] = []
          children[parents[i]].append(i)
          parentposvec.append(parentpos[i])

        self.examples.append(
          {'src': js['nl'],
           'origcode': js['code'],
           'code': [x.replace('concodeclass_', '').replace('concodefunc_', '') for x in js['code']],
           'next_rules': js['rules'],
           'next_rules_with_tag': js['rules_with_tag'],
           'prev_rules': prevRules,
           'parent_rules': parentRules,
           'nt': nonTerminals,
           'seq2seq': js["seq2seq_nop"],
           'children' : children,
           'parents' : parents, # For every rule i, parents[i] is the index of its parent rule
           'parentpos': parentposvec # For every rule i, b -> something, parentpos[i] is the index of b in its parent
           }
        )

        # This file is used for atis too. These are the extra things needed for concode
        if opt.dataset == "concode":
          self.examples[-1].update({
           'varNames': js['varNames'],
           'varTypes': js['varTypes'],
           'methodNames': js['methodNames'],
           'methodReturns': js['methodReturns'],
           'concode':[j for i in zip(js['varTypes'], js['varNames']) for j in i] + [j for i in zip(js['methodReturns'], js['methodNames']) for j in i], # alternating type, name, type, name
           'concode_vocab': Vocab(js['varNames'] + js['varTypes']  + js['methodReturns'] + js['methodNames'] + ['concode_copy_placeholder'], 0, 1000000, start=False, stop=False),
           'concode_var': [j for i in zip(js['varTypes'], js['varNames']) for j in i], # Alternating type, name, type, name
           'concode_method': [j for i in zip(js['methodReturns'], js['methodNames']) for j in i]
          })

        #compute seq2seq copy vector
        seq2seq_copy = []
        for w in range(0, len(self.examples[-1]['code'])):
          codeTok = self.examples[-1]['code'][w]
          tmpCopy = []
          for s in range(0, len(self.examples[-1]['seq2seq'])):
            srcTok = self.examples[-1]['seq2seq'][s]
            if srcTok == codeTok and srcTok != ';' and srcTok != ':':
              tmpCopy.append(1)
            else:
              tmpCopy.append(0)
          seq2seq_copy.append(tmpCopy)
        self.examples[-1]['seq2seq_copy'] = seq2seq_copy

        # For every nt, store the list
        # of possible rights
        for rule in js['rules']:
          (nt, r) = rule.split('-->')
          if nt not in self.rhs:
            self.rhs[nt] = []
          if rule not in self.rhs[nt]:
            self.rhs[nt].append(rule)
        if len(self.examples) == trunc: # If trunc is -1, this will never be true
          break

      else:
        length_filtered += 1

      if len(self.examples) % 100 == 0:
        print("Done: " + str(len(self.examples)))

    print('Number length filtered: {}'.format(length_filtered))

    # sort by src length
    if not test:
      self.examples.sort(key=lambda x: len(x['src']), reverse=True)

  def toNumbers(self, vocabs, prevRules=True):
    if 'names_combined' in vocabs and vocabs['names_combined'].codes != None:
      vocabs['names_combined'].bpe = apply_bpe.BPE(vocabs['names_combined'].codes)
      vocabs['bpe'] = apply_bpe.BPE(vocabs['names_combined'].codes)
    if vocabs['seq2seq'].codes != None:
      vocabs['seq2seq'].bpe = apply_bpe.BPE(vocabs['seq2seq'].codes)

    for e in self.examples:

      e['code_nums'] = vocabs['code'].to_num(e['code'])
      seq2seq_tokens = vocabs['seq2seq'].bpe.segment_tokens(e['seq2seq']) if vocabs['seq2seq'].codes is not None else e['seq2seq']
      e['seq2seq_nums'] = vocabs['seq2seq'].to_num(seq2seq_tokens)
      e['seq2seq_vocab'] = Vocab(seq2seq_tokens, 0, 100000000, start=False, stop=False) # A vocab just for this sentence
      e['seq2seq_in_src_nums'] = e['seq2seq_vocab'].to_num(vocabs['seq2seq'].addStartOrEnd(seq2seq_tokens)) # use the local vocab for this sentence
      e['code_in_src_nums'] = e['seq2seq_vocab'].to_num(vocabs['code'].addStartOrEnd(e['code'])) # use the local vocab for this sentence

      if self.opt.dataset == "concode":
        # For concode decoder------- -------
        # We have to do this because we concat them in the decoder
        # and there is padding between the nl, vars and methods in the same example because of batching
# This isnt used, commenting it out
#         e['src_in_src_nums'] = e['concode_vocab'].to_num(e['src']) # use the local vocab for this sentence
        e['var_in_src_nums'] = e['concode_vocab'].to_num(e['concode_var']) # use the local vocab for this sentence
        e['method_in_src_nums'] = e['concode_vocab'].to_num(e['concode_method']) # use the local vocab for this sentence
        #-------------------------------------------------------
        e['concode_next_rules_in_src_nums'] = e['concode_vocab'].to_num(
          vocabs['next_rules'].addStartOrEnd(
            [rhs(x) if lhs(x) in CDDataset.pre_terminal_symbols else '<unk>' for x in e['next_rules']]
          )) # use the local vocab for this sentence
        #------------------------

        # --- Our Model -----------
        e['src_nums'] = vocabs['names_combined'].to_num([y for w in e['src'] for y in split_camel_case(w, vocabs)])
        e['varTypes_nums'] = vocabs['names_combined'].to_num([(split_camel_case(w, vocabs)) for w in e['varTypes']], char=1)
        e['methodReturns_nums'] = vocabs['names_combined'].to_num([(split_camel_case(w, vocabs)) for w in e['methodReturns']], char=1)
        e['varNames_nums'] = vocabs['names_combined'].to_num([(split_camel_case(w, vocabs)) for w in e['varNames']], char=1)
        e['methodNames_nums'] = vocabs['names_combined'].to_num([ (split_camel_case(w, vocabs)) for w in e['methodNames']], char=1)

        #-----------------------------------
      e['next_rules_in_src_nums'] = e['seq2seq_vocab'].to_num(
        vocabs['next_rules'].addStartOrEnd(
          [rhs(x) if lhs(x) in CDDataset.pre_terminal_symbols else '<unk>' for x in e['next_rules']]
        )) # use the local vocab for this sentence

      # ------- Rule decoder
      # There is no unk in the vocab, so this will throw an error
      # if the rule isnt there in the vocab
      if prevRules:
        # We don't need to do this during prediction?
        e['prev_rules_nums'] = vocabs['prev_rules'].to_num(e['prev_rules'][:-1])
        e['prev_rules_split_nums'] = vocabs['nt'].to_num([['<s>']] + [[w] if '-->' not in w else [lhs(w)] + ['<sep>'] + rhs(w).split('___') for w in e['prev_rules'][:-1]], char=1)
        e['parent_rules_nums'] = vocabs['prev_rules'].to_num(e['parent_rules'])
        e['parent_rules_split_nums'] = vocabs['nt'].to_num([['<s>']] + [[w] if '-->' not in w else [lhs(w)] + ['<sep>'] + rhs(w).split('___') for w in e['parent_rules']], char=1)

        # We need to ensure that only certain rules can be unked, not all. This
        # is taken care of when building the vocab
        e['nt_nums'] = vocabs['nt'].to_num(e['nt'])
        e['next_rules_nums'] = vocabs['next_rules'].to_num(e['next_rules'])
        #-------------------------------------

  def outputStats(self, vocabs):
    print('Average NL length: ' + str(sum([len(e['src']) for e in self.examples]) * 1.0 / len(self.examples)))
    print('Average Code Characters: ' + str(sum([len(' '.join(e['code'])) for e in self.examples]) * 1.0 / len(self.examples)))
    print('Average Code Tokens : ' + str(sum([len(e['code']) for e in self.examples]) * 1.0 / len(self.examples)))
    print('Average Rule length: ' + str(sum([len(e['next_rules']) for e in self.examples]) * 1.0 / len(self.examples)))
    print('Max Code Tokens : ' + str(max([len(e['code']) for e in self.examples])))
    print('Average AST Nodes: ' + str(sum([len(rhs(r).split('___')) for e in self.examples for r in e['next_rules']]) * 1.0 / len(self.examples)))
    print('Max AST Nodes: ' + str(max([len(rhs(r).split('___')) for e in self.examples for r in e['next_rules']]) ))
    if opt.dataset == "concode":
      print('Percent getters: ' + str(sum([int(isGetter(e['origcode'])) for e in self.examples]) * 1.0 / len(self.examples)))
      print('Percent setters: ' + str(sum([int(isSetter(e['origcode'])) for e in self.examples]) * 1.0 / len(self.examples)))

      var_copies = np.mean([1 if "concodeclass_" in ' '.join(e['origcode']) else 0 for e in self.examples]) * 100.0
      fn_copies = np.mean([1 if "concodefunc_" in ' '.join(e['origcode']) else 0 for e in self.examples]) * 100.0
      def match_source(src, code, names):
        for w in src:
          if  (w not in vocabs['code'].stoi or vocabs['code'].stoi[w] == vocabs['code'].stoi['<unk>']) and w in code and w not in names:
            return True
        return False
      src_copies = np.mean([1 if match_source(e['src'], e['origcode'], e['varNames'] + e['varTypes'] + e['methodReturns'] + e['methodNames']) else 0 for e in self.examples]) * 100.0

      def match_type(type_list, code):
        for typ in type_list:
          if typ in code and (typ not in vocabs['code'].stoi or vocabs['code'].stoi[typ] == vocabs['code'].stoi['<unk>']):
            return True
        return False

      type_copies = np.mean([1 if match_type(e['methodReturns'] + e['varTypes'], e['origcode']) else 0 for e in self.examples]) * 100.0

      print('Number of variable copies: {}, function copies: {}, source copies: {}, Type copies: {} '.format(var_copies, fn_copies, src_copies, type_copies))


  @staticmethod
  def make_batch_into_tensor(batch, vocabs, max_chars):

    torch_batch = {}
    # -------- for seq2seq
    torch_batch['seq2seq'] = make_batch_elem_into_tensor(batch, 'seq2seq_nums', vocabs['seq2seq'].stoi['<blank>'])
    torch_batch['code'] = make_batch_elem_into_tensor(batch, 'code_nums', vocabs['code'].stoi['<blank>'])
    local_vocab_blank = batch[0]['seq2seq_vocab'].stoi['<blank>']
    torch_batch['seq2seq_in_src'] = make_batch_elem_into_tensor(batch, 'seq2seq_in_src_nums', local_vocab_blank)
    # src_map maps positions in the source to source vocab entries, so that we can accumulate copy scores for each vocab entry based on all
    # positions in which it appears
    torch_batch['src_map'] = expandBatchOneHot(torch_batch['seq2seq_in_src'], local_vocab_blank) # src token mapped to vocab

    if 'concode_vocab' in batch[0]:
      #-----------for concode
      max_local_vocab_in_batch = max(len(x['concode_vocab']) for x in batch)
#       torch_batch['src_in_src'] = make_batch_elem_into_tensor(batch, 'src_in_src_nums', batch[0]['concode_vocab'].stoi['<blank>'])
      torch_batch['var_in_src'] = make_batch_elem_into_tensor(batch, 'var_in_src_nums', batch[0]['concode_vocab'].stoi['<blank>'])
      torch_batch['method_in_src'] = make_batch_elem_into_tensor(batch, 'method_in_src_nums', batch[0]['concode_vocab'].stoi['<blank>'])
      torch_batch['concode_src_map_methods'] = expandBatchOneHot(torch_batch['method_in_src'], batch[0]['concode_vocab'].stoi['<blank>'], width=max_local_vocab_in_batch)
      torch_batch['concode_src_map_vars'] = expandBatchOneHot(torch_batch['var_in_src'], batch[0]['concode_vocab'].stoi['<blank>'], width=max_local_vocab_in_batch)
      torch_batch['concode_vocab'] = [b['concode_vocab'] for b in batch] # Store this for replace unk
      torch_batch['concode_next_rules_in_src_nums'] = make_batch_elem_into_tensor(batch, 'concode_next_rules_in_src_nums', local_vocab_blank)
      torch_batch['concode'] = [b['concode'] for b in batch] # Store this for replace unk
      torch_batch['concode_var'] = [b['concode_var'] for b in batch] # Store this for replace unk
      torch_batch['concode_method'] = [b['concode_method'] for b in batch] # Store this for replace unk
      #---------------------------------------------

    torch_batch['code_in_src_nums'] = make_batch_elem_into_tensor(batch, 'code_in_src_nums', local_vocab_blank)
    torch_batch['next_rules_in_src_nums'] = make_batch_elem_into_tensor(batch, 'next_rules_in_src_nums', local_vocab_blank)
    torch_batch['seq2seq_vocab'] = [b['seq2seq_vocab'] for b in batch] # Store this for replace unk
    torch_batch['raw_code'] = [b['code'] for b in batch] # Store this for replace unk
    torch_batch['raw_seq2seq'] = [b['seq2seq'] for b in batch] # Store this for replace unk
    torch_batch['parents'] = [b['parents'] for b in batch] #
    torch_batch['parentpos'] = make_batch_elem_into_tensor(batch, 'parentpos', 0)
    #-------------------------Prod Decoder
    if 'prev_rules_nums' in batch[0]:
      # prev rules will not be there during testing. So don't compute these.
      torch_batch['nt'] = make_batch_elem_into_tensor(batch, 'nt_nums', vocabs['nt'].stoi['<blank>'])
      torch_batch['prev_rules'] = make_batch_elem_into_tensor(batch, 'prev_rules_nums', vocabs['prev_rules'].stoi['<blank>'])
      torch_batch['prev_rules_split'] = make_batch_char_elem_into_tensor(batch, 'prev_rules_split_nums', pad=vocabs['nt'].stoi['<blank>'], maxl=1000, minl=1)
      torch_batch['parent_rules'] = make_batch_elem_into_tensor(batch, 'parent_rules_nums', vocabs['prev_rules'].stoi['<blank>'])
      torch_batch['parent_rules_split'] = make_batch_char_elem_into_tensor(batch, 'parent_rules_split_nums', pad=vocabs['nt'].stoi['<blank>'], maxl=1000, minl=1)

      torch_batch['next_rules'] = make_batch_elem_into_tensor(batch, 'next_rules_nums', vocabs['next_rules'].stoi['<blank>'])
      torch_batch['children'] = [b['children'] for b in batch] # Store this for replace unk

    torch_batch['seq2seq_copy'] = CDDataset.stack_with_padding([torch.LongTensor(b['seq2seq_copy']) for b in batch], 0, start_symbol=True, stop_symbol=True)
    #------------------------------

    if 'concode_vocab' in batch[0]:
      #---- Our Encoder --------------
      torch_batch['src'] = make_batch_elem_into_tensor(batch, 'src_nums', vocabs['names_combined'].stoi['<blank>'])
      torch_batch['varTypes'] = make_batch_char_elem_into_tensor(batch, 'varTypes_nums', pad=vocabs['names_combined'].stoi['<blank>'], maxl=max_chars, minl=1)
      torch_batch['methodReturns'] = make_batch_char_elem_into_tensor(batch, 'methodReturns_nums', pad=vocabs['names_combined'].stoi['<blank>'], maxl=max_chars, minl=1)
      torch_batch['varNames'] = make_batch_char_elem_into_tensor(batch, 'varNames_nums', pad=vocabs['names_combined'].stoi['<blank>'], maxl=max_chars, minl=1)
      torch_batch['methodNames'] = make_batch_char_elem_into_tensor(batch, 'methodNames_nums', pad=vocabs['names_combined'].stoi['<blank>'], maxl=max_chars, minl=1)
      torch_batch['raw_src'] = [b['src'] for b in batch] # Store this for replace unk
      torch_batch['raw_varNames'] = [b['varNames'] for b in batch] # Store this for replace unk
      torch_batch['raw_methodNames'] = [b['methodNames'] for b in batch] # Store this for replace unk
      #-------------------------------------

    return torch_batch

  @staticmethod
  def stack_with_padding(batch, pad_, start_symbol=False, stop_symbol=False):
    max_sizes = [len(batch[0]), len(batch[0][0])]
    for b in batch:
      if len(b) > max_sizes[0]:
        max_sizes[0] = len(b)
      if (len(b[0]) > max_sizes[1]):
        max_sizes[1] = len(b[0])

    t = torch.LongTensor(len(batch), max_sizes[0], max_sizes[1]).fill_(pad_)
    for i in range(0, len(batch)):
      for j in range(0, batch[i].size(0)):
        for k in range(0, batch[i].size(1)):
          t[i][j][k] = batch[i][j][k]

    if start_symbol:
      t = torch.cat((torch.LongTensor(len(batch), 1, max_sizes[1]).fill_(pad_), t), 1)
    if stop_symbol:
      t = torch.cat((t, torch.LongTensor(len(batch), 1, max_sizes[1]).fill_(pad_)), 1)
    return t


  @staticmethod
  def compute_masks(rhs, vocabs):
    masks = torch.LongTensor(len(vocabs['nt'].itos), len(vocabs['next_rules'].itos)).fill_(-10000000)  # nt x rules
    for (nt, rules) in rhs.items():
      nt_num = vocabs['nt'].stoi[nt]
      for r in rules:
        r_num = None
        if r in vocabs['next_rules'].stoi:
          r_num = vocabs['next_rules'].stoi[r]
        elif CDDataset._is_terminal_rule(r):
          r_num = vocabs['next_rules'].stoi[CDDataset._unk_rule_from_Nt(CDDataset.getAnonRule(r))]
        masks[nt_num][r_num] = 0
    return masks


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='preprocess.py')
  parser.add_argument('-dataset', required=True,
                      help="Path to the training source data")
  parser.add_argument('-train', required=True,
                      help="Path to the training source data")
  parser.add_argument('-valid', required=True,
                      help="Path to the validation source data")
  parser.add_argument('-test', required=True,
                      help="Path to the validation source data")
  parser.add_argument('-src_seq_length', type=int, default=50,
                      help="Maximum source sequence length")
  parser.add_argument('-tgt_seq_length', type=int, default=50,
                      help="Maximum target sequence length to keep.")
  parser.add_argument('-seq2seq_words_min_frequency', type=int, required=True)
  parser.add_argument('-seq2seq_words_max_vocab', type=int, required=True)
  parser.add_argument('-tgt_words_min_frequency', type=int, required=True)
  parser.add_argument('-next_rules_max_vocab', type=int, required=True)
  parser.add_argument('-names_min_frequency', type=int, default=1)
  parser.add_argument('-names_max_vocab', type=int, default=-1)
  parser.add_argument('-bpe_vocab', type=int, default=-1)
  parser.add_argument('-train_max', type=int, default=200000)
  parser.add_argument('-valid_max', type=int, default=5000)
  parser.add_argument('-save_data', required=True,
                      help="Output file for the prepared data")
  parser.add_argument('-use_new_split', action='store_true',
                      help="Output file for the prepared data")
  opt = parser.parse_args()
  print(opt)
  if opt.dataset == "sql":
    assert(opt.seq2seq_words_min_frequency == 0)
    assert(opt.tgt_words_min_frequency == 0)
  elif opt.dataset == "concode":
    assert(opt.seq2seq_words_min_frequency >= 6)
    assert(opt.tgt_words_min_frequency >= 2)
    assert(opt.names_min_frequency >= 7)
    assert(opt.names_max_vocab >= 0)

  valid = CDDataset(opt.valid, opt, trunc=opt.valid_max)
  train = CDDataset(opt.train, opt, trunc=opt.train_max)
  test = CDDataset(opt.test, opt, trunc=opt.valid_max)

  print("Building Vocab...")
  vocabs = {'use_new_split': opt.use_new_split}

  if opt.dataset == "concode":
    vocabs.update({
      'names_combined': Vocab(
      [c for e in train.examples for w in e['src'] for c in split_camel_case(w, vocabs)] +\
      [c for e in train.examples for w in e['methodNames'] for c in split_camel_case(w, vocabs)] + \
      [c for e in train.examples for w in e['varNames'] for c in split_camel_case(w, vocabs)] + ([c for e in train.examples for w in e['varTypes'] for c in split_camel_case(w, vocabs)] + [c for e in train.examples for w in e['methodReturns'] for c in split_camel_case(w, vocabs)])
      , opt.names_min_frequency if opt.bpe_vocab < 0 else 0, opt.names_max_vocab if opt.bpe_vocab < 0 else 10000000000000, start=False, stop=False, bpe=opt.bpe_vocab),
    })
  vocabs.update({
    'seq2seq': Vocab([w for e in train.examples for w in e['seq2seq']], opt.seq2seq_words_min_frequency if opt.bpe_vocab < 0 else 0, opt.seq2seq_words_max_vocab if opt.bpe_vocab < 0 else 10000000000000, start=False, stop=False, bpe=opt.bpe_vocab),
    'code': Vocab([w for e in train.examples for w in e['code']], opt.tgt_words_min_frequency, 25000),
    'dataset': opt.dataset
  })

  vocabs['next_rules'] = Vocab(
    [w for e in train.examples for w in e['next_rules_with_tag']] + \
    [w for e in valid.examples for w in e['next_rules_with_tag'] if not CDDataset._is_terminal_rule(w)], 
    opt.tgt_words_min_frequency, opt.next_rules_max_vocab, start=False, stop=False, pad=True, rule=True, unk=False)

  vocabs['prev_rules'] = Vocab(
    [CDDataset.getAnonRule(x) for x in vocabs['next_rules'].stoi],
    0, 100000000, stop=False, pad=True, unk=False)

  vocabs['nt'] = Vocab(
      [w for e in train.examples for w in e['nt']] + CDDataset.pre_terminal_symbols * 10  + ['<sep>'] * 10 + [y for x in vocabs['prev_rules'].stoi if "-->" in x for y in rhs(x).split('___')] + ['<s>'] * 10 , 0, 10000, start=False, stop=False, pad=True, unk=False)

  train.toNumbers(vocabs)
  print('Training stats')
  train.outputStats(vocabs)

  print("Building Valid...")
  valid.toNumbers(vocabs)
  print('Valid stats')
  valid.outputStats(vocabs)

  vocabs['rhs'] = combine_dicts(train.rhs, valid.rhs)
  mask = CDDataset.compute_masks(vocabs['rhs'], vocabs) # compute_masks needs rhs
  vocabs['mask'] = mask

  print("Saving train/valid/vocabs")
  print('Vocab Statistics')
  for key in vocabs:
    try:
      print(key + ' : ' + str(len(vocabs[key].itos)) + '/' + str(len(vocabs[key].stoi)) )
    except:
      pass

  torch.save(vocabs, open(opt.save_data + '.vocab.pt', 'wb'))
  torch.save(train, open(opt.save_data + '.train.pt', 'wb'))
  torch.save(valid, open(opt.save_data + '.valid.pt', 'wb'))
