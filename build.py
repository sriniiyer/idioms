import json
import argparse
import collections
import os
from Tree import getAllSubtreesOfDepth2, applyIdiom, treeFromJSON
from multiprocessing import Pool
from itertools import product
import random
import time
from typing import Tuple, List, Dict
from ConcodeProcessor import Processor, ConcodeProcessor, AtisSqlProcessor
from enum import Enum

random.seed(1123)

# Get trees before we do BPE. This is a helper for Pool in bpe()
# This runs a parser on the raw code
def treeFromJson(id_js):
  (idx, js) = id_js
  processor = ThisProcessor(js)
  tnt = processor.getTree()
  return tnt

def bpe(lines, num_steps, fname):

  idioms = []
  idf = open(fname, "w")

  start_time = time.time()
  with Pool(processes=opt.threads) as pool:
    dataset_trees = pool.map(treeFromJson, lines)
  elapsed_time = time.time() - start_time
  print('Finished creating all the trees. Took {} seconds.'.format(elapsed_time))

  for idiom_number in  range(0, num_steps):
    print('Doing idiom_number {}'.format(idiom_number))
    depth2_subtrees = {}

    start_time = time.time()

    with Pool(processes=opt.threads) as pool:
      ret = pool.starmap(getAllSubtreesOfDepth2, product(dataset_trees, [opt.dataset]))
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    start_time = time.time()
    bestTree = None
    bestTreeFreq = 0
    for strees in ret:
      for r in strees:
        if r not in depth2_subtrees:
          depth2_subtrees[r] = 0
        freq = depth2_subtrees[r]
        depth2_subtrees[r] = freq + 1
        if freq + 1 > bestTreeFreq:
          bestTree = r
          bestTreeFreq = freq + 1
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    most_common_depth2_subtree_json = bestTree
    most_common_depth2_subtree = treeFromJSON(json.loads(most_common_depth2_subtree_json))
    print(most_common_depth2_subtree)

    # Apply this rule to everything
    start_time = time.time()

    # This is faster than using Pool. Maybe because we dont need to do the product?
    for k in range(0, len(dataset_trees)):
      applyIdiom(dataset_trees[k], most_common_depth2_subtree, collections.Counter()) 

    idioms.append(most_common_depth2_subtree_json)

    # Doing lots of rewriting here.
    idf.write(most_common_depth2_subtree_json + '\n')
    idf.flush()

    elapsed_time = time.time() - start_time
    print(elapsed_time)

  idf.close()
  return idioms, dataset_trees

class ProcessError(Enum):
  GOOD = 1
  BAD_EXAMPLE = 2
  BAD_PRODUCTIONS = 3
  DIDNT_PARSE = 4
  ERROR_EXAMPLE = 5

def compressExample(nl_code: Tuple[int, str, str], trainNls, orig) -> Tuple[List[Dict], collections.Counter, ProcessError]:
  compressed_examples = []
  (idx, js) = nl_code
  processor = ThisProcessor(js)
  tnt = processor.getTree()

  initial_nodes = len(tnt.vertices())
  idioms_applied : collections.Counter = collections.Counter()

  try:
    example_template = processor.getTemplate(idx, trainNls) # This can be None if the example doesnt have NL etc.
  except:
    return (compressed_examples, idioms_applied, ProcessError.ERROR_EXAMPLE, 0, 0)

  if not example_template:
    return (compressed_examples, idioms_applied, ProcessError.BAD_EXAMPLE, 0, 0)

  # Lets also have an option to keep the original uncompressed tree if we need to
  try:
    original_rule_seq = ThisProcessor.getProductions(tnt)
  except:
    assert('Could not get productions. Please investigate.')
    return (compressed_examples, idioms_applied, ProcessError.BAD_PRODUCTIONS) # will be empty

  if not original_rule_seq:
    return (compressed_examples, idioms_applied, ProcessError.DIDNT_PARSE, 0, 0) # will be empty

  if orig:
    # For valid and test, we need the original
    compressed_examples.append(dict(example_template, **{'rules': original_rule_seq}))

  total_nodes = 0 # before compression
  greedy_compression = 0

  # For train and valid
  if len(idioms_loaded) > 0: # This check lets us avoid creating extra examples for the no idioms case
    # bpe idioms have to be applied in order
    tnt = tnt.applyAllIdioms(idioms_loaded, idioms_applied)

    greedy_rule_seq = ThisProcessor.getProductions(tnt)
    assert(greedy_rule_seq is not None)

    compressed_examples.append(dict(example_template, **{'rules': greedy_rule_seq}))
    final_nodes = len(tnt.vertices())
    print('Code greedily compressed from {} to {}'.format(initial_nodes, final_nodes))
    greedy_compression = final_nodes
    total_nodes = initial_nodes

  return (compressed_examples, idioms_applied, ProcessError.GOOD, greedy_compression, total_nodes)


def loadFile(fname, trunc):
  lines = []
  for i, line in enumerate(open(fname, 'r')):
    if i % 100 == 0:
      print(i)
    if len(lines) >= trunc:
      break
    js = json.loads(line)

    lines.append((i, js))

  return lines

# trainNLs filter is a list of NLs that we dont want in the valid or test set
def processFiles(fname: str, prefix: str, trunc: int, trainNls_filter, orig=False):
  lines = loadFile(fname, trunc)
  print('Loaded {} lines from {}'.format(len(lines), fname))

  compressed_examples : List[Tuple[Dict, collections.Counter]] = []

  start = time.time()
  with Pool(processes=opt.threads) as pool:
    compressed_examples = pool.starmap(compressExample, product(lines, [trainNls_filter], [orig]))
  end = time.time()
  print('Finished compressing {} examples. Took {} seconds'.format(len(compressed_examples), end - start))

  # We do this because its hard to raise exceptions when running in multiple threads
  for e, i, s, comp, tot in compressed_examples:
    if s == ProcessError.ERROR_EXAMPLE:
      raise ValueError('Error in processing examples. Please investigate.')

  start = time.time()
  # Aggregate all the idiom counters
  full_idioms_applied = collections.Counter()
  compression_stat = 0
  total_before_compression = 0
  for e, i, s, comp, tot in compressed_examples:
    full_idioms_applied += i
    compression_stat += comp
    total_before_compression += tot
  end = time.time()
  print('Took {} seconds to aggregate idioms applied'.format(end - start))
  print('Total greedy compression: {} from {}'.format(compression_stat, total_before_compression))

  f = open(prefix + '.idioms_applied', 'w')
  for idiom in full_idioms_applied:
    f.write(idiom.tostring() + ' ' + str(full_idioms_applied[idiom]) + ' times\n')
  f.close()

  f = open(prefix + '.dataset', 'w')
  # For the traiing set, filter the bad examples. Leave it for the other two sets
  f.write(json.dumps([r for e, i, s, comp, tot in compressed_examples for r in e if s == ProcessError.GOOD], indent=4))
  f.close()

  NLs = [' '.join(r['nl']) for x in compressed_examples for r in x[0]]
  return NLs

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='build.py')

  parser.add_argument('-train_file', required=True,
                      help="Path to the training source data")
  parser.add_argument('-valid_file', required=True,
                      help="Path to the validation source data")
  parser.add_argument('-test_file', required=True,
                      help="Path to the test source data")
  parser.add_argument('-train_num', type=int, default=100000,
                      help="No. of Training examples")
  parser.add_argument('-valid_num', type=int, default=2000,
                      help="No. of Validation examples")
  parser.add_argument('-max_idioms_to_load', type=int, default=50,
                      help="No. of Validation examples")
  parser.add_argument('-threads', type=int, default=1,
                      help="No. of Validation examples")
  parser.add_argument('-dataset', type=str, required=True,
                      help="No. of Validation examples")
  parser.add_argument('-get_idioms', action='store_true',
                      help="No. of Validation examples")
  parser.add_argument('-idiom_folder', type=str,
                      help="No. of Validation examples")
  parser.add_argument('-idioms', type=str, default='',
                      help="No. of Validation examples")
  parser.add_argument('-bpe_steps', type=int, default=0,
                      help="No. of Validation examples")
  parser.add_argument('-color', action='store_true',
                      help="No. of Validation examples")
  parser.add_argument('-bpe', action='store_true',
                      help="No. of Validation examples")
  parser.add_argument('-output_folder',
                      help="Only when applying idioms")
  opt = parser.parse_args()
  print(opt)


  if opt.dataset == "concode":
    ThisProcessor = ConcodeProcessor
  elif opt.dataset == "sql":
    ThisProcessor = AtisSqlProcessor

  # First pass to get the idioms
  if opt.get_idioms:
    try:
      os.makedirs(opt.idiom_folder)
    except:
      pass

    print('Starting to get idioms')
    lines = loadFile(opt.train_file, opt.train_num)
    # passing file here, so that we can write the idioms one at a time as they are being generated
    idioms, dataset_trees = bpe(lines, opt.bpe_steps, opt.idiom_folder + "/idioms0.json")

  else:
    # Apply the idioms
    try:
      os.makedirs(opt.output_folder)
    except:
      pass

    print('Loading idioms from combined file')
    idioms_loaded = []
    for idiom in open(opt.idioms):
      if len(idioms_loaded) == opt.max_idioms_to_load:
        break
      tnt = treeFromJSON(json.loads(idiom))
      idioms_loaded.append(tnt)
    print('Total number of idioms loaded = {}'.format(len(idioms_loaded)))


    # Second pass to use the idioms

    # First we load valid, then we make sure train doesn't have any of those NLs
    validNLs = processFiles(opt.valid_file, opt.output_folder + '/valid', opt.valid_num, [])
    testNLs = processFiles(opt.test_file, opt.output_folder + '/test', opt.valid_num, [])

    trainNLs = processFiles(opt.train_file, opt.output_folder + '/train', opt.train_num, validNLs + testNLs)

    # Create a new file called predict.dataset
    # Clear idioms loaded, so that we don't get duplicated in predict and test
    idioms_loaded = {}
    # when orig=True, the original tree is used without any idioms applied
    processFiles(opt.valid_file, opt.output_folder + '/predict', opt.valid_num, [], orig=True)
    processFiles(opt.test_file, opt.output_folder + '/test', opt.valid_num, [], orig=True)
