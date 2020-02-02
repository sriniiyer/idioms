import json
from nltk.stem import PorterStemmer
import os
import random
porter=PorterStemmer()

# This file creates random subsets of atis train for learning curve tests

files = ['train', 'valid', 'test']

train = []

for f in files:
  g = open(f + '.json', 'w')
  for nl, sql in  zip(open(f + '.nl.tem'), open(f + '.sql.tem')):
    nl_stemmed = ' '.join([porter.stem(x) for x in nl.split()])
    g.write(json.dumps({'nl': nl_stemmed.strip(), 'code': sql.strip().replace('@', '_').replace('.', '.')}) + '\n')
    if f == 'train':
      train.append((nl_stemmed.strip(), sql.strip().replace('@', '_').replace('.', '.')))
  g.close()

random.seed(1123)
random.shuffle(train)
for i in range(20, 101, 20):
  foldername = '../atis_part/' + str(i)
  try:
    os.mkdir(foldername)
  except:
    pass
  g = open(foldername + '/train.json', 'w')
  for j in range(0, int((i * len(train)) // 100)):
    g.write(json.dumps({'nl': train[j][0], 'code': train[j][1]}) + '\n')
  g.close()
  os.system('cp valid.json ' + foldername + '/')
  os.system('cp test.json ' + foldername + '/')
  idiom_command = 'python ../build.py -train_file {}/train.json  -valid_file {}/valid.json -test_file {}/test.json  -train_num 100000 -valid_num 2000 -dataset sql -get_idioms -idiom_folder {}/ -bpe_steps 400 -threads 10 > {}/idioms.print'
  os.system(idiom_command.format(foldername, foldername, foldername, foldername, foldername))
