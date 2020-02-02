import json
import argparse

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-file', required=True, type=str)
opt = parser.parse_args()

js  = json.loads(open(opt.file, 'r').read())
for j in js:
  print(' '.join(j["nl"]))
