import sys
import re
import argparse

parser = argparse.ArgumentParser(description='exact_sql.py')

parser.add_argument('-gold', type=str, required=True,
                    help='Path to model .pt file')
parser.add_argument('-pred', type=str, required=True,
                    help='Path to model .pt file')
parser.add_argument('-nl', type=str, required=True,
                    help='Path to model .pt file')
opt = parser.parse_args()

f = open(opt.gold, 'r')
g = open(opt.pred, 'r')

exact = 0
total = 0

for (gold, pred) in zip(f, g):
  if gold.strip() == re.sub("% (.*?) %", r"%\1%", pred.strip(), flags=re.DOTALL):
    exact +=1
  total += 1

print(str(exact * 100.0/total))
