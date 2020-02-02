import subprocess
import sys
import re
import argparse

parser = argparse.ArgumentParser(description='bleu.py')

parser.add_argument('-gold', type=str, required=True,
                    help='Path to model .pt file')
parser.add_argument('-pred', type=str, required=True,
                    help='Path to model .pt file')
parser.add_argument('-nl', type=str, required=True,
                    help='Path to model .pt file')
opt = parser.parse_args()

p = open(opt.pred, 'r')

p = subprocess.Popen(['perl', 'tools/multi-bleu.perl', '-lc'] + [opt.gold] + ['--pred'], stdin=p, stdout=subprocess.PIPE)
score = p.stdout.read()
print(str(score.strip().decode('ascii')))
