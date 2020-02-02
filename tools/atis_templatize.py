import argparse
import sys
import json
from pattern.en import singularize
import re
import shlex

def tokenizeNL(s):
  s = re.sub('([,!?()])', r' \1 ', s)
  s = re.sub("'([a-zA-Z]+)\s", r" '\1 ", s) 
  s = re.sub("\.$", " .", s) 
  return s.split()

def tokenizeQuery(q):
  q = re.sub('([\\\,!?;+*<>()=/\-])', r' \1 ', q)
  try:
    tokens = shlex.split(q, posix=False)
  except:
    tokens = q.split()
  return tokens

tables = ['aircraft', 'airline', 'airport', 'city', 'days', 'fare', 'flight', 'month', 'restriction', 'state']

def normalizeTables(words):
  for i in range(0, len(words)):
    singular = singularize(words[i].lower())
    if singular in tables:
      words[i] = singular

def process(nl, sql, alignments):

  mapping = {}
  invMapping = {}
  nl = ' ' + nl + ' '
  for (src, tgt) in alignments:
    nl = re.sub(' ' + src + ' ', ' ' + tgt + ' ', nl)

  tokens = tokenizeQuery(sql)
  typNum = {}


  for i in range(0, len(tokens)):
    if tokens[i][0] == "'" or tokens[i][0] == '"':
      try:
        if tokens[i - 1] == "=" or tokens[i-1] == "LIKE":
          (tab, col) = tokens[i - 2].split('.')
        else:
          (tab, col) = tokens[i - 3].split('.')
      except:
        import pdb
        pdb.set_trace()
      if col in ["compartment", "meal_description", "city_name", "state_name", "class_type", "airline_flight", "aircraft_code", "aircraft_description", "basic_type","airport_code", "airport_name", "airline_name", "transport_type"]:

        typ = col.upper().replace('_', '')

        constant = tokens[i][1:-1].lower()
        mapping[constant] = typ

  # now process nl
  words = tokenizeNL(nl)
  i = 0
  usedMapping = {}
  usedInvMapping = {} 

  while i < len(words):
    if i < len(words) - 2 and (words[i] + ' ' + words[i+1] + ' ' + words[i + 2]) in mapping: 
      phrase = words[i] + ' ' + words[i+1] + ' ' + words[i + 2]
      typ = ""
      if phrase in usedMapping:
        typ = usedMapping[phrase]
      else:
        typ = mapping[phrase]
        typNum[typ] = typNum.setdefault(typ, -1) + 1
        typ = typ + '@' + str(typNum[typ])
        usedMapping[phrase] = typ
        usedInvMapping[typ] = "'" + phrase + "'"

      words[i] = typ
      words[i + 1] = ''
      words[i + 2] = ''
      i += 2
    elif i < len(words) - 1 and (words[i] + ' ' + words[i+1]) in mapping:
      typ = ""
      phrase = words[i] + ' ' + words[i+1]
      if phrase in usedMapping:
        typ = usedMapping[phrase]
      else:
        typ = mapping[phrase]
        typNum[typ] = typNum.setdefault(typ, -1) + 1
        typ = typ + '@' + str(typNum[typ])
        usedMapping[phrase] = typ
        usedInvMapping[typ] = "'" + phrase + "'"

      words[i] = typ
      words[i + 1] = ''
      i += 1

    elif words[i] in mapping:
      typ = ""
      phrase = words[i]
      if phrase in usedMapping:
        typ = usedMapping[phrase]
      else:
        typ = mapping[phrase]
        typNum[typ] = typNum.setdefault(typ, -1) + 1
        typ = typ + '@' + str(typNum[typ])
        usedMapping[phrase] = typ
        usedInvMapping[typ] = "'" + phrase + "'"

      words[i] = typ

    i += 1

  # substitute usedMapping in SQL
  for i in range(0, len(tokens)):
    if tokens[i][1:-1].lower() in usedMapping:
      tokens[i] = usedMapping[tokens[i][1:-1].lower()]

  normalizeTables(words)
  return(' '.join(words), ' '.join(tokens), usedInvMapping)

def deanonymize(mp, sql):
  tokens = tokenizeQuery(sql)
  invMapping = json.loads(mp)
  for i in range(0, len(tokens)):
    if tokens[i] in invMapping:
      tokens[i] = invMapping[tokens[i]]
  return ' '.join(tokens)


def main():
  parser = argparse.ArgumentParser(description='translate.py')

  parser.add_argument('-nlfile', type=str)
  parser.add_argument('-sqlfile', type=str)
  parser.add_argument('-mapfile', type=str)
  parser.add_argument('-alignments', type=str)
  parser.add_argument('-inst', type=str)
  opt = parser.parse_args()

  alignments = []
  for line in open(opt.alignments, 'r'):
    (src, tgt) = line.strip().split('\t')
    alignments.append((src, tgt))


  if opt.inst == "templatize":
    fnl = open(opt.nlfile + '.tem', 'w')
    mnl = open(opt.nlfile + '.tem.map', 'w')
    fsql = open(opt.sqlfile + '.tem', 'w')

    for (nl, sql) in zip(open(opt.nlfile, 'r'), open(opt.sqlfile, 'r')):
      (t_nl, t_sql, m_nl) = process(nl.strip(), sql.strip(), alignments)
      fnl.write(t_nl + '\n')
      mnl.write(json.dumps(m_nl) + '\n')
      fsql.write(t_sql + '\n')

    fnl.close()
    fsql.close()
  elif opt.inst == "deanonymize":
    fsql = open(opt.sqlfile + '.deanon', 'w')

    for (mp, sql) in zip(open(opt.mapfile, 'r'), open(opt.sqlfile, 'r')):
      deanon = deanonymize(mp.strip(), sql.strip())
      fsql.write(deanon + '\n')
    fsql.close()

if __name__ == '__main__':
  main()
