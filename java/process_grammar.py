import argparse

parser = argparse.ArgumentParser(description='process_grammar.py')

parser.add_argument('-grammar', type=str, required=True,
                    help="Path to the training source data")
parser.add_argument('-new_grammar', type=str, required=True,
                    help="Path to the validation source data")
parser.add_argument('-lexer', type=str, required=True,
                    help="Path to the test source data")
parser.add_argument('-expand_stars_fully', action='store_true',
                    help="Path to the test source data")

opt = parser.parse_args()
print(opt)
# Read antlr rules
# Make sure that there are no comments in the file before running this script

rules = []

curr = ""
for line in open(opt.grammar + '.g4', 'r'):
  line = line.strip()
  curr += ' ' + line
  if line != '' and line[-1] == ';': # semi colon is also a terminal
    if ':' in curr: # this is safe coz all colons are in valid rules
      curr = curr.strip()
      curr = curr[:-1]
      rules.append(curr)
    curr = ""

new_non_terminal = 0


# Repeat until there are no more parentheses

while True:
  found = False

  new_rules = []
  # remove parentheses from rules
  for r in range(0, len(rules)):
    rule = rules[r]

    closing = -1
    for i in range(0, len(rule)):
      if rule[i] == '(' and rule[i-1] != "'": # coz '('
        found = True
        total = 1
        #find closing
        for j in range(i + 1, len(rule)):
          if rule[j] == '(' and rule[j-1] != "'": # some brackets are in quotes
            total += 1
          elif rule[j] == ')' and rule[j-1] != "'":
            total -=1
          if total == 0:
            closing = j
            break

        parenRule = rule[i+1:closing] # extract paren withtout ()
        # add new rule
        new_rules.append('nt_' + str(new_non_terminal) + '_NT : ' + parenRule)
        # add modified old rule

        new_rules.append(rule[:i] + ' nt_' + str(new_non_terminal) + '_NT ' + rule[closing + 1:])
        new_non_terminal += 1
        break
    if closing == -1: # no paren was found
      new_rules.append(rule)

  rules = new_rules

  if not found:
    break

# Hooray, no more parentheses
# Now get rid of ?
# Also treat * as a possible ?

def getOptions(rule):
  options = rhs.replace("'|'", "concode").split('|') # we cannot do this
  return [x.replace("concode", "'|'") for x in options]

for r  in range(0, len(rules)):
  rule = rules[r]

  (lhs, rhs) = rule.split(':', 1) # coz of ':'
  options = getOptions(rhs) # take care of '|'

  while True: # Do it until no ? is found
    found = False
    new_options = []
    for option in options:
      words = option.split()
      flag = False
      for i in range(0, len(words)):
        if words[i-1] != "'" and (words[i] == "?" or words[i] == "*"):  # coz of '*' and '?'
          new_options.append(' '.join(words[:i - 1] + words[i+1:]))
          if words[i] == "?":
            new_options.append(' '.join(words[:i] + words[i+1:]))
          else: # we can change the star to a +
            new_options.append(' '.join(words[:i] + ['+'] + words[i+1:]))
          flag = True
          found = True
          break
      if flag == False:
        new_options.append(option)

    options = new_options
    if not found:
      break

  rules[r] = lhs + ': ' + '|'.join(options) 

# Hooray, no more ?
# Now remove *
star_non_terminal = 0
star_nt_map = {}
new_rules = []

for r  in range(0, len(rules)):
  rule = rules[r]

  (lhs, rhs) = rule.split(':', 1)
  options = getOptions(rhs)

  while True:
    found = False
    new_options = []
    for option in options:
      words = option.split()
      flag = False
      for i in range(0, len(words)):
        if (words[i] == "+") and words[i-1] != "'": # Coz of '*'
          # create new non terminal
          if words[i - 1] in star_nt_map:
            star_nt = star_nt_map[words[i-1]]
          else:

            star_nt = 'star_' + str(star_non_terminal) + '_NT'
            star_nt_map[words[i-1]] = star_nt
            star_non_terminal += 1

            if opt.expand_stars_fully:
              new_rules.append('{} : {} | {} {} | {} {} {} | {} {} {} {} | {} {} {} {} {} | {} {} {} {} {} {} | {} {} {} {} {} {} {} | {} {} {} {} {} {} {} {} '.format(star_nt, *([words[i - 1]] * 36)))
            else:
              new_rules.append(star_nt + ' : ' + words[i - 1] + ' | ' + star_nt + ' ' + star_nt)
          new_options.append(' '.join(words[:i - 1] + [star_nt] + words[i+1:]))
          flag = True
          found = True
          break
      if flag == False:
        new_options.append(option)

    options = new_options
    if not found:
      break

  rules[r] = lhs + ': ' + '|'.join(options) 


rules = rules + new_rules

g = open(opt.new_grammar + '.g4', 'w')
g.write('parser grammar {};\n'.format(opt.new_grammar))
g.write('options { tokenVocab=' + opt.lexer + ';  }\n')

for rule in rules:
  g.write(rule + ';\n')

g.close()
