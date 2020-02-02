import sys

def order(toks, pred): # pred is and or or
  children = []
  i = 0
  bracketStart = 0
  bracketEnd = 0
  predFound = False
  predStart = 0

  while i < len(toks):
    if toks[i] == pred and not predFound:
      predFound = True
      predStart = i + 1
      brackets = 0
    if predFound and toks[i] == '(':
      if brackets == 0:
        bracketStart = i
      brackets += 1
    if predFound and toks[i] == ')':
      brackets -= 1
      if brackets == 0:
        bracketEnd = i
        children.append(order(toks[bracketStart:i + 1], pred))
      elif brackets < 0:
        children.sort(key=lambda x: ' '.join(x))
        toks[predStart: bracketEnd + 1] = [y for x in children for y in x]
        children = []
        predFound = False

    i += 1

  return toks

if __name__ == '__main__':

  gold = sys.argv[1]

  f = open(gold, 'r')

  exact = 0
  total = 0

  for line in sys.stdin:
    g = f.readline()
    if order(g.strip().split(), 'and') == order(line.strip().split(), 'and'):
      exact +=1
    total += 1

  print(str(exact * 100.0/total))
