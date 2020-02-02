import json
import itertools
from copy import deepcopy
import collections

def print_tree(current_node,  indent="", last='updown'):
    tree_str = ""

    nb_children = lambda node: sum(nb_children(child) for child in node.children) + 1 
    size_branch = {child: nb_children(child) for child in current_node.children}

    """ Creation of balanced lists for "up" branch and "down" branch. """
    up = sorted(current_node.children, key=lambda node: nb_children(node))
    down = []
    while up and sum(size_branch[node] for node in down) < sum(size_branch[node] for node in up):
        down.append(up.pop())

    """ Printing of "up" branch. """
    for child in up:
        next_last = 'up' if up.index(child) is 0 else ''
        next_indent = '{0}{1}{2}'.format(indent, ' ' if 'up' in last else '│', " " * len(current_node.name))
        tree_str += print_tree(child, indent=next_indent, last=next_last)

    """ Printing of current node. """
    if last == 'up': start_shape = '┌'
    elif last == 'down': start_shape = '└'
    elif last == 'updown': start_shape = ' '
    else: start_shape = '├'

    if up: end_shape = '┤'
    elif down: end_shape = '┐'
    else: end_shape = ''

    tree_str += ('{0}{1}{2}{3}'.format(indent, start_shape, current_node.name, end_shape)) + '\n'

    """ Printing of "down" branch. """
    for child in down:
        next_last = 'down' if down.index(child) is len(down) - 1 else ''
        next_indent = '{0}{1}{2}'.format(indent, ' ' if 'down' in last else '│', " " * len(current_node.name))
        tree_str += print_tree(child, indent=next_indent, last=next_last)

    return tree_str

class TNTTree:
  idiom_terminal = 'concode_idiom'
  pre_terminal_symbols = [
    'Argument_NT', 'Predicate_NT', 'Constant_NT', 'Identifier_NT', 'Nt_char_literal_NT', 'Nt_string_literal_NT', 'Nt_bool_literal_NT',
      'Nt_null_literal_NT', 'Nt_decimal_literal_NT', 'Nt_hex_literal_NT', 'Nt_oct_literal_NT', 'Nt_binary_literal_NT', 'Nt_float_literal_NT',
      'Nt_hex_float_literal_NT', 'Placeholder_NT', 'Name_NT']
  def __init__(self, name, typ):
    self.name = name
    self.typ = typ
    self.children = []
    self.idioms = []
    self.parent = None
    self.idiom_number = []

  def __str__(self):
    return print_tree(self)

  def tostring(self):
    return json.dumps(self.json())

  def __hash__(self):
    return hash(json.dumps(self.json()))

  def __eq__(self, other):
    return self.json() == other.json()

  def json(self):
    return {'name': self.name, 'idiom_number' : self.idiom_number, 'children': [c.json() for c in self.children] }

  def to_sorted_string(self):
    return json.dumps(self.sorted_json())

  @staticmethod
  def leaves_from_json(js):
    add = []
    if js['type'] == "T":
      add = [js['name']]

    return add + [x for c in js['children'] for x in TNTTree.leaves_from_json(c)]

  def sorted_json(self):
    if len(self.children) > 0 and self.children[0].name == "and":
      import ipdb; ipdb.set_trace()
    children = [c.sorted_json() for c in self.children]
    children.sort(key=lambda x: ' '.join(TNTTree.leaves_from_json(x)))
    return {'name': self.name, 'idiom_number' : self.idiom_number, 'type': self.typ, 'children': children}

  def nodeCount(self):
    return 1  + sum(c.nodeCount() for c in self.children)

  def vertices(self):
    if self.name == TNTTree.idiom_terminal:
      return []
    return [self] + [x for c in self.children for x in c.vertices()]

  def leaves(self):
    if len(self.children) == 0:
      return [self.name]
    else:
      return [x for c in self.children for x in c.leaves()]

  def clone(self):
    children = [c.clone() for c in self.children]
    root = deepcopy(self)
    root.children = children
    root.idioms = self.idioms
    for c in children:
      c.parent = root
    return root

  def __lt__(self, other):
    return self.nodeCount() < other.nodeCount()

  def check(self, idiom): # Check if the root follows the idiom
    if self.name != idiom.name or (len(idiom.children) > 0 and len(self.children) != len(idiom.children)):
      return False
    for i in range(0, len(idiom.children)):
      if not self.children[i].check(idiom.children[i]):
        return False
    return True

  def replace(self, idiom, frontier, color):
    if color: # Never pass this on. Only append once.
      frontier.append(TNTTree(TNTTree.idiom_terminal, 'T'))
    if len(idiom.children) == 0:
      frontier.append(self)
    else:
      for i in range(0, len(idiom.children)):
        self.children[i].replace(idiom.children[i], frontier, False)

  def applyAllIdioms(self, idioms, idioms_applied):
    tree = self
    for idx, idiom in enumerate(idioms):
      tree, _ = applyIdiom(tree, idiom, idioms_applied)
    return tree

# Get the largest idiom that checks out, replace it, repeat
def applyIdiom(root, idiom, idioms_applied, start_at=-1):
  vertices = root.vertices() # vertices in inorder
  num_applied = 0

  for i, v in enumerate(vertices): # Also get the vertex index
    if i < start_at:
      continue
    if v.typ == "NT" and v.check(idiom):
      frontier = []
      vertices[i].replace(idiom, frontier, False) # use the vertex at the same location idiomVertex in the clone
      vertices[i].children = frontier
      idioms_applied[idiom] += 1

      root, num_applied_r = applyIdiom(root, idiom, idioms_applied, i)
      num_applied = 1 + num_applied_r
      break

  return root, num_applied

def treeFromJSON(js) -> TNTTree:
  node = TNTTree(js['name'], None)
  node.idiom_number = js['idiom_number']
  for c in js['children']:
    node.children.append(treeFromJSON(c))
  return node

def sqlparseToTNTTree(tree):
  typ = 'NT' if 'tokens' in tree.__dir__() else 'T'

  name = (tree.getValue() if tree.getValue() != "Identifier" else "SqlIdentifier")  + ('_NT' if typ == 'NT' else '')
  tnt = TNTTree(name, typ)
  if typ == 'NT':
    for c in tree.tokens:
      child = sqlparseToTNTTree(c)
      child.parent = tnt
      if child.name != 'Whitespace_NT':
        tnt.children.append(child)
    if tnt.name == "SqlIdentifier_NT" and len(tnt.children) == 3 and [x.name for x in tnt.children] == ['Name_NT', 'Punctuation_NT', 'Name_NT']:
      tnt.children = [TNTTree(''.join(x.children[0].name for x in tnt.children), 'T')]
  elif typ == 'T':
    tnt_par = TNTTree(str(tree.ttype).split('.')[-1] + '_NT', 'NT')
    tnt_par.children.append(tnt)
    tnt = tnt_par

  return tnt

def convert_to_TNTTree(tree):
  (name, typ) = nname(tree)
  tnt = TNTTree(name, typ)
  for c in range(0, tree.getChildCount()):
    child = convert_to_TNTTree(tree.getChild(c))
    child.parent = tnt
    tnt.children.append(child)
  return tnt

def nname(node):
  if "TerminalNodeImpl" in str(node.__class__):
    return (node.getText(), "T")
  else:
    return (str(node.__class__).split('.')[-1][:-9], "NT")

def getAllSubtreesOfDepth2(root, dataset):
  strees = getAllSubtreesOfDepth2AtRoot(root, dataset)
  for c in root.children:
    if c.typ == "NT":
      strees += getAllSubtreesOfDepth2(c, dataset)
  return strees

def getAllSubtreesOfDepth2AtRoot(tree, dataset):
  nt_children_indexes = [i for i, child in enumerate(tree.children) if child.typ == "NT"]
  if len(nt_children_indexes) == 0: # No depth-2 subtrees here
    return collections.Counter()

  # Fetch all the subtrees
  subset_subtrees = {}
  nt_children_indexes_todo = []
  for i in nt_children_indexes:
    subset_subtrees[i] = TNTTree(tree.children[i].name, tree.children[i].typ) # Each NT child has only one subtree
    for sc in tree.children[i].children:
      subset_subtrees[i].children.append(TNTTree(sc.name, sc.typ)) # Each NT child has only one subtree
#     if len(tree.children[i].children) > 1:
    nt_children_indexes_todo.append(i)

  subtrees = collections.Counter()
  for i in range(1, len(nt_children_indexes_todo) + 1 if dataset == "concode" else 2): # skipped for leaves
    subsets = set(itertools.combinations(nt_children_indexes_todo, i)) # get all subsets of children.
    for s in subsets: # A particular combination of NT children
      copy_of_tree = TNTTree(tree.name, tree.typ) # We need to copy the terminals in this node. Then we will eliminate NTs and add subtrees
      for sc in tree.children:
        copy_of_tree.children.append(TNTTree(sc.name, sc.typ))
      for index in set(nt_children_indexes) - set(nt_children_indexes_todo): # we have til fill th eindexes one b one.
        copy_of_tree.children[index] = subset_subtrees[index]

      for index in s: # we have til fill th eindexes one b one.
        copy_of_tree.children[index] = subset_subtrees[index]

      copy_of_tree_json = json.dumps(copy_of_tree.json())
      subtrees[copy_of_tree_json] += 1

  return subtrees
