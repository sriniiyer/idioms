import antlr4
from java.JavaLexer import JavaLexer
from java.JavaParserModified import JavaParserModified
from Tree import TNTTree, convert_to_TNTTree, sqlparseToTNTTree
import sqlparse

def getRuleAtNode(node):
  rule = []
  for ch in node.children:
    rule.append(ch.name)
  return node.name + '-->' + '___'.join(rule)

class Processor:
  def __init__(self, js):
    self.js = js

  @staticmethod
  def getProductions(tree):

    # Run a transition based parser on it to generate the dataset
    st = []
    st.append(tree)

    rule_seq = []

    while(len(st) > 0):
      top = st.pop()
      if top.name == "ErrorN":
        return None # There is a parsing error
      if top.typ == "T": # Terminal
        pass
      else: # Non-terminal
        rule = getRuleAtNode(top)
        rule_seq.append(rule)
        # put the rule in to the buffer
        for i in range(len(top.children) - 1, -1, -1):
          st.append(top.children[i])
    return rule_seq

class AtisSqlProcessor(Processor):
  filter_list = []
  def __init__(self, js):
    super(AtisSqlProcessor, self).__init__(js)

  def getCode(self):
    return self.js['code']

  def getNL(self):
    return ' '.join(self.js['nl'])

  def getTemplate(self, idx, trainNls):
    nlToks = self.js['nl'].split()
    return {
      'nl': nlToks,
      'code': self.js['code'].split(),
      'idx': str(idx),
      'seq2seq': nlToks,
      'seq2seq_nop': nlToks
    }

  def getTree(self) -> TNTTree:
    code = self.getCode()
    parse = sqlparse.parse(code)
    self.tnt = sqlparseToTNTTree(parse[0])
    return self.tnt

  @staticmethod
  def getProductions(tree, ):
    prods = []
    if len(tree.children) == 0:
      return prods
    prods.append(tree.name + '-->' + '___'.join([c.name for c in tree.children]))
    for c in tree.children:
      prods = prods + AtisSqlProcessor.getProductions(c)
    return prods

class ConcodeProcessor(Processor):
  def __init__(self, js):
    super(ConcodeProcessor, self).__init__(js)
    self.code = self.js['renamed']
    self.codeToks = [cTok.encode('ascii', 'replace').decode().replace("\x0C", "").strip() for cTok in self.code]
    self.nlToks = ConcodeProcessor.processNlToks(self.js['nlToks'])

  def getCode(self):
    return 'class PlaceHolderClass { ' + ' '.join(self.codeToks) + ' }'

  def getTree(self) -> TNTTree:
    code = self.getCode()
    stream = antlr4.InputStream(code)
    lexer = JavaLexer(stream)
    toks = antlr4.CommonTokenStream(lexer)
    parser = JavaParserModified(toks)

    # We are always passing methods
    tree = parser.memberDeclaration_NT()
    tnt = convert_to_TNTTree(tree)
    return tnt.children[0].children[2].children[1].children[0].children[0]

  @staticmethod
  def getProductions(tree):
    return Processor.getProductions(tree)

  @staticmethod
  def processNlToks(nlToks):
    nlToks = [tok.encode('ascii', 'replace').decode().strip() for tok in nlToks \
             if tok != "-RCB-" and \
             tok != "-LCB-" and \
             tok != "-LSB-" and \
             tok != "-RSB-" and \
             tok != "-LRB-" and \
             tok != "-RRB-" and \
             tok != "@link" and \
             tok != "@code" and \
             tok != "@inheritDoc" and \
             tok.encode('ascii', 'replace').decode().strip() != '']
    return nlToks

  def getNl(self):
    return ' '.join(self.nlToks)

  def getTemplate(self, idx, trainNls):

    if len(self.nlToks) == 0 or len(self.codeToks) == 0:
      return None
    # put placeholder variables and methods
    if len(self.js["memberVariables"]) == 0:
      self.js["memberVariables"]["placeHolder"] = "PlaceHolder"
    if len(self.js["memberFunctions"]) == 0:
      self.js["memberFunctions"]["placeHolder"] = [['placeholderType']]

    # pull out methods
    methodNames, methodReturns, methodParamNames, methodParamTypes = [], [], [], []
    for methodName in self.js["memberFunctions"]:
      for methodInstance in self.js["memberFunctions"][methodName]:
        # Alwaya have a parameter
        methodNames.append(methodName)
        methodReturns.append("None" if methodInstance[0] is None else methodInstance[0]) # The first element is the return type
        if len(methodInstance) == 1:
          methodInstance += ['NoParams noParams']
        methodParamNames.append([methodInstance[p].split()[-1] for p in range(1, len(methodInstance))])
        methodParamTypes.append([' '.join(methodInstance[p].split()[:-1]).replace('final ', '') for p in range(1, len(methodInstance))])

    # Find and annotate class variables
    memberVarNames = [key.split('=')[0].encode('ascii', 'replace').decode() for key, value in self.js["memberVariables"].items()]
    memberVarTypes = [value.encode('ascii', 'replace').decode() for key, value in self.js["memberVariables"].items()]

    for t in range(0, len(self.codeToks)):
      if self.codeToks[t] in memberVarNames and (self.codeToks[t - 1] != '.' or (self.codeToks[t - 1] == '.' and self.codeToks[t - 2] == "this")):
        self.codeToks[t] = 'concodeclass_' + self.codeToks[t]
      elif self.codeToks[t] == '(' and self.codeToks[t - 1] in methodNames and (self.codeToks[t - 2] != '.' or (self.codeToks[t - 2] == '.' and self.codeToks[t - 3] == "this")):
          self.codeToks[t - 1] = 'concodefunc_' + self.codeToks[t - 1]

    if " ".join(self.nlToks) in trainNls: # This will be empty for the training set, so it will only be used for valid and test
      return None

    try:
      seq2seq = ( ' '.join(self.nlToks).lower() + ' concode_field_sep ' + \
    ' concode_elem_sep '.join([vtyp + ' ' + vnam for (vnam, vtyp) in zip(memberVarNames, memberVarTypes)]) + ' concode_field_sep ' + \
      ' concode_elem_sep '.join([mret + ' ' + mname + ' concode_func_sep ' + ' concode_func_sep '.join(mpt + ' ' + mpn for (mpt, mpn) in zip(mpts, mpns) )  for (mret, mname, mpts, mpns) in zip(methodReturns, methodNames, methodParamTypes, methodParamNames)] ) ).split()
      seq2seq_nop = ( ' '.join(self.nlToks).lower() + ' concode_field_sep ' + \
    ' concode_elem_sep '.join([vtyp + ' ' + vnam for (vnam, vtyp) in zip(memberVarNames, memberVarTypes)]) + ' concode_field_sep ' + \
    ' concode_elem_sep '.join([mret + ' ' + mname for (mret, mname) in zip(methodReturns, methodNames)])).split()
    except:
      raise ValueError('Creating seq2seq failed')

    return \
      {'nl': self.nlToks,
       'code': self.codeToks,
       'idx': str(idx),
       'varNames': memberVarNames,
       'varTypes': memberVarTypes,
       'methodNames': methodNames,
       'methodReturns': methodReturns,
       'methodParamNames': methodParamNames,
       'methodParamTypes': methodParamTypes,
       'seq2seq': seq2seq,
       'seq2seq_nop': seq2seq_nop
       }
