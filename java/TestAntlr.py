import antlr4
from JavaLexer import JavaLexer
from JavaParserModified import JavaParserModified

if __name__ == '__main__':
  code = open('AllInOne8.java', 'r').read()

  stream = antlr4.InputStream(code)
  lexer = JavaLexer(stream)
  toks = antlr4.CommonTokenStream(lexer)
  parser = JavaParserModified(toks)

  tree = parser.memberDeclaration_NT()
  print ("Tree " + tree.toStringTree(recog=parser))
