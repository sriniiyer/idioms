java -jar ../java/antlr-4.6-complete.jar -Dlanguage=Python3 ./AtisLexer.g4
python ../java/process_grammar.py -grammar AtisParser -new_grammar AtisParserModified -lexer AtisLexer -expand_stars_fully
java -jar ../java/antlr-4.6-complete.jar -Dlanguage=Python3 ./AtisParserModified.g4
