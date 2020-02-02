parser grammar AtisParserModified;
options { tokenVocab=AtisLexer;  }
logicalForm_NT :   LPAREN expression_NT RPAREN | argument_NT | constant_NT ;
variable_NT:   VARIABLE;
argument_NT:   ARGUMENT;
constant_NT:   CONSTANT;
predicate_NT:   PREDICATE;
nt_0_NT :   variable_NT | argument_NT | logicalForm_NT;
nt_1_NT :   variable_NT | argument_NT | logicalForm_NT;
nt_2_NT :   variable_NT | argument_NT | logicalForm_NT;
nt_3_NT :   variable_NT | argument_NT | logicalForm_NT;
nt_4_NT :    logicalForm_NT | constant_NT ;
nt_5_NT :    logicalForm_NT | constant_NT ;
nt_6_NT :   variable_NT | logicalForm_NT | argument_NT | constant_NT;
expression_NT :   LAMBDA variable_NT E logicalForm_NT | LAMBDA variable_NT I logicalForm_NT | ARGMAX variable_NT logicalForm_NT logicalForm_NT | ARGMIN variable_NT logicalForm_NT logicalForm_NT | COUNT variable_NT logicalForm_NT | EXISTS variable_NT logicalForm_NT | MAX variable_NT logicalForm_NT | MIN variable_NT logicalForm_NT | EQUALS  nt_0_NT   nt_1_NT  | EQUALTO  nt_2_NT   nt_3_NT  | GREATER logicalForm_NT  nt_4_NT  | LESSER logicalForm_NT  nt_5_NT  |AND star_0_NT| SUM variable_NT logicalForm_NT logicalForm_NT |OR star_0_NT| NOT logicalForm_NT | THE variable_NT logicalForm_NT |predicate_NT star_1_NT;
star_0_NT : logicalForm_NT | logicalForm_NT logicalForm_NT | logicalForm_NT logicalForm_NT logicalForm_NT | logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT | logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT | logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT | logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT | logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT logicalForm_NT ;
star_1_NT : nt_6_NT | nt_6_NT nt_6_NT | nt_6_NT nt_6_NT nt_6_NT | nt_6_NT nt_6_NT nt_6_NT nt_6_NT | nt_6_NT nt_6_NT nt_6_NT nt_6_NT nt_6_NT | nt_6_NT nt_6_NT nt_6_NT nt_6_NT nt_6_NT nt_6_NT | nt_6_NT nt_6_NT nt_6_NT nt_6_NT nt_6_NT nt_6_NT nt_6_NT | nt_6_NT nt_6_NT nt_6_NT nt_6_NT nt_6_NT nt_6_NT nt_6_NT nt_6_NT ;
