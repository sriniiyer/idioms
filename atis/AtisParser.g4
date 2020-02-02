logicalForm_NT
  : LPAREN expression_NT RPAREN | argument_NT | constant_NT
  ;

variable_NT: VARIABLE;
argument_NT: ARGUMENT;
constant_NT: CONSTANT;
predicate_NT: PREDICATE;

expression_NT
  : LAMBDA variable_NT E logicalForm_NT
    | LAMBDA variable_NT I logicalForm_NT
    | ARGMAX variable_NT logicalForm_NT logicalForm_NT
    | ARGMIN variable_NT logicalForm_NT logicalForm_NT
    | COUNT variable_NT logicalForm_NT
    | EXISTS variable_NT logicalForm_NT
    | MAX variable_NT logicalForm_NT
    | MIN variable_NT logicalForm_NT
    | EQUALS (variable_NT | argument_NT | logicalForm_NT) (variable_NT | argument_NT | logicalForm_NT)
    | EQUALTO (variable_NT | argument_NT | logicalForm_NT) (variable_NT | argument_NT | logicalForm_NT)
    | GREATER logicalForm_NT ( logicalForm_NT | constant_NT )
    | LESSER logicalForm_NT ( logicalForm_NT | constant_NT )
    | AND logicalForm_NT +
    | SUM variable_NT logicalForm_NT logicalForm_NT
    | OR logicalForm_NT +
    | NOT logicalForm_NT
    | THE variable_NT logicalForm_NT
    | predicate_NT (variable_NT | logicalForm_NT | argument_NT | constant_NT) +
  ;
