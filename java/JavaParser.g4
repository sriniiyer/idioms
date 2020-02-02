compilationUnit_NT
    : packageDeclaration_NT ?  importDeclaration_NT *  typeDeclaration_NT *  EOF
     ;

identifier_NT: IDENTIFIER ;

packageDeclaration_NT
    : annotation_NT *  PACKAGE qualifiedName_NT ';'
     ;

importDeclaration_NT
    : IMPORT STATIC ?  qualifiedName_NT  ( '.' '*' )  ?  ';'
     ;

typeDeclaration_NT
    : classOrInterfaceModifier_NT * 
       ( classDeclaration_NT | enumDeclaration_NT | interfaceDeclaration_NT | annotationTypeDeclaration_NT ) 
    | ';'
     ;

modifier_NT
    : classOrInterfaceModifier_NT
    | NATIVE
    | SYNCHRONIZED
    | TRANSIENT
    | VOLATILE
     ;

classOrInterfaceModifier_NT
    : annotation_NT
    | PUBLIC
    | PROTECTED
    | PRIVATE
    | STATIC
    | ABSTRACT
    | FINAL
    | STRICTFP
     ;

variableModifier_NT
    : FINAL
    | annotation_NT
     ;

classDeclaration_NT
    : CLASS identifier_NT typeParameters_NT ? 
       ( EXTENDS typeType_NT )  ? 
       ( IMPLEMENTS typeList_NT )  ? 
      classBody_NT
     ;

typeParameters_NT
    : '<' typeParameter_NT  ( ',' typeParameter_NT )  *  '>'
     ;

typeParameter_NT
    : annotation_NT *  identifier_NT  ( EXTENDS typeBound_NT )  ? 
     ;

typeBound_NT
    : typeType_NT  ( '&' typeType_NT )  * 
     ;

enumDeclaration_NT
    : ENUM identifier_NT  ( IMPLEMENTS typeList_NT )  ?  '{' enumConstants_NT ?  ',' ?  enumBodyDeclarations_NT ?  '}'
     ;

enumConstants_NT
    : enumConstant_NT  ( ',' enumConstant_NT )  * 
     ;

enumConstant_NT
    : annotation_NT *  identifier_NT arguments_NT ?  classBody_NT ? 
     ;

enumBodyDeclarations_NT
    : ';' classBodyDeclaration_NT * 
     ;

interfaceDeclaration_NT
    : INTERFACE identifier_NT typeParameters_NT ?   ( EXTENDS typeList_NT )  ?  interfaceBody_NT
     ;

classBody_NT
    : '{' classBodyDeclaration_NT *  '}'
     ;

interfaceBody_NT
    : '{' interfaceBodyDeclaration_NT *  '}'
     ;

classBodyDeclaration_NT
    : ';'
    | STATIC ?  block_NT
    | modifier_NT *  memberDeclaration_NT
     ;

memberDeclaration_NT
    : methodDeclaration_NT
    | genericMethodDeclaration_NT
    | fieldDeclaration_NT
    | constructorDeclaration_NT
    | genericConstructorDeclaration_NT
    | interfaceDeclaration_NT
    | annotationTypeDeclaration_NT
    | classDeclaration_NT
    | enumDeclaration_NT
     ;

methodDeclaration_NT
    : typeTypeOrVoid_NT identifier_NT formalParameters_NT  ( '[' ']' )  * 
       ( THROWS qualifiedNameList_NT )  ? 
      methodBody_NT
     ;

methodBody_NT
    : block_NT
    | ';'
     ;

typeTypeOrVoid_NT
    : typeType_NT
    | VOID
     ;

genericMethodDeclaration_NT
    : typeParameters_NT methodDeclaration_NT
     ;

genericConstructorDeclaration_NT
    : typeParameters_NT constructorDeclaration_NT
     ;

constructorDeclaration_NT
    : identifier_NT formalParameters_NT  ( THROWS qualifiedNameList_NT )  ?  constructorBody=block_NT
     ;

fieldDeclaration_NT
    : typeType_NT variableDeclarators_NT ';'
     ;

interfaceBodyDeclaration_NT
    : modifier_NT *  interfaceMemberDeclaration_NT
    | ';'
     ;

interfaceMemberDeclaration_NT
    : constDeclaration_NT
    | interfaceMethodDeclaration_NT
    | genericInterfaceMethodDeclaration_NT
    | interfaceDeclaration_NT
    | annotationTypeDeclaration_NT
    | classDeclaration_NT
    | enumDeclaration_NT
     ;

constDeclaration_NT
    : typeType_NT constantDeclarator_NT  ( ',' constantDeclarator_NT )  *  ';'
     ;

constantDeclarator_NT
    : identifier_NT  ( '[' ']' )  *  '=' variableInitializer_NT
     ;

interfaceMethodDeclaration_NT
    : interfaceMethodModifier_NT *   ( typeTypeOrVoid_NT | typeParameters_NT annotation_NT *  typeTypeOrVoid_NT ) 
      identifier_NT formalParameters_NT  ( '[' ']' )  *   ( THROWS qualifiedNameList_NT )  ?  methodBody_NT
     ;

interfaceMethodModifier_NT
    : annotation_NT
    | PUBLIC
    | ABSTRACT
    | DEFAULT
    | STATIC
    | STRICTFP
     ;

genericInterfaceMethodDeclaration_NT
    : typeParameters_NT interfaceMethodDeclaration_NT
     ;

variableDeclarators_NT
    : variableDeclarator_NT  ( ',' variableDeclarator_NT )  * 
     ;

variableDeclarator_NT
    : variableDeclaratorId_NT  ( '=' variableInitializer_NT )  ? 
     ;

variableDeclaratorId_NT
    : identifier_NT  ( '[' ']' )  * 
     ;

variableInitializer_NT
    : arrayInitializer_NT
    | expression_NT
     ;

arrayInitializer_NT
    : '{'  ( variableInitializer_NT  ( ',' variableInitializer_NT )  *   ( ',' )  ?   )  ?  '}'
     ;

classOrInterfaceType_NT
    : identifier_NT typeArguments_NT ?   ( '.' identifier_NT typeArguments_NT ?  )  * 
     ;

typeArgument_NT
    : typeType_NT
    | '?'  (  ( EXTENDS | SUPER )  typeType_NT )  ? 
     ;

qualifiedNameList_NT
    : qualifiedName_NT  ( ',' qualifiedName_NT )  * 
     ;

formalParameters_NT
    : '(' formalParameterList_NT ?  ')'
     ;

formalParameterList_NT
    : formalParameter_NT  ( ',' formalParameter_NT )  *   ( ',' lastFormalParameter_NT )  ? 
    | lastFormalParameter_NT
     ;

formalParameter_NT
    : variableModifier_NT *  typeType_NT variableDeclaratorId_NT
     ;

lastFormalParameter_NT
    : variableModifier_NT *  typeType_NT '...' variableDeclaratorId_NT
     ;

qualifiedName_NT
    : identifier_NT  ( '.' identifier_NT )  * 
     ;

literal_NT
    : integerLiteral_NT
    | floatLiteral_NT
    | nt_char_literal_NT
    | nt_string_literal_NT
    | nt_bool_literal_NT
    | nt_null_literal_NT
     ;

nt_char_literal_NT: CHAR_LITERAL;
nt_string_literal_NT: STRING_LITERAL;
nt_bool_literal_NT: BOOL_LITERAL;
nt_null_literal_NT: NULL_LITERAL;
nt_decimal_literal_NT: DECIMAL_LITERAL;
nt_hex_literal_NT: HEX_LITERAL;
nt_oct_literal_NT: OCT_LITERAL;
nt_binary_literal_NT: BINARY_LITERAL;
nt_float_literal_NT: FLOAT_LITERAL;
nt_hex_float_literal_NT: HEX_FLOAT_LITERAL;

integerLiteral_NT
    : nt_decimal_literal_NT
    | nt_hex_literal_NT
    | nt_oct_literal_NT
    | nt_binary_literal_NT
     ;

floatLiteral_NT
    : nt_float_literal_NT
    | nt_hex_float_literal_NT
     ;

annotation_NT
    : '@' qualifiedName_NT  ( '('  (  elementValuePairs_NT | elementValue_NT  )  ?  ')' )  ? 
     ;

elementValuePairs_NT
    : elementValuePair_NT  ( ',' elementValuePair_NT )  * 
     ;

elementValuePair_NT
    : identifier_NT '=' elementValue_NT
     ;

elementValue_NT
    : expression_NT
    | annotation_NT
    | elementValueArrayInitializer_NT
     ;

elementValueArrayInitializer_NT
    : '{'  ( elementValue_NT  ( ',' elementValue_NT )  *  )  ?   ( ',' )  ?  '}'
     ;

annotationTypeDeclaration_NT
    : '@' INTERFACE identifier_NT annotationTypeBody_NT
     ;

annotationTypeBody_NT
    : '{'  ( annotationTypeElementDeclaration_NT )  *  '}'
     ;

annotationTypeElementDeclaration_NT
    : modifier_NT *  annotationTypeElementRest_NT
    | ';' 
     ;

annotationTypeElementRest_NT
    : typeType_NT annotationMethodOrConstantRest_NT ';'
    | classDeclaration_NT ';' ? 
    | interfaceDeclaration_NT ';' ? 
    | enumDeclaration_NT ';' ? 
    | annotationTypeDeclaration_NT ';' ? 
     ;

annotationMethodOrConstantRest_NT
    : annotationMethodRest_NT
    | annotationConstantRest_NT
     ;

annotationMethodRest_NT
    : identifier_NT '(' ')' defaultValue_NT ? 
     ;

annotationConstantRest_NT
    : variableDeclarators_NT
     ;

defaultValue_NT
    : DEFAULT elementValue_NT
     ;

block_NT
    : '{' blockStatement_NT *  '}'
     ;

blockStatement_NT
    : localVariableDeclaration_NT ';'
    | statement_NT
    | localTypeDeclaration_NT
     ;

localVariableDeclaration_NT
    : variableModifier_NT *  typeType_NT variableDeclarators_NT
     ;

localTypeDeclaration_NT
    : classOrInterfaceModifier_NT * 
       ( classDeclaration_NT | interfaceDeclaration_NT ) 
    | ';'
     ;

statement_NT
    : blockLabel=block_NT
    | ASSERT expression_NT  ( ':' expression_NT )  ?  ';'
    | IF parExpression_NT statement_NT  ( ELSE statement_NT )  ? 
    | FOR '(' forControl_NT ')' statement_NT
    | WHILE parExpression_NT statement_NT
    | DO statement_NT WHILE parExpression_NT ';'
    | TRY block_NT  ( catchClause_NT + finallyBlock_NT ?  | finallyBlock_NT ) 
    | TRY resourceSpecification_NT block_NT catchClause_NT *  finallyBlock_NT ? 
    | SWITCH parExpression_NT '{' switchBlockStatementGroup_NT *  switchLabel_NT *  '}'
    | SYNCHRONIZED parExpression_NT block_NT
    | RETURN expression_NT ?  ';'
    | THROW expression_NT ';'
    | BREAK identifier_NT ?  ';'
    | CONTINUE identifier_NT ?  ';'
    | SEMI
    | statementExpression=expression_NT ';'
    | identifierLabel=identifier_NT ':' statement_NT
     ;

catchClause_NT
    : CATCH '(' variableModifier_NT *  catchType_NT identifier_NT ')' block_NT
     ;

catchType_NT
    : qualifiedName_NT  ( '|' qualifiedName_NT )  * 
     ;

finallyBlock_NT
    : FINALLY block_NT
     ;

resourceSpecification_NT
    : '(' resources_NT ';' ?  ')'
     ;

resources_NT
    : resource_NT  ( ';' resource_NT )  * 
     ;

resource_NT
    : variableModifier_NT *  classOrInterfaceType_NT variableDeclaratorId_NT '=' expression_NT
     ;

switchBlockStatementGroup_NT
    : switchLabel_NT + blockStatement_NT +
     ;

switchLabel_NT
    : CASE  ( constantExpression=expression_NT | enumConstantName=identifier_NT )  ':'
    | DEFAULT ':'
     ;

forControl_NT
    : enhancedForControl_NT
    | forInit_NT ?  ';' expression_NT ?  ';' forUpdate=expressionList_NT ? 
     ;

forInit_NT
    : localVariableDeclaration_NT
    | expressionList_NT
     ;

enhancedForControl_NT
    : variableModifier_NT *  typeType_NT variableDeclaratorId_NT ':' expression_NT
     ;

parExpression_NT
    : '(' expression_NT ')'
     ;

expressionList_NT
    : expression_NT  ( ',' expression_NT )  * 
     ;

expression_NT
    : primary_NT
    | expression_NT '.'
       ( identifier_NT
      | THIS
      | NEW nonWildcardTypeArguments_NT ?  innerCreator_NT
      | SUPER superSuffix_NT
      | explicitGenericInvocation_NT
       ) 
    | expression_NT '[' expression_NT ']'
    | expression_NT '(' expressionList_NT ?  ')'
    | NEW creator_NT
    | '(' typeType_NT ')' expression_NT
    | expression_NT ( '++' | '--' ) 
    |  ( '+'|'-'|'++'|'--' )  expression_NT
    |  ( '~'|'!' )  expression_NT
    | expression_NT  ( '*'|'/'|'%' )  expression_NT
    | expression_NT  ( '+'|'-' )  expression_NT
    | expression_NT  ( '<' '<' | '>' '>' '>' | '>' '>' )  expression_NT
    | expression_NT  ( '<=' | '>=' | '>' | '<' )  expression_NT
    | expression_NT INSTANCEOF typeType_NT
    | expression_NT  ( '==' | '!=' )  expression_NT
    | expression_NT '&' expression_NT
    | expression_NT '^' expression_NT
    | expression_NT '|' expression_NT
    | expression_NT '&&' expression_NT
    | expression_NT '||' expression_NT
    | expression_NT '?' expression_NT ':' expression_NT
    | <assoc=right> expression_NT
       ( '=' | '+=' | '-=' | '*=' | '/=' | '&=' | '|=' | '^=' | '>>=' | '>>>=' | '<<=' | '%=' ) 
      expression_NT
    | lambdaExpression_NT 

    | expression_NT '::' typeArguments_NT ?  identifier_NT
    | typeType_NT '::'  ( typeArguments_NT ?  identifier_NT | NEW ) 
    | classType_NT '::' typeArguments_NT ?  NEW
     ;

lambdaExpression_NT
    : lambdaParameters_NT '->' lambdaBody_NT
     ;

lambdaParameters_NT
    : identifier_NT
    | '(' formalParameterList_NT ?  ')'
    | '(' identifier_NT  ( ',' identifier_NT )  *  ')'
     ;

lambdaBody_NT
    : expression_NT
    | block_NT
     ;

primary_NT
    : '(' expression_NT ')'
    | THIS
    | SUPER
    | literal_NT
    | identifier_NT
    | typeTypeOrVoid_NT '.' CLASS
    | nonWildcardTypeArguments_NT  ( explicitGenericInvocationSuffix_NT | THIS arguments_NT ) 
     ;

classType_NT
    :  ( classOrInterfaceType_NT '.' )  ?  annotation_NT *  identifier_NT typeArguments_NT ? 
     ;

creator_NT
    : nonWildcardTypeArguments_NT createdName_NT classCreatorRest_NT
    | createdName_NT  ( arrayCreatorRest_NT | classCreatorRest_NT ) 
     ;

createdName_NT
    : identifier_NT typeArgumentsOrDiamond_NT ?   ( '.' identifier_NT typeArgumentsOrDiamond_NT ?  )  * 
    | primitiveType_NT
     ;

innerCreator_NT
    : identifier_NT nonWildcardTypeArgumentsOrDiamond_NT ?  classCreatorRest_NT
     ;

arrayCreatorRest_NT
    : '['  ( ']'  ( '[' ']' )  *  arrayInitializer_NT | expression_NT ']'  ( '[' expression_NT ']' )  *   ( '[' ']' )  *  ) 
     ;

classCreatorRest_NT
    : arguments_NT classBody_NT ? 
     ;

explicitGenericInvocation_NT
    : nonWildcardTypeArguments_NT explicitGenericInvocationSuffix_NT
     ;

typeArgumentsOrDiamond_NT
    : '<' '>'
    | typeArguments_NT
     ;

nonWildcardTypeArgumentsOrDiamond_NT
    : '<' '>'
    | nonWildcardTypeArguments_NT
     ;

nonWildcardTypeArguments_NT
    : '<' typeList_NT '>'
     ;

typeList_NT
    : typeType_NT  ( ',' typeType_NT )  * 
     ;

typeType_NT
    : annotation_NT ?   ( classOrInterfaceType_NT | primitiveType_NT )   ( '[' ']' )  * 
     ;

primitiveType_NT
    : BOOLEAN
    | CHAR
    | BYTE
    | SHORT
    | INT
    | LONG
    | FLOAT
    | DOUBLE
     ;

typeArguments_NT
    : '<' typeArgument_NT  ( ',' typeArgument_NT )  *  '>'
     ;

superSuffix_NT
    : arguments_NT
    | '.' identifier_NT arguments_NT ? 
     ;

explicitGenericInvocationSuffix_NT
    : SUPER superSuffix_NT
    | identifier_NT arguments_NT
     ;

arguments_NT
    : '(' expressionList_NT ?  ')'
     ;
