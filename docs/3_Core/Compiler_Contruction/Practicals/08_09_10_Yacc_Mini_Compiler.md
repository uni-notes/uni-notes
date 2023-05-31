Write the LEX and YACC source to recognize the following:

## The template for the C program is 

```
#include<stdio.h>
int main( )
{
} 
PGM -> HEADER INT MAIN LB RB LCB RCB
```

## Declaration statements:

Allow declaration statements inside the program body. Integer variables separated by comma can be declared inside the program body. A program can have multiple declaration statements. Variables are sequence of lower-case alphabets.Each declaration statement is ended by a semicolon.
int a, b;

```
PGM -> HEADER INTMAIN LB RB LCB BODY RCB
BODY -> DECL_STMTS
DECL_STMTS -> DECL_STMT DECL_STMTS | DECL_STMT
DECL_STMT ->INT VAR_LIST SC
VAR_LIST->VAR COMA VAR_LIST | VAR
```

## Operators & Program Statements

Allow declaration statements to be followed by program statements inside the program body. Program statements are ended by a semicolon. Program statements can be arithmetic expressions involving +-*/ operators.

```
PGM -> HEADER INT MAIN LB RB LCB BODY RCB
BODY -> DECL_STMTS PROG_STMTS
DECL_STMTS -> DECL_STMT DECL_STMTS | DECL_STMT
PROG_STMTS -> PROG_STMT PROG_STMTS | PROG_STMT	
DECL_STMT -> INT VAR_LIST SC
VAR_LIST -> VAR COMA VAR_LIST | VAR
PROG_STMT -> VAR EQ A_EXPN SC
A_EXPN -> A_EXPN OP A_EXPN | LB A_EXPN RB | VAR
```

## Modify your LEX program to incorporate the following changes

As per the current set up, the programmer is supposed to use only lower-case alphabets in variable names in their C program. Modify your lex program so as to let the programmer have uppercase letters A to Z together with digits 0 to 9 and underscore character in variable names. Ensure that a variable name always begin with a character.

Terminate your program with an error message if in case the programmer uses keywords if, while, do, and for as variable names. Note that it is permitted to have variable names beginning with keywords (ifvar, thenextcount, donut etc.) (hint: rely on conflict resolution rules in LEX).

Add provision to declare variables of type float, double and char.

## Adding operators to the language

- Incorporate arithmetic expressions involving binary operators +, -, *, /, ^ (exponent) into your compiler. Note that the exponent operator is a right associative operator and has higher precedence than other arithmetic operators (+, -, *, /).
- Incorporate unary pre/post increment ++ and pre/post decrement -- operators too (are of highest precedence and left associative).
- Incorporate the modulo operator (%). It has the same precedence as * and / operators and is left associative. 
- Include numeric integer constants as expressions.
- Also include parenthesized expressions
- Variables can be of int/float/char/double type
- In the given implementation, the input C file is expected to have all declaration statements in the beginning, followed by program statements. Rewrite your grammar to let the user to have declarative and program statements in any arbitrary interleaved order in their input C program.

## Combined Program

### `file.l`

```c
%{
#include "y.tab.h"
extern int yylval;
%}
%%
"#include<stdio.h>"    { return T_HEADER; }
"int"    { return T_INT; }
"float"  { return T_FLOAT; }
"double" { return T_DOUBLE; }
"char"   { return T_CHAR; }
"main"   { return T_MAIN; }
"do"						{ printf("ERROR! Reserved keyword do\n"); return -1;}
"if"						{ printf("ERROR! Reserved keyword if\n"); return -1;}
"while"					{ printf("ERROR! Reserved keyword while\n"); return -1;}
"for"						{ printf("ERROR! Reserved keyword for\n"); return -1;}
\{       { return T_LCB; }
\}       { return T_RCB; }
\(       { return T_LB; }
\)       { return T_RB; }
\n       { yylineno++; }
[ \t]    ;
[0-9]+   { return T_NUMBER; }
[-+*/]   { return T_OP; }
=        { return T_EQ; }
[a-zA-Z][a-zA-Z0-9_]*       { return T_VARIABLE; }
","        { return T_COMMA; }
";"        { return T_SC; }
.      { return yytext[0]; }
%%
int yywrap()
{
return 1;
}
```

### `file.y`

```c
%{
#include <stdio.h>
#include <stdlib.h>
int yylex(void);
void yyerror(char *);
%}
%token T_HEADER T_INT T_CHAR T_FLOAT T_DOUBLE T_MAIN T_LB T_RB T_LCB T_RCB
%token T_NUMBER T_VARIABLE T_COMMA T_SC T_OP T_EQ
%left T_OP
%%
program: HEADER INT MAIN LB RB LCB BODY RCB
{
	printf("Valid\n");
}
;
BODY: DECL_STMTS PROG_STMTS
{
	printf("Token: Body\n");
}
;
HEADER: T_HEADER
{
	printf("Token: Header\n");
}
;
INT: T_INT
{
	printf("Token: Int\n");
};
MAIN: T_MAIN
{
	printf("Token: Main\n");
}
;
LB: T_LB
{
	printf("Token: Left_Bracket\n");
}
;
RB: T_RB
{
	printf("Token: Right_Bracket\n");
}
;
LCB: T_LCB
{
	printf("Token: Left_Curly_Bracket\n");
}
;
RCB: T_RCB
{
printf("Token: Right_Curly_Bracket\n");
}
;
DECL_STMTS: DECL_STMT DECL_STMTS
					|	DECL_STMT
{
	printf("Token: Statement\n");
}
;

PROG_STMTS: PROG_STMT PROG_STMTS
					| PROG_STMT
;
DECL_STMT:	T_INT VAR_LIST T_SC
					| T_FLOAT VAR_LIST T_SC
					| T_DOUBLE VAR_LIST T_SC
					| T_CHAR VAR_LIST T_SC
					;

VAR_LIST: T_VARIABLE T_COMMA VAR_LIST
					|   T_VARIABLE
					;

PROG_STMT: T_VARIABLE T_EQ A_EXPN T_SC
					 ;

A_EXPN: A_EXPN T_OP A_EXPN
			| T_LB A_EXPN T_RB
			| T_VARIABLE
			| T_NUMBER
			;
%%
void main() 
{
	printf("Enter a C program:\n");
	yyparse();
}

void yyerror(char *s) 
{
	printf("The provided code is invalid\n", s);
	exit(1);
}
```
