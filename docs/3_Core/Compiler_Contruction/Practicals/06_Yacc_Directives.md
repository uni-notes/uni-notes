## Directives

Along with the tokens

```c
%left PLUS MINUS
%left MUL DIV
%right POW
%%
%%
```

The directives written later have higher precedence.

## Question 1

Write a simple calculator which can gives result for an addition/subtraction expression (without ambiguity).

### `06_1.l`

same as [Basic Calculator](05_Yacc_Introduction.md#`05_1.l`) 

### `06_1.y`

```c
%{
  int yylex(void);
  void yyerror(char *);
        
  #include <stdio.h>
  #include <stdlib.h>
%}

%token INTEGER PLUS MINUS NL
%left PLUS MINUS											/* 1 */
%%
program:
expr NL {printf("%d\n", $1); exit(0);}
;
expr:
INTEGER {$$=$1;}
| expr PLUS expr	{$$=$1+$3;}
| expr MINUS expr	{$$=$1-$3;}
;
%%
void yyerror(char *s)
{
	printf("%s\n", s);
}
int main()
{
	yyparse();
}
```

### Output

```bash
1-5+2
-2
```

## Question 2

The program should keep going on until the user exits using `Ctrl-D`

### `06_2.l`

same as [Basic Calculator](05_Yacc_Introduction.md#`05_1.l`) 

### `06_2.y`

```c
%{
  int yylex(void);
  void yyerror(char *);

  #include <stdio.h>
  #include <stdlib.h>
  %}

%token INTEGER PLUS MINUS NL
%left PLUS MINUS											/* 1 */
%%
program:
program expr NL {printf("%d\n", $2);}	/* 2 */
| 
;
expr:
INTEGER {$$=$1;}
| expr PLUS expr	{$$=$1+$3;}
| expr MINUS expr	{$$=$1-$3;}
;
%%
void yyerror(char *s)
{
  printf("%s\n", s);
}
int main()
{
  yyparse();
}
```

### Output

```bash
1 - 5 + 2
-2
1 - 5 + 2
-2
1 - 5 + 2
-2
```

## Question 3

Extend the calculator to incorporate some new functionality. New features include arithmetic operators * and / that can multiply and divide integers respectively. Parentheses may be used to over-ride operator precedence. Note * and / operators have higher precedence over + and – operators. Also note that * and / are left associative. Ensure this using directive in YACC. 

### `06_3.l`

```c
%{
#include "y.tab.h"
extern int yylval;
%}
%%
[0-9]+ {
	yylval = atoi(yytext);
	return INTEGER;
}
"+" return PLUS;
"-" return MINUS;
"*" return MUL;		/* 1 */
"/" return DIV;		/* 2 */
"^" return POW;		/* 3 */
"\n" return NL;		/* 4 */
[ \t] ; 
. printf("Invalid");
%%
```

### `06_3.y`

```c
%{
  int yylex(void);
  void yyerror(char *);

  #include <stdio.h>
  #include <stdlib.h>
  #include <math.h>
%}

%token INTEGER PLUS MINUS NL
%left PLUS MINUS
%left MUL DIV
%right POW												/* 1 */
%%
program:
program expr NL {printf("%d\n", $2);}
| 
;
expr:
INTEGER {$$=$1;}
| expr PLUS expr	{$$=$1+$3;}
| expr MINUS expr	{$$=$1-$3;}
| expr MUL expr	{$$=$1*$3;} 			/* 2 */
| expr DIV expr	{$$=$1/$3;}				/* 3 */
| expr POW expr	{$$=pow($1, $3);}	/* 4 */
;
%%
void yyerror(char *s)
{
  printf("%s\n", s);
}
int main()
{
  yyparse();
}
```

### Output

```bash
2 - 2 ^ 2
-3
2 - 3 * 3 ^ 6
-2185
```

## Question 4

Modify the calculator application so that it works for floating point values also.

## Question 5

Modify the grammar to allow single-character variables to be specified in assignment statements. The following illustrates sample input and calculator output:

```bash
user: 3 * (4 + 5)
calc: 27
user: x = 3 * (4 + 5)
user: y = 5
user: x
calc: 27
user: y
calc: 5
user: x + 2*y
calc: 37
```

### `06_5.l`

```c
%{
  #include "y.tab.h"
  extern int yylval;
%}
%%

[a-z] {
    yylval = *yytext - 'a';
    return VARIABLE;
}

[0-9]+ {
    yylval = atoi(yytext);
    return INTEGER;
}

[-+()=*/\n] { return *yytext; }
[ \t] ;

. yyerror("invalid character");
%%

int yywrap(void) {
 return 1;
} 
```

### `06_5.y`

```c
%{
    #include<stdio.h>
    int flag=0;
    int yylex(void);
    int sym[26];
%}
%token INTEGER VARIABLE
%left '+' '-'
%left '*' '/'
%%

program:
    program statement '\n'
    |
    ;
statement:
    expr { printf("%d\n", $1); }
    | VARIABLE '=' expr { sym[$1] = $3; }
    ;
expr:
    INTEGER
    | VARIABLE { $$ = sym[$1]; }
    | expr '+' expr { $$ = $1 + $3; }
    | expr '-' expr { $$ = $1 - $3; }
    | expr '*' expr { $$ = $1 * $3; }
    | expr '/' expr { $$ = $1 / $3; }
    | '(' expr ')' { $$ = $2; }
    ;
%%
void main()
{
printf("\nEnter Any Arithmetic Expression which	can have operations Addition, Subtraction, Multiplication, Division, Modulus and Round brackets:\n");

yyparse();
if(flag==0)
printf("\nEntered arithmetic expression is Valid\n\n");
}

void yyerror()
{
printf("\nEntered arithmetic expression is Invalid\n\n");
flag=1;
}
```

