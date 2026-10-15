## Yacc

yet another compiler compiler

Parser Generator: Bottom-up parser

==Lexical analyzer is a dependency for syntax analyzer==. In this case, lex is a dependency for yacc.

## Structure of program

### `filename.l`

```c
%{
#include "y.tab.h"
extern int yylval;
%}
%%

%%
```

### `filename.y`

```c
%{
  int yylex(void);
  void yyerror(char *);
  
  #include <stdio.h>
  #include <stdlib.h>
  
  C includes
  C declaration
%}

%token token_declaration_1 token_declaration_2
// lex must be able to identify these tokens

%%

LHS	: RHS1	{Action1}
		| RHS2	{Action2}
		| RHS3	{Action3}
		;

%%
  
void yyerror(char *s)
{
  printf("%s", s);
}

void main()
{
  yyparse();
}

C functions
```

## Compilation & Execution

```bash
yacc -d filename.y
lex filename.l
cc lex.yy.c y.tab.c -ll -lm
a.out
```

|       |                                                              | Required? |
| ----- | ------------------------------------------------------------ | --------- |
| `-d`  | Flag that instructs to generate the definitions of the tokens | ✅         |
| `-ll` | Link lex loader                                              | ✅         |
| `-lm` | Link math                                                    |           |

## Output Files

| File Name | Purpose                                                      |
| --------- | ------------------------------------------------------------ |
| `y.tab.h` | Header file containing definitions of tokens<br />**(must be included in lex file)** |
| `y.tab.c` | Parser C Code                                                |

## Value of Symbols

Every yacc grammar symbol has a value associated with it.

- LHS = `$$`
- RHS = `$1, $2, ...`

$$
\$\$ = \$1 \text{ Operation } \$3 \\
$$

Example

$$
\begin{aligned}
\$\$ &= \$1 \text{ + } \$3 \\
\$\$ &= \$1 \text{ - } \$3 \\
\$\$ &= \$1
\end{aligned}
$$

## Question 1

Write a simple calculator which can gives result for an addition/subtraction expression (ambiguity allowed).

### `05_1.l`

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
"\n" return NL;
[ \t] ; 
. printf("Invalid");
%%
```

### `05_1.y`

```c
%{
  int yylex(void);
  void yyerror(char *);
        
  #include <stdio.h>
  #include <stdlib.h>
%}

%token INTEGER PLUS MINUS NL

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

### Compilation

```bash
yacc -d calculator.y
```

```bash
yacc: 4 shift/reduce conflicts
```

### Output

```bash
1 - 5 + 2
-6 (should actually be -2)
```

## Conversion from BNF to YACC Rule

```
A -> a | ε

A : a
	| 
	;
```

