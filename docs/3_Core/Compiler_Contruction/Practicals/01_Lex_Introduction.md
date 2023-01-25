## Lex

Language translator, which converts

- **from** lex source program having regular expression, to match tokens in input string
- **to** a C program which has the function `yylex()` which is used to scan the input for tokens

  This will be the source file for lexical analysis of a C program. It will take a C program as input.

We use regular expressions to match lexemes and generate tokens

## Lex Source Code

```c
%{
#include
C Declarations
%}
Lex Symbols
%%
Rule
%%
Auxilliary functions (optional)
```

### Simplest Program

Default program which copies input to output

```
%%
%%
```

## Compilation

```bash
lex analyzer.l
cc lex.yy.c -ll
```

## Execution

```bash
a.out

// Takes user input
```

```bash
a.out < my_program.c > sample.txt
```

## Variables

|          | Data Type | Meaning                   | Default Value    |
| -------- | --------- | ------------------------- | ---------------- |
| `yytext` | `*char`   | pointer to matched string |                  |
| `yyleng` | `int`     | length of matched string  |                  |
| `yyin`   | `*file`   | Input Source              | STDIN (console)  |
| `yyout`  | `*file`   | Output Destination        | STDOUT (console) |

