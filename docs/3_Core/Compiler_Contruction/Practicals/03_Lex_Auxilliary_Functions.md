## File Input & Console Output

```c
%%
[0-9]+ printf("Found a number");
%%
void main()
{
	yyin = fopen("in.txt", "r"); // open file in read mode
	yyout = fopen("out.txt", "w"); // open file in read mode
	yylex(); // invoke scanner
}
```

## File Input & File Output

```c
%%
[0-9]+ fprintf(yyout, "Found a number"); // print to file
%%
void main()
{
	yyin = fopen("in.txt", "r"); // open file in read mode
	yyout = fopen("out.txt", "w"); // open file in write mode
	yylex(); // invoke scanner
}
```

## User-Defined Vars & Functions

```c
%{
void display();
%}
digit [0-9]
number {digit}+
%%
number display();
%%
void display()
{
printf("Found a number");
}
```

## User-Defined Vars

```c
%{
void display();
int a;
%}
digit [0-9]
number {digit}+
%%
{number} {
  a = atoi(yytext);
  display(a);
}
%%
void display()
{
printf("Found number %d", a);
}
```

or

```c
%{
void display();
%}
digit [0-9]
number {digit}+
%%
{number} display(yytext)
%%
void display(yytext)
{
  int a = atoi(yytext);
	printf("Found number %d", a);
}
```

## Question 1

Write a LEX program to recognize the following 

- Operators: +, -, *, /, |, 
- Numbers
- newline
- Any other character apart from the above should be recognized as mystery character

For each of the above mentioned matches (classes of lexeme) in your input, the program should print the following: PLUS, MINUS, MUL, DIV, ABS, NUMBER, NEW LINE, MYSTERY CHAR respectively. Your program should also strip of whitespaces.

```c
%%
"+" printf("PLUS");
"-" printf("MINUS");
"*" printf("MUL");
"/" printf("DIV");
"|"" printf("ABS");
[0-9]+ printf("Number");
\n printf("Newline\n");
. printf("Wildcard ");
%%
```

## Question 2

Write a LEX program to print the number of words, characters and lines in a given input.

```c
%{
int cc = 0, wc = 0, lc = 0;
%}
%%
[a-zA-Z]+ {wc++; cc+=strlen(yytext);}
\n {lc++; cc++;}
. {cc++;}
%%

int yywrap()
{
	return 1;
}

void main()
{
	yylex();
	printf("%d\n", cc);
	printf("%d\n", wc);
	printf("%d\n", lc);
}
```

## Question 3

Write a LEX program to print the number of words, characters and lines in a given input, but a word and its characters are counted only if its length is greater than or equal to 6.

```c
%{
  #include <stdio.h>
  int num_words = 0;
  int num_chars = 0;
  int num_lines = 0;
%}
%%
[\t]+ {
  // Ignore whitespace
}

\n {
  num_lines++;
}

[a - zA - Z]{6,} {
  num_words++;
  num_chars += yyleng;
}

. {
  if (yyleng >= 6)
    {
      num_chars += yyleng;
    }
}

%%
int main ()
{
  yylex ();
  printf ("Number of words: %d\n", num_words);
  printf ("Number of characters: %d\n", num_chars);
  printf ("Number of lines: %d\n", num_lines);
  return 0;
}
```

## Question 4

Write a LEX program to print if the input is an odd number or an even number along with its length. Also, the program should check the correctness of the input (i.e. if the input is one even number and one odd number).

```c
%{
#include<stdlib.h>
#include<stdio.h>
  int number_1;
  int number_2;
%}
number_sequence [0 - 9]*
%%
{number_sequence}[0 | 2 | 4 | 6 | 8] {
  printf ("Even number [%d]", yyleng);
  return atoi (yytext);
}

{number_sequence}[1 | 3 | 5 | 7 | 9] {
  printf ("Odd number [%d]", yyleng);
  return atoi (yytext);
}
%%
int main()
{
  printf ("\nInput an even number and an odd number\n");
  number_1 = yylex ();
  number_2 = yylex ();
  int diff = number_1 - number_2;
  if (diff % 2 != 0)
    printf
      ("\nYour inputs were checked for correctness, \nResult : Correct\n");
  else
    printf
      ("\nYour inputs were checked for correctness,\nResult : Incorrect\n");
  return 1;
}
```

