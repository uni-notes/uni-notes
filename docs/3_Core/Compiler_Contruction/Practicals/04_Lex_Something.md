## Question 1

Write a LEX program to get a binary input and print whether the given input is a power of two or not.

```c
%{
#include <stdio.h>
#include <stdlib.h>
%}
%%
[01]+ {
    int num = 0, base = 2, exp;
    int result;
    int len = strlen(yytext);
    for (int i = 0; i < len; i++) {
        if (yytext[i] == '1') {
	    result = 1;
	    exp = len-i-1;
            while (exp != 0) {
        	result *= base;
        	--exp;
    	}
	    num += result;
        }
    }
    if (num == 0) {
        printf("The input is not a power of two\n");
    } else if ((num & (num-1)) == 0) {
        printf("The input is a power of two\n");
    } else {
        printf("The input is not a power of two\n");
    }
}
. {
    printf("Invalid input\n");
}
%%
int main() {
    yylex();
    return 0;
}
```

## Question 2

Write a LEX program to insert line numbers to a file. For this copy your favourite C program “input.c” to your folder which would be the input to your LEX program.

```c
%{
#include <stdio.h>
int line_number = 1;
%}

%%

\n {
    line_number++;
    printf("\n%d: ", line_number);
}

. {
    printf("%c", yytext[0]);
}

%%

int main() {
    FILE *input_file = fopen("input.c", "r");
    if (input_file == NULL) {
        printf("Error: cannot open file.\n");
        return 1;
    }
    yyin = input_file;
    printf("1: ");
    yylex();
    fclose(input_file);
    return 0;
}
```

## Question 3

Write a LEX program to save the contents of an input file excluding comment lines to another file.

```c
%{
#include <stdio.h>
#include <stdbool.h>
bool in_block_comment = false;  /* initialize in_block_comment flag to false */
bool in_line_comment = false;   /* initialize in_line_comment flag to false */
%}

%%

"/*"    { in_block_comment = true; }  /* set in_block_comment flag to true on start of block comment */
"*/"    { in_block_comment = false; }  /* set in_block_comment flag to false on end of block comment */
"//"    { in_line_comment = true; }   /* set in_line_comment flag to true on start of line comment */
\n      {
            if (in_line_comment) { in_line_comment = false; }  /* set in_line_comment flag to false on end of line */
            if (!in_block_comment) { fputc('\n', yyout); }     /* write newline character to output file if not in comment */
         }
.       { if (!in_block_comment && !in_line_comment) { fputc(yytext[0], yyout); } }  /* write character to output file if not in comment */
%%

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s input_file output_file\n", argv[0]);
        return 1;
    }

    FILE* input_file = fopen(argv[1], "r");
    if (input_file == NULL) {
        printf("Error: could not open input file %s\n", argv[1]);
        return 1;
    }

    FILE* output_file = fopen(argv[2], "w");
    if (output_file == NULL) {
        printf("Error: could not open output file %s\n", argv[2]);
        return 1;
    }

    yyin = input_file;
    yyout = output_file;
    yylex();

    fclose(input_file);
    fclose(output_file);
    return 0;
}
```

## Question 4

Write a LEX program that would take a BITS student's roll number as input and prints the details of the student based on that. You are expected to write regular expressions that would synthesize information like, year of joining, specialization, PS/Thesis, Registration index, Campus (U) etc. from the given roll number. If the given input does not abide by the Roll number format, print some error message.

```c
%{
#include <stdio.h>
%}

%%

^[0-9]{4}[A-Za-z0-9]{2}(PS|TS)[0-9]{4}[HPUG]$  {
    printf("Year of Joining: %c%c%c%c\n",yytext[0], yytext[1],yytext[2], yytext[3]);
    printf("Specialization: %c%c\n", yytext[4], yytext[5]);
    printf("Thesis/Practice School: %c%c\n", yytext[6], yytext[7]);
    printf("Registration Index: %c%c%c%c\n", yytext[8], yytext[9], yytext[10], yytext[11]);
    printf("Campus: %c\n", yytext[12]);
}
.  {
    printf("Invalid roll number format.\n");
}

%%

int main() {
    yylex();
    return 0;
}
```

