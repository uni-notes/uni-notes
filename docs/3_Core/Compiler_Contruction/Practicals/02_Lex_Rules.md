## Match Real Numbers

```c
digit [0-9]
sign [+|-]
%%
{sign}?{digit}+(\.{digit}+)? printf("Matched real no: %s of length: %d", yytext, yyleng);
%%
```

```
300.21
Matched real no: 300.21 of length: 6
```

## Conflict Resolution

### Different Rules

```c
%%
a printf("Matched %d a\n", yyleng);
aa printf("Matched %d a\n", yyleng);
%%
```

```
aaaaaaa
Matched 2 a
Matched 2 a
Matched 2 a
Matched 1 a
```

### Similar Rules

`Warning: Rule not matched`

```c
letter [a-z A-Z]
digit [0-9]
%%
{letter}({letter}|{digit})* printf("Matched id");
{letter}+ printf("Matched word");
%%
```

```
sum
Matched id
hello
Matched id
```

## Match word immediately followed by number

```c
letter [a-z A-Z]
word {letter}+
digit [0-9]
%%
{word}/{digit} printf("Found word %s followed by number ", yytext);
%%
```

```
hello123
Found word hello followed by number 123
```

