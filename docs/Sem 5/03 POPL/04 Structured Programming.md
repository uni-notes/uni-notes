## Structured Programming Context

CA sequential computation consists of sequence of actions. These computations are dynamic occur during program execution

Program text is static

It is essential that the program text represents the computation that occurs when the program runs

## Design Principles of Imperative Languages

1. Structure of program text should help understand what program does. Reliability of structured programs make them easier to modify for efficiency
2. Language must allow underlying machine to be used efficiently

## Structured Control Flow

A program is structured if the flow of control through the program is evident from the syntactic structure of the program text.

This is done by making structured statements ==single-entry, single-exit==

## Sequential Statements

sequence of statements

A compound statement is a block of code, which is a grouped statement enclosed in some manner, such as `{...}`, `begin ... end`

## Selection/Conditional Statements

### `if ... else`

- `if <exp1> then <stmt1>`
- `if <exp1> then <stmt1> else <stmt2>`
- Nested conditionals

### `switch...case`

```c
switch(exp)
{
  case const1:
    ; // stmt1
    break;
  case const2:
    ; // stmt2
    break;
  default:
    ; // default stmt
}
```

**Properties**

- Case constants can appear in any order
- Case constants need not be consecutive
- Several case constants can select the same sub-statement
- Case constants must be distinct to avoid ambiguity

**Notes**

- Pascal gives error if none of the cases are selected
- Pascal uses `else` case instead of `default` case

### `if ... else` vs `switch ... case`

Case statements are preferred over `if ... else` when we have **adjacent conditionals**. Otherwise, use `if … else`

```c
// ✅
switch(s)
{
  case 1:
  case 2:
  case 3:
}

// ❌
switch(s)
{
  case 1:
  case 2000:
  case 30000:
}

// ✅
if (s == 1)
else if (s == 2000)
else if (s == 30000)
```

When using cases, a **jump table** is created, which contains entry $i$ that is machine instruction to jump to code for case $i$. The no of entries in the jump table $= \max - \min + 1$. However, only the cases that are in the our code are actually used.

Compiler uses jump table if atleast half of the entries are used. Else, the compiler uses a **hash table** instead.

## Iterative/Loops Statements

- Definite Loop
  Known number of iterations
- Indefinite Loop
  Number of iterations is only known at run time
- Infinite loop
  Loop keeps iterating

### While

`while <exp> do <stmt`

![while_loop](assets/while_loop.png)

### Do-While

`repeat <stmt> until <exp>`

### For

`for <id> = <exp> to <exp> do <stmt>`

**Components**

- index/iterative variable
- Step
- Limit

### Handling Special Cases

- `break`
- `continue`

## Return Statements

`return <exp>`

Sends control from a procedure back to a caller carrying the value of the expression. If return is not inside a procedure the program halts.

Break vs Return

- Break : control goes out of a loop
- Return : control goes out of a procedure

## Goto Statements

Basically unconditional jump

```c
goto L
  ...
L: <stmt>
```

