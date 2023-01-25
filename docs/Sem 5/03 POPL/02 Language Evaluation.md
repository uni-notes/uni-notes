## Qualities of a Language

| Quality         | Meaning                                                      |
| --------------- | ------------------------------------------------------------ |
| Readability     | Ease with humans can read and understand programs            |
| Writability     | Ease with humans can write programs                          |
| Reliability     | Program performs to its specifications under all conditions  |
| Cost-efficiency | - Efficiency of Training programmers<br />- Efficiency of writing, compiling, executing, reading programs<br />- Efficiency of computation and development<br />- Monetary cost of compilers, license<br />- Maintenance cost, due to reliability<br />- Portability (standardization of the language)<br />- Generality (applicability to a wide range of use-cases/applications) |

## Features

| Characteristic          | Affects<br />Readability? | Affects<br />Writability? | Affects<br />Reliability? |
| ----------------------- | :-----------------------: | :-----------------------: | :-----------------------: |
| Simplicity              |             ✅             |             ✅             |             ✅             |
| Orthogonality           |             ✅             |             ✅             |             ✅             |
| Data Types & Structures |             ✅             |             ✅             |             ✅             |
| Syntax Design           |             ✅             |             ✅             |             ✅             |
| Abstraction             |                           |             ✅             |             ✅             |
| Expressivity            |                           |             ✅             |             ✅             |
| Type Checking           |                           |                           |             ✅             |
| Exception Handling      |                           |                           |             ✅             |
| Restricted Aliasing     |                           |                           |             ✅             |

## Simplicity

### Fewer features and basic constructs

The main cause of readability issues is because program author uses a subset different from what the reader is familiar with

### Fewer Feature Multiplicity

Feature Multiplicity = Ability to do the same operation in different ways

```java
count = count + 1;
count += 1;
count++;
++count;
```

### Fewer Operator Overloading

Ambiguity arises due to ability of operator to perform multiple operations

## Orthogonality

Constructs in programming languages should be independent of each other; should not be redundant. Every combination of features should be meaningful.

Any operation has minimal undesired side effects.

> Orthogonality is the property that means "Changing A does not change B".
>
> An example of an orthogonal system would be a radio, where changing the station does not change the volume and vice-versa.
>
> A non-orthogonal system would be like a helicopter where changing the speed can change the direction.
>
> In programming languages this means that when you execute an instruction, nothing but that instruction happens (which is very important for debugging).

## Data Types & Structures

For eg, the existence of `boolean` data type in a programming language is important, as otherwise we have to use integers (which make the program less clear)

## Syntax Design

### Identifiers

Names for variables, functions, arrays, structures, etc

#### Rules

- Starting character must be alphabet/underscore
- Other characters can be
  - Alphabet
  - Underscore
  - Digits
- Max Length = 31 characters

### Key Words

`while`, `for`, `class`

Most programming languages use braces for pairing control structers

### Semantics should follow Syntax

Semantics = meaning of your code; basically the logic

For eg, use of `static`, `extern` in C

### Form and Meaning

Self-descriptive constructs and meaningful keywords

## Abstraction

Ability to define and use complex structures/operations in ways that allow details to be ignored

### Process Abstraction

Functions/Sub-routines for codes that will be required multiple times

### Data Abstraction

Ability to create own data structures, such as binary tree using pointers/integers

## Expressivity

Set of relatively convenient ways of specifying operations

### Examples

- `count++` instead of `count = count + 1`
- `for` instead of `while`

## Type Checking

Testing for type errors - ensuring that operands of an operator are of compatible type

Run-Time type checking is expensive, hence compile-time type checking is preferred

| Statically-Typed Languages                | Dynamically-Typed Languages                                  |
| ----------------------------------------- | ------------------------------------------------------------ |
| C<br />C++<br />C#<br />Java<br />Fortran | Objective-C<br />Groovy<br />Javascript<br />LISP<br />Lua<br />PHP<br />Prolog<br />Python<br />Ruby<br />Smalltalk<br />TCL |

## Exception Handling

Intercept run-time errors and take corrective measures

## Restricted Aliasing

Aliasing = Presence of multiple names for the same memory location

Changing value pointed by one pointer changes the value pointed by another pointer to the same location.

❌ In C, union members and pointers may be set to point to the same variable.

## Grep

**G**et **re**gular **e**x**p**ression