## LISP

LISt Processing

In this course, we will use PICO LISP

## Functional Programming

is a subset of declarative programming

is not tied to von Neumman machine

Functional programs do not concern themselves with state and memory locations. They work exclusively with values, expressions and functions that compute values.

### Characteristics

- Simple and concise syntax and semantics
- Repetition is done through recursion instead of iteration
- Functions are manipulated easily like any data type
- Data as functions
  We can build a function on-the-fly and execute it
- Higher order functions
  Arguments and results of a function can be functions
- Lazy evaluation
  Expressions are evaluated only when necessary
- Garbage collection
  Dynamic memory that is no longer required is automatically reclaimed by the system
- Polymorphic types
  Functions can work on data of different types
- Easier mathematical manipulation compared to procedural programming
- Global assignments are not permitted (side effects are avoided)
- Easier parallelization
  Possibility of performing function evaluation in parallel is inherent in the function definition. Hence, no new language construct is required to express parallelism.

## Symbolic Expressions

|       |                                                   | Examples                                                     |
| ----- | ------------------------------------------------- | ------------------------------------------------------------ |
| Atom  |                                                   | happy<br />birthday<br />you<br />are<br />20 (numeric atom) |
| Lists | Group of atoms<br />==(**not** comma-separated)== | (happy birthday)<br />(you are 20)                           |

## LISP Procedures

Defined in ==**pre-fix**== format

Invokation consists of 

- pair of enclosing parentheses `(...)`
- procedure
- arguments

### Primitives

A **primitive** is an inbuilt procedure such as `+, -, *, /`

```lisp
(+ 3.14 3) ; 6 (rounded-off)
(- 3.14 3) ; 0 (rounded-off)
(* 3.14 3) ; 9 (rounded-off)
(/ 3.14 3) ; 1 (rounded-off)
(% 3.14 3) ; 0 (rounded-off)

(* (+ 3.14 3) 5) ; 30.0 (inner and outer calculations are rounded-off)

(- -8) ; 8

(max 3 5)	; 5
(min 3 5)	; 3
(sqrt 4)	; 2
(abs -5)	; 5
```
### Procedure Abstraction

constructing new user-defined procedures by combining existing ones

A program is a collection of procedures

Defined using `de` keyword

```lisp
(
 de simple(x)
 x
)

(simple 2) ; 2

(
 de pow(x n)
 (
  if(= n 0)
  1
  ( ; else
   *
   x
   (pow x (- n 1))
  )
 )
)

(pow 2 3) ; 8
```

### Side Effect

Anything done by a procedure that persists after it returns its value

[`setq`](#`setq`) function has a side effect that value of a variable changes

## Advanced Primitives

### `quote`

Takes 1 argument

returns the argument

Used for characters; basically for the programming language to know if we are passing a string or a character

`“...”` is for strings

```lisp
(quote (A))
(quote (A B C))

'A
'(A B C)
```

### `setq`

Sets a value to a variable

2 arguments

```lisp
(setq <variable> <value>)

(setq my_list (1 2 3))
(setq my_list '(A B C))
```

### `car`

First element of a non-empyt list

1 argument

```lisp
(car my_list) ; A
(car '(A B C)) ; A

(car 'L) ; error, as this is not a list
```

### `cdr`

All elements except the first elemnt

1 argument

```lisp
(car my_list) ; (B C)
(car '(A B C)) ; (B C)

(car 'L) ; error, as this is not a list
```

### `cons`

Create record structure

2 arguments

Returns a list such that

- arg1 is car
- arg2 is cdr

```lisp
; dotted pair
(cons 1 2) ; (1.2)
(cons 'a 'b); (a.b)
(cons 1 'b); (1.b)

; regular list
(cons 'a '(b c d)) ; (a b c d)
(cons '(a b) '(c d)) ; ((a b) c d)
```

### `append`

2 Arguments

```lisp
(append (1 2) (3 4)) ; (1 2 3 4)

(append '(A B) (3 4)) ; (A B 3 4)

(append (1) (2) (3) 4) ; (1 2 3.4)
```

### `list`

infinite arguments

converts a list of arguments to a list

```lisp
(list 1 2 3 4) ; (1 2 3 4)
(list 'a 'b 'c 'd) ; (a b c d)

(list 'a "hello world" (2 3) 'd) ; (a "hello world" (2 3) d)
```

### `cond`

conditional branching

```lisp
(
 cond
 (test_1 action_1)
 (test_2 action_2)
 ...
 (test_n action_n)
)

; with t
; t is like else
; t in a clause ensures that the last action is performed if none other would.
(t(
 cond
 (test_1 action_1)
 (test_2 action_2)
 ...
 (testn action_n)
 (action_default)
))
```

## Dotted Pair

created using [`cons`](#`cons`)

neither atoms nor lists
