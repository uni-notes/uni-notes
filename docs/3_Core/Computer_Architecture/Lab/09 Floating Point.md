## 1: Area of circle

```assembly
.data
pi: .double 3.1415926535897924 
r: .float 2.2

.text
main:
ldc1 $f2, pi
lwc1 $f4, r
cvt.d.s $f4, $f4

mul.d $f12, $f4, $f4	## r^2
mul.d $f12, $f2, $f12	## pi r^2

addi $v0, $zero, 3
syscall

## exit
addi $v0, $zero, 10
syscall
```

## 2: Convert from F to C

```assembly
.data 
f: .float 98.6

.text
main:
li.s $f2, 5.0
li.s $f3, 9.0

lwc1 $f4, f
li.s $f5, 32.0

div.s $f6, $f2, $f3	## 5/9
sub.s $f7, $f4, $f5	## f - 32

mul.s $f12, $f6, $f7	## (5/9)*(f-32)

addi $v0, $zero, 2
syscall

## exit
addi $v0, $zero, 10
syscall
```

## 3: Value of $ax^2 + bx + c$ for inputted $x$

```assembly
.data 

.text
main:

## read x
addi $v0, $zero, 6
syscall

mov.s $f2, $f0 ## x
mul.s $f3, $f0, $f0 ## x^2

li.s $f4, 1.0 ## a
li.s $f4, 1.0 ## b 
li.s $f5, 1.0 ## c

mul.s $f6, $f4, $f3 ## ax^2
mul.s $f7, $f5, $f2 ## bx

add.s $f12, $f6, $f7 ## ax^2 + bx
add.s $f12, $f12, $f5 ## ax^2 + bx + c

## Print
addi $v0, $zero, 2
syscall

## exit
addi $v0, $zero, 10
syscall
```

