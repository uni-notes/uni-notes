## Comparator/Magnitude checker

used to check if a binary number is less than/equal to/greater than another binary no

## 1 bit comparator

| x    | y    | G    | E    | L    |
| ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 1    | 0    |
| 0    | 1    | 0    | 0    | 1    |
| 1    | 0    | 1    | 0    | 0    |
| 1    | 1    | 0    | 1    | 0    |

$L = x'y, E = x \odot y, G = xy'$

## 2 Bit Comparator

| A_1  | A_0  | B_1  | B_0  | G    | E    | L    |
| ---- | ---- | :--- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 0    |      | 1    |      |
| 0    | 0    | 0    | 1    |      |      | 1    |
| 0    | 0    | 1    | 0    |      |      | 1    |
| 0    | 0    | 1    | 1    |      |      | 1    |
| 0    | 1    | 0    | 0    | 1    |      |      |
| 0    | 1    | 0    | 1    |      | 1    |      |
| 0    | 1    | 1    | 0    |      |      | 1    |
| 0    | 1    | 1    | 1    |      |      | 1    |
| 1    | 0    | 0    | 0    | 1    |      |      |
| 1    | 0    | 0    | 1    | 1    |      |      |
| 1    | 0    | 1    | 0    |      | 1    |      |
| 1    | 0    | 1    | 1    |      |      | 1    |
| 1    | 1    | 0    | 0    | 1    |      |      |
| 1    | 1    | 0    | 1    | 1    |      |      |
| 1    | 1    | 1    | 0    | 1    |      |      |
| 1    | 1    | 1    | 1    |      | 1    |      |

- $G = A_1 {B_1}' + A_0 A_1 {B_0}' + A_0 {B_0}' {B_1}'$
- $E = (A_1 \odot B_1) (A_2 \odot B_2)$
- $L = {A_1}' B_1  + {A_0}' {A_1}' B_0 + {A_0}' B_0 B_1$

## 3 Bit Comparator

- E
    - $x_2 = A_2 \odot B_2 \quad (A_2 = B_2)$
    - when $A_2 = 0, B_2 = 0, A_2 = 1, B_2 = 1$
    - $x_1 = A_1 \odot B_1 \quad (A_1 = B_1)$
    - $x_0 = A_0 \odot B_0 \quad (A_0 = B_0)$
    - $E = x_2 \cdot x_1 \cdot x_0 = (A_2 \odot B_2) \cdot (A_1 \odot B_1) \cdot (A_0 \odot B_0)$
- L
    - if $A_2 < B_2$
    - $A_2 = 0, B_2 = 1 \implies {A_2}' B_2$
    - if $A_2 = B_2, A_1 < B_1$
    - $x_2$ and$A_1 = 0, B_1 = 1 \implies {A_1}' B_1$
    - $x_2 \cdot {A_1}' B_1$
    - if $A_2 = B_2, A_1 = B_1, A_0 < B_0$
    - $x_2, x_1$ and$A_0 = 0, B_0 = 1 \implies {A_0}' {B_0}$
    - $x_2 \cdot x_1 \cdot {A_0} B_0$
    - $\therefore, L = {A_2}' B_2 + x_2 {A_1}' B_1 + x_2 x_1 {A_0}' B_0$
- G
    - $G = A_2 {B_2}' + x_2 A_1 {B_1}' + x_2 x_1 A_0 {B_0}'$

## 4 bit comparator

$E = x_3 x_2 x_1 x_0$

## Diagram

![comparator](img/comparator.svg){ loading=lazy }
