## Elementary Row Operations

$$
A =
\begin{bmatrix}
1 & 2 & -1 \\
-9 & 6 & 4 \\7 & 3 & -1
\end{bmatrix}_{3 \times 3}
$$

- Any 2 rows can be interchanged
  
	$R_1 \iff R_2$
- Any row can be multiplied/divided by any number other than 0
  
	$R_1 \to 2R_1$
- Any row can be added/subtracted to any row
  
	$R_1 \to R_1 \pm 2 R_2$

## REF

Reduced Echelon Form

Upper $\triangle$r matrix

- 1st non-zero elment in a row should be 1
  
	(called as leading one)
- Leading one should occur to the right side of previous rows’ leading one(s)
- If there is any zero row, it should be the last row
  otherwise, we need to interchange rows to ensure this rule

example

$$
\begin{bmatrix}
1 & 4 & 5 & 3 \\
0 & 1 & 2 & 8 \\0 & 0 & 1 & 5
\end{bmatrix} \quad
\begin{bmatrix}
1 & 4 & 3 & 5 \\
0 & 1 & 8 & 2 \\0 & 0 & 0 & 1
\end{bmatrix}
$$

## RREF

diagonal matrix

is the REF matrix where the elements of the columns of the leading ones (other than itself) are 0.

$$
\begin{bmatrix}
1 & 0 & 0 & 3\\
0 & 1 & 0 & 8\\0 & 0 & 1 & 5
\end{bmatrix} \quad
\begin{bmatrix}
1 & 0 & 5 & 0\\
0 & 1 & 8 & 0\\0 & 0 & 0 & 1
\end{bmatrix}
$$

## Rank

no of non-zero rows of a matrix in REF/RREF

## Gauss Methods

| Method            | Form |
| ----------------- | ---- |
| Gauss Elimination | REF  |
| Gauss Jordan      | RREF |

1. Write equation in matrix form $AX = B$, where
    - $A$ is coefficients matrix
    - $B$ is constant matrix
    - $X$ is variable matrix

   Converted augmented matrix = $[A | B]$ into REF

2. Cases
   
	 $n$ is the number of unknown variables
   
   | Rank(A\vert B)        |                    |
   | ----------------- | ------------------ |
   | $\ne$ rank(A)     | no solutions       |
   | $=$ rank(A) $= n$ | unique solutions   |
   | $=$ rank(A) $< n$ | infinite solutions |
   
3. Back Substitution
   
	 Degree of freedom = no of vars - no of equations

## Homogeneous Linear System

There will **always** be a solution.

If there is unique solution, it is always all 0s. This is called as trivial solution.

## Inverse of matrix

If $A$ and $B$ are 2 non-singular matrices such that $|A| \ne 0$, then $A^{-1} = B \iff A\cdot B = I$

$I$ is identity matrix

$$
\begin{aligned}
I_{2 \times 2} &=
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} \\
I_{3 \times 3} &=
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\0 & 0 & 1
\end{bmatrix}
\end{aligned}
$$

To find inverse

- use row transformations to convert $[A:I] \to [I:B]$
- then $B = A^{-1}$

If $A$ is singular, inverse does not exist
