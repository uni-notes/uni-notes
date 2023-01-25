## Linearly-Dependent/Independent

Let $\alpha_1 u_1 + \alpha_2 u_2 + \dots + \alpha_n u_n = \vec 0$

|                  Condition                   | Conclusion  |
| :------------------------------------------: | :---------: |
| $\alpha_1 = \alpha_2 = \alpha_3 = \dots = 0$ | Independent |
|                     else                     |  Dependent  |

### Working

1. Column-wise

2. | Condition  |        Solution        | Conclusion  |
   | :--------: | :--------------------: | :---------: |
   | $r(A) = n$ | unique $(0, 0, \dots)$ | independent |
   |    else    |    infinitely-many     |  dependent  |

## Span

Let $\vec v = \alpha_1 u_1 + \alpha_2 u_2 + \dots + \alpha_n u_n$

### Working

1. Column-wise

2. 

3. | Condition           | Solution | Conclusion |
   | ------------------- | :------: | :--------: |
   | $r(A) = r(A:B) = n$ |  unique  |    span    |
   | $r(A) = r(A:B) < n$ | infinite |    span    |
   | else                |          |  not span  |

## Basis

1. $S$ is Linearly-independent –> row-wise working
2. $S$ spans $V$ –> dim($V$) = no of vectors in $S$

**Note:** $\vec 0$ has no basis

## Dimension

no of unknowns

no of vectors in its basis

dim$(\vec 0)= 0 $