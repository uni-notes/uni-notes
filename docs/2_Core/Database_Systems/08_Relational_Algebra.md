## Query Languages

Procedural

Non-Procedural

## Relational Algebra

|      Operator      |                      Symbol                       |
| :----------------: | :-----------------------------------------------: |
|       select       |                     $\sigma$                      |
|      project       |                       $\pi$                       |
|       union        |                      $\cup$                       |
|   set difference   |                        $-$                        |
| cartesian product  |                     $\times$                      |
|       rename       |                    $\rho_x(E)$                    |
| natural/inner join |                      $\Join$                      |
|  left outer join   |                         ⟕                         |
|  right outer join  |                         ⟖                         |
|  full outer join   |                         ⟗                         |
|        sum         | $_\text{Semester} \ g_\text{sum(age)}({student})$ |
|      average       | $_\text{Semester} \ g_\text{avg(age)}({student})$ |

### Merging

$\sigma_{A=B} T_1 \times T_2 (\text{instructor})$

### Insertion

$$
\begin{aligned}
&\text{account} \leftarrow \text{account } \cup \{ \\ 
&\text{(“Ahmed", A-973, 1200)} \\&\text{(“Thahir", A-193, 1300)} \\&\}
\end{aligned}
$$

### Update

$$
\begin{aligned}
&\text{account} \leftarrow
\Pi (something)
\end{aligned}
$$

## Additional operations

Not exactly part of relational algebra, but 

1. Set Intersection
2. Natural Join
3. Division
4. Assignment

### Division

> Find all guests names who have a booking with ==all== tour agencies located in Dubai.

- **Column** - common to A&B
- **Tuples** - records of A having the same records in B

$$
\begin{aligned}
R ÷ S = & \\
\{ \quad
& t[a_1,...,a_n] : \quad t \in R \\& \land \forall s \in S \Big( (t[a_1, \dots ,a_n] \cup s) \in R \Big)
\quad \}
\end{aligned}
$$

### View

$$
\begin{aligned}
&\text{create view allCustomers as} \\&\Pi_\text{branchName, customerName} (
	\text{depositor$\Join$account}
) \\&\Pi_\text{branchName} ( \\&\sigma \text{ something}\\&)
\end{aligned}
$$

