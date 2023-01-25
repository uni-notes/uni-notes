## Propositional Symbols

| Symbol        | Meaning                         |
| ------------- | ------------------------------- |
| $\top$        | True                            |
| $\bot$        | False                           |
| $\land, \lor$ | and, or (whichever comes first) |
| $\to$         | implies                         |
| $\vdash$      | Conclusion                      |

## Natural Deduction

$\phi_1, \phi_2, \dots$ are Premises

$\psi$ is Conclusion

$\phi_1, \phi_2, \dots, \phi_n \vdash\psi$ is called sequent

## Rules

1. $\frac{\phi \quad \psi}{\phi \land \psi} \quad (\land i)$
2. $\frac{\phi \land \psi}{\phi}, \frac{\phi \land \psi}{\psi} \quad (\land e_1, \land e_2)$
3. $\frac{\phi}{\lnot \lnot \phi} \quad(\lnot \lnot i)$
4. $\frac{\lnot \lnot \phi}{\phi} \quad(\lnot \lnot e)$
5. $\frac{ 
   \begin{bmatrix} \phi \\ \vdots \\ \psi
   \end{bmatrix}
   }{\phi \to \psi} \quad (\to i)$
6. $\frac{\phi \quad \phi \to \psi}{\psi} \quad (\to e)$
7. $\frac{\lnot \psi \quad \phi \to \psi}{\lnot \phi}$ (MT)
8. $\frac{\phi}{\phi \lor \psi}, \frac{\psi}{\phi \lor \psi} \quad (\lor i_1, \lor i_2)$
9. $\frac{\phi \lor \psi \quad 
   \begin{bmatrix}
   \phi \\ \vdots \\ \chi
   \end{bmatrix}
   \begin{bmatrix}
   \psi \\ \vdots \\ \chi
   \end{bmatrix}
   }{\chi} \quad (\lor e)$
10. Copy rule
11. $\frac{\begin{bmatrix}
    \phi \\ \vdots \\ \bot
    \end{bmatrix}
    }{\lnot \phi} \quad (\lnot i)$
12. $\frac{\begin{bmatrix}
    \lnot \phi \\ \vdots \\ \bot
    \end{bmatrix}
    }{\phi}$ PBC
13. $\frac{\phi \quad \lnot \phi}{\bot} \quad (\lnot e)$
14. $\frac{\bot}{\phi} \quad (\bot e)$
15. $\frac{}{\phi \lor \lnot \phi}$(LEM)

## Equivalence Relation

If a formula can be proved in both directions, then it is called as an equivalence relation.

Denoted by $\dashv \vdash$

## WFF

Well-formed formula

There's brackets for every operation

$(p \land (\lnot q)) \to (p \lor (q \lor (\lnot r) ))$

## Parse Tree

Shows the order at which the terms and operations are parsed

## Semantic entailment

$\phi \models \psi$

this means that whenever$\phi$ is true, $\psi$ is true

ie $\phi \to \psi$ is true for all cases

## Soundness

if ND is true, then even semantic entailment is true

$\phi \vdash \psi \implies \phi \models \psi$

## Completeness

if semantic entailment is true, then even ND is true

$\phi \models \psi \implies \phi \vdash \psi$

## Semantic equivalence

$\phi \equiv \psi$

$\phi \models \psi$ and $\psi \models \phi$

both have the same truth table

## Interpretation

assigning values (putting inputs)

## Valuation

getting outputs

## Tautology

function whose output is always true

## Valid

true for all interpretations

## Satisfiable

true for at least one interpretation

### Conclusions

- A is valid$\iff \lnot A$ is un-satisfiable
- A is satisfiable$\iff \lnot A$ is invalid

## CNF

basically POS

1. literal - variable
2. clause
3. formula

$\underbrace{ ( \underbrace{p}_\text{literal} \lor q) \and \underbrace{(r \lor s)}_\text{clause}  }_\text{formula}$

if $p$ and $p’$ both exist within all clauses, all clauses are true
then formula is valid

### Implies Conversion

$p \to q = \lnot p \lor q$

## Horn Clauses

useful for checking satisfiability

only contains $\land$ and $\to$

every clause contains $\to$

**cannot** contain

- $\lor$
- $\lnot$

$$
\underbrace{ 

( \underbrace{p}_\text{proposition} \land q \to s) 

\land 

\underbrace{(
\underbrace{ p \land q \land r }_\text{assumption}\to s)}_\text{clause}
}_\text{formula}
$$

### Checking Satisfiability

we just have to check if there exists **atleast one** combination of variables such that the entire formula is true.

if nothing is possible, then it is false.
