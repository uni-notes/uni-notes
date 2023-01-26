## Predicate Logic

- $\exists$ there exists (at least one)
  similar to $\lor$
- $\forall$ for all
  similar to $\land$

### Predicates

return true or false

all caps

eg:

- MAN(x): x is a man
  // returns T/F
- ADULT(x): x is an adult
  // returns T/F

### Functions

returns object

eg:

- mother(x)
  // returns the mother of x (object)
- age(x)
  // returns the age of x (integer)

All predicates are functions, but **not** all functions are predicates

### Variables

they can occur in 2 places

1. along with $\forall$ and $\exists$
   1. eg: $\forall x, \exists y$
2. as leaf nodes (terminal vertices)
   1. $x, y$

#### Bounded variables

under $\forall$ or $\exists$ in parse tree

#### Free variables

are not under any restrictions
they can be substituted/replaced by bounded/free variables

### Substitution

$\phi[t/x]$ means that $t$ replaces free variable $x$

$t$ can be anything - bounded/free var
for eg:

- $t = t$
- $t = f(x,y)$
- $t = g(x,y,z)$

any of the above can replace free variable

### ND

1. $= e$
   1. $a=b, b = c \implies a = c$
   2. Basically transitiveness
   
2. $\forall e$
   1. $\frac{\forall x \ \phi}{ \phi [x_0/x] }$
   2. if $\phi$ is true for all $x$, then we can replace $x$ by $x_0$ in $\phi$, and conclude that $\phi[x_0/x]$ is also true
   2. eg: if all students are teens, then a student Ahmed is also a teen
   
3. $\forall i$
   
   1. $\frac{
      \begin{bmatrix}
      x_0 \\
\vdots \\
\phi
      \end{bmatrix}
      }{\forall x}$
   2. assumption box
   3. if we prove that a student is a teen, then all students are teens (considering that all $x$, ie students are identical)
   
4. $\exists i$

   1. $$
         \frac{
         \phi[t/x]
         }{\exists x \quad \phi} \quad
         \exists x \quad i
		 $$
   
   2. We can deduce $\exist x \quad \phi$ whenever we have $\phi[t/x]$; $t$ has to be free for $x$ in $\phi$
5. $\exists e$
   1. $$
      \frac{
        \exists x \ \phi \quad
        \begin{bmatrix}
        	x_0 \quad \phi[x_0/x] \\
					\vdots \\
					\chi
        \end{bmatrix}
        }
        {\chi} \quad
        \exists x \quad e
      $$
   
   2. if $\exists x \quad \phi$ is true, there should be atleast one value of $x$ for which $\phi$ is true
   
   3. Let $x_0$ represent those values
   
   4. Substituting $x_0$ for $x$, we arrive at formula $\chi$
   
   5. we then conclude $\chi$

### Quantifier equivalences

1. De-Morgan’s rule

   convert bw $\forall$ and $\exists$ when there is negation
     - $\lnot \forall x (\phi) \dashv \vdash \exists x (\lnot \phi)$
     - $\lnot \exists x (\phi) \dashv \vdash \forall x (\lnot \phi)$
2. Distributive
   1. $\forall x (\phi) \land \forall x(\psi) \dashv \vdash \forall x (\phi \land \psi)$
   2. $\exists x (\phi) \lor \exists x (\psi) \dashv \vdash \exists(\phi \lor \psi)$
3. Commutative
   1. $\forall x \forall y (\phi) \dashv \vdash \forall y \forall x (\phi)$
   2. $\exists x \exists y (\phi) \dashv \vdash \exists y \exists x (\phi)$

## IDK

|   Symbol    | Term                        | Meaning                |
| :---------: | --------------------------- | ---------------------- |
|     $P$     | predicate                   |                        |
|     $l$     | lookup table                | gives us environment   |
|             | environment                 | conditions             |
| $\mathbb M$ | Model                       | shows relations        |
|  $\models$  | Semantic Entailment         |                        |
| $\models_l$ | Models wrt lookup table $l$ |                        |
|  $\Gamma$   | Set of formulae             |                        |
|             | Arity                       | no of vars/relations?? |

## Properties

### Compactness

let $\Gamma$ is a set of formulae in predicate logic.

If all finite subsets of $\Gamma$ are satisfiable, then $\Gamma$ is satisfiable

### Godel’s Completeness Theorem

### Underdesirablility

If there are a large instances, it will be hard to determine the validity of a formula. This is called as undesirability.

## Verification

1. framework for modelling
2. specification language - eg: Predicate logic
3. verification language - eg: ND

| Verification Type |                          |
| ----------------- | ------------------------ |
| Proof-Based       | $\Gamma \vdash \phi$     |
| Model Based       | $\mathbb M \models \phi$ |
