## Tuple Relational Calculus (TRC)

Display loans over $1200

$$
\set{
t |
t \in \text{loan} \land
t[\text{amount}] > 1200
}
$$
Display loan number for every loan > $1200

$$
\begin{align}
\{
t | \exists s \in \text{loan} ( \\&& t[\text{loanNumber}] &= s[\text{loanNumber}] \\&& \land s[\text{amount}] &> 1200 \\)
\}
\end{align}
$$
Names of customers having loan at Perry branch ”

$$
\begin{align}
\{
& t |
\textcolor{purple}{
\underbrace{
	\exists b \in \text{borrower} \land \exists l \in \text{loan}
}_\text{from}} (\\&
\textcolor{green}{
\underbrace{t.cn = b.cn}_\text{select}}
,\\&
\textcolor{orange}{
\underbrace{l.bn = \text{“Perry"} \land l.ln = b.ln}_\text{where} \\}
&) \}
\end{align}
$$

## Domain Relational Calculus (DRC)

Display loans over $1200

$$
\{
\textcolor{purple} {\underbrace{<l, b, a>}_\text{select}}
|
\textcolor{green} {\underbrace{<l, b, a> \in \text{loan}}_\text{from}}
\and
\textcolor{orange} {\underbrace{a > 1200}_\text{where}}
\}
$$
Display names of customers having loan > $1200

$$
\{
<n> | \exists l, b, a somethign
\}
$$
