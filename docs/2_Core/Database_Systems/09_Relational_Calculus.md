## Tuple Relational Calculus (TRC)

Display loans over $1200

$$
\{ t | t \in \text{loan} \land t[\text{amount}] > 1200 \}
$$

Display loan number for every loan > $1200

$$
\begin{aligned}
\{ t | \exists s \in \text{loan} ( \\
 && t[\text{loanNumber}] &= s[\text{loanNumber}] \\ && \land s[\text{amount}] &> 1200 \\ ) \}
\end{aligned}
$$

Names of customers having loan at Perry branch ”

$$
\begin{aligned}
\{ & t | \textcolor{purple}{
\underbrace{
	\exists b \in \text{borrower} \land \exists l \in \text{loan}
}_\text{from}} ( \\
 &
\textcolor{hotpink}{
\underbrace{t.cn = b.cn}_\text{select}} , \\
 &
\textcolor{orange}{\underbrace{l.bn = \text{“Perry"} \land l.ln = b.ln}_\text{where} \\
} &) \} \end{aligned}
$$

## Domain Relational Calculus (DRC)

Display loans over $1200

$$
\{ \textcolor{purple}{\underbrace{<l, b, a>}_\text{select}} | \textcolor{hotpink}{\underbrace{<l, b, a> \in \text{loan}}_\text{from}} \land \textcolor{orange}{\underbrace{a > 1200}_\text{where}} \}
$$

Display names of customers having loan > $1200

$$
\{ <n> | \exists l, b, a something \}
$$

