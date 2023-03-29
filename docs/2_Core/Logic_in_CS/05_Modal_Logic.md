## Modal Logic

it extends propositional and predicate logic

## World

is similar to state

it’s like a reality in Rick and Morty. We can assume every thing that can/cannot happpens as part of infinite realities.

## Symbols

|   Symbol   | Meaning     | Interpretation | CTL Equivalent |
| :--------: | ----------- | -------------- | -------------- |
|   $\Box$   | Necessarily | All worlds     | $AX$           |
| $\Diamond$ | Possibly    | Some world     | $EX$           |

## Scenarios

| Type          | Representation                              | Interpretation                       |
| ------------- | :------------------------------------------ | ------------------------------------ |
| Possibility   | $\Diamond \phi = \lnot \Box (\lnot \phi)$   | possibly true; not necessarily false |
| Necessity     | $\Box \phi = \lnot \Diamond (\lnot \phi)$   | necessarily true; not possibly false |
| Uncertainity  | $\lnot(\Box \phi) = \Diamond (\lnot \phi)$  | not necessarily true; possibly false |
| Impossibility | $\lnot (\Diamond \phi) = \Box (\lnot \phi)$ | not possibly true; necessarily false |

### Notes

1. Necessity requires possibility, impossibility requires uncertainity

2. Necessity $\implies$ possibility, impossibility $\implies$ uncertainity

5. Necessity and impossibility are not symbolically contradictory
   (look at the position of the $\lnot$ symbol)
   
$$
\begin{aligned}
&\Box(\phi) \\   \underbrace{}_\text{not here}
&\Box ( \lnot \phi)
\end{aligned}
$$
