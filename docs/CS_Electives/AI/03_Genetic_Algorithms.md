# Genetic Algorithm

Nature-inspired search technique to find true/approximate solutions to optimization and search problems

Categorized as global search heuristics

Neither complete nor optimal

## Terminology

| Term       | Meaning                                                      |
| ---------- | ------------------------------------------------------------ |
| Individual | Any possible solution                                        |
| Population | Collection of all individuals                                |
| Gene       | Single bit that represents an attribute                      |
| Trait      | Possible features of an individual                           |
| Chromosome | String of genes that represent the trait of an individual    |
| Genome     | Collection of all chromosome (traits) for an individual      |
| Fitness    | Target function that we are optimizing<br />(each individual has a fitness) |

## Algorithm

1. Initialize random population of solution guesses
2. Repeat for each generation
   1. Evaluate each chromosome in population using a fitness function
   2. Apply GA operators to create a new population
3. Repeat until desired fitness/stopping criterion is met

## GA Operators

### Representation

- Binary strings
- Arrays of integers (usually bound)
- Array of letters

### Selection

Selecting a subset of individuals $x$ according to fitness function $f(x)$ like beam search

| Selection Technique                        | Logic                                                        |
| ------------------------------------------ | ------------------------------------------------------------ |
| Roulette-wheel/<br />Fitness-proportionate | Each individual gets slice of wheel proportional to fitness<br />$p_i = \dfrac{f_i}{\sum_j^n f_j}$ |
| Elitist                                    | Select only $n$ most fit members of each generation          |
| Cutoff selection                           | Select only members with fitness > threshold                 |
| Scaling                                    |                                                              |
| Rank-Space                                 | Fitness ignores diversity, hence populations tend to become uniform<br/>1. Sort population by sum of fitness rank and diversity rank<br/>2. Diversity rank is the result of sorting by the function $(1/d^2)$ |

### Crossover/Recombination

- Parents are randomly (with prob) recombined to form new offsprings
- Chromosome of other parents copied onto next generation as is

#### Simple

1. For each couple, using a pre-determined prob, decide if crossover to be performed
2. Select 2 parents
3. Select cross site
4. Cut & substitute substring of one parent with another

$$
\begin{aligned}
&
\textcolor{hotpink}{101} 110 \quad
\textcolor{orange}{110} 001 \\
\\
\implies
&
\textcolor{orange}{110} 110 \quad
\textcolor{hotpink}{101} 001
\end{aligned}
$$

#### 2 Point

Helps avoid cases when genes at the beginning and end of chromosome are always split
$$
\begin{aligned}
&
1 \textcolor{hotpink}{01} 110 \quad
1 \textcolor{orange}{10} 001 \\
\\
\implies
&
1 \textcolor{orange}{10} 110 \quad
1 \textcolor{hotpink}{01} 001
\end{aligned}
$$

#### k-point/Multi-point

1. Pick $k$ random splice points
2. Splice for $k-1$ substrings

#### Uniform crossover

- Random subset is chosen for both parents
- Subset of parent 1 is substituted with subset of parent 2

### Mutation

Random alteration of mutating offsprings with small probability

- An insurance policy against lost bits
- Pushes out of local minima

#### Inversion

Reverse selected subsequence

1011011 -> 10 | 110 | 11 -> 10 | 011 | 11 -> 1001111

- Preserves adjacency information
- Discards order information

### Elitism

Best chromosomes from prev generation replace few of the worst chromosomes in current generation

## IDK

- Order1 crossover: inversion and recombination

## Applications

| Domain       | Application                                                  |
| ------------ | ------------------------------------------------------------ |
| Control      | Gas Pipelines<br />Missile evasion                           |
| Design       | Aircraft design<br />Keyboard configuration<br />Communication networks |
| Game playing | Poker<br />Checkers                                          |
| Security     | Encryption<br />Decryption                                   |
| Robotics     | Trajectory Planning                                          |

## Advantages

- Concept is easy
- Modular, separate from application
- Supports multi-objective optimization
- Easily exploits previous/alternate solutions
- Flexible building blocks for hybrid applications

## Issues

- How to select original population
- How to handle non-binary solution types
- What should be size of population
- What is the optimal mutation rate
- How are mates picked for crossover
- Can any chromosome appear more than once in a population
- Stopping criteria: When should GA halt
- How to deal with local minima
- How to parallelize

## Classifier Systems

- GAs & load balancing
- SAMUEL
