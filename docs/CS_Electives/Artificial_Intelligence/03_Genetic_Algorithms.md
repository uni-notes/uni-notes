Search technique used in computing to find true or approximate solutions to optimization and search problems.

- Individual - Any possible solution
- Population - Group of all individuals
- Fitness – Target function that we are optimizing (each individual has a fitness)
- Trait - Possible features of an individual
- Genome - Collection of all traits for an individual

## Algorithm

1. Selection  
Choosing a subset of individuals to be mutated and copied over to next generation  
Different methods of selection :
    1. Roulette-wheel selection
    2. Elitist selection
    3. Fitness-proportionate selection
    4. Scaling selection
    5. Rank selection
2. Crossover  
The parents are randomly recombined (crossed-over) to form new off springs  
    1. Multi point crossover  
    2. Uniform crossover
3. Mutation
Reordering and randomly mutating the newly formed off springs

We keep performing the entire process repeatedly until the desired fitness or stopping criterion is met
