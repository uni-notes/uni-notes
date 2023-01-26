## Steps

1. Divide dataset into subsets
2. For each subset, apply a model
     - This model is usually decision tree
3. Aggegrate the results

## Stability of Classifier

For unstable models, we have to change model when adding new point

For stable models, not required

## Learning Techniques

|                   | Single            | B<span style="color:hotpink">agg</span>ing<br />(Boostrap <span style="color:hotpink">agg</span>regation) | Boosting                                                     |
| ----------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| No of learners    | 1                 | $n$                                                          | $n$                                                          |
| Training          | Complete training | Random sampling with replacement                             | Random sampling with replacement **over weighted data**      |
|                   |                   | <span style="color:hotpink">Agg</span>regage the results at the end |                                                              |
| Training sequence | N/A               | Parallel                                                     | Sequential                                                   |
|                   |                   |                                                              | Only pass over the mis-classified points<br />We boost the probability of mis-classified points to be picked again |
| Preferred for     |                   | Linear Data                                                  | Non-Linear Data                                              |
| Example           |                   | Random forest                                                |                                                              |

### Training Speed

It **cannot** be said that boosting is slower than bagging, just because it is sequential and bagging is parallel.

This is because, boosting may end in just 10 iterations, but you may need 1000 classifiers for bagging.

## Random Forest

Bagging

Uses Decision Tree

- $n$ is sample size
- $S_n$ is number of samples
- $T$ no of trees (chosen by engineer)
- $M$ Total no of features
- $m$ no of features
    - $m$ chosen randomly
    - $m << M$

### Steps

- collect votes from every tree in the forest

- Use majority voting to decide class label
   $$
  y = \frac{1}{n} \sum_i y_i
   $$

## Ada Boost

Boosting algorithm

Stump = weak classifier (decision tree)

Level of tree $\le$ 3 (max 2 root nodes)

### Steps

1. Set initial weight of each sample point as $\frac{1}{n}$

2. Use every stump

   1. Find mis-classified points

   2. Get error rate for each stump
      
$$
{\Large \epsilon} = \sum_\text{wrong} w_i
$$

3. Pick lowest error rate classifier

4. $$
   \alpha = \frac{1}{2} \ \log \left|\frac{1-\epsilon}{\epsilon}\right|
   $$

5. $$
   f(x) = \sum_{i=1}^T \alpha_i \ h_i(x)
   $$

     - $f(x)$ is the final function
     - $h_i(x)$ is the hypothesis function of each ada boost iteration

6. $$
   w_\text{new} = 
   \begin{cases}
   \dfrac{w_\text{old}}{2(1-\epsilon)} & \text{✅ Point Classified Correctly}  \\   \dfrac{w_\text{old}}{2 \epsilon} & \text{❌ Point Misclassified}
   \end{cases}
   $$

7. Repeat steps 2-6 until sufficient accuracy is obtained

8. Find the final value
   
$$
y = \frac{1}{n} \sum_i w_i y_i
$$
