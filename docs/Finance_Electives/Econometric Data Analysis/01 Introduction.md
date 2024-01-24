# Econometric Analysis

Machine Learning $\to$ Statistics $\to$ Econometrics

The focus is on using the predictions of Machine Learning to analyse future causalities. This is more insight-oriented, as identifying patterns without understanding causes and implications is useless

## Applications

### Supervised ML

Logistic Regression can be used for classification of whether or not a person will default on their loans, based on their balance.

Linear/Polynomial Regression is obviously used everywhere

### Unsupervised ML

Clustering can be used for identifying the type of consumers

# Deep Learning

It’s just Machine Learning neural network, but with many layers. Instead of choosing the features on your own, you let the algorithm learn the best features for the model.

Meant to be used for complex models, rather than for models which can fitted with just regular linear regression.

[Deep Learning | Andrew Ng](..\Andrew NG\Machine Learning\ML.md#Deep Learning)

# Causal vs Statistical

|                                       | Causal Prediction                                            | Statistical Prediction                                       |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| what will be $y$ if I ___<br />$x=a$? | set                                                          | observe                                                      |
| probability of                        | $E \big [y |\text{do}(x = a) \big]$                          | $E \big [y |(x = a) \big]$                                   |
| characteristic                        | manual action                                                | natural outcome                                              |
| statement                             | if $x$ has a causal effect on $y$, then we can change $x$ and expect it to cause a change in $y$. | if $x$ is correlated with $y$ without any causal effect on $y$, then we can only observe the correlation without the ability to change $y$ by changing $x$ |

- $\text{do}(x= 1)$ means that you manually **perform some action** to set $x$ as 1.
- $\text{see}(x= 1)$ means that you **observe** the change of $x$ as 1. 

# Correlation $\centernot \implies$ Causation

Correlation doesn’t always imply causation. correlation is useful for prediction, but not for understanding exactly why.

For eg, labor wage and their years of education has a strong correlation, but the reason for that could be

- education actually helps
- education just acts as the signal/proof for the employees that you possess knowledge
- or, it could just be that well-off people get better jobs, and coincidentally, they are getting more educated

# Examples

### Rain

P(rain | barometer = low) > P(rain | barometer = high), this is seeing. But
P( rain | do(barometer = low) ) = P(rain | do(barometer = high) ). This is because me doing something to manipulate the barometer reading isn’t gonna change the rainfall.

We can now conclude that the barometer reading itself has no causal effect on rainfall. It is infact the atmospheric pressure that has causal effect on the rainfall.

### Hospital

Let’s say there are 2 hospitals

|               | A    | B    |
| ------------- | ---- | ---- |
| Recovery Rate | 0.6  | 0.4  |

It seems like A is better than B. But A may not necessarily be the better hospital for me to go to. What if A is a regular community hospital and B is a speciality hospital for cancer patients. Obviously, A will have a better recovery rate. 
So, statistical prediction (is not wrong) is accurate saying that recovery rate for a patient going to hospital A is 0.6, because we are seeing the patient going there; the patient that goes there is a patient who chose to go there. If they were a serious patient, then they would’ve gone to B.
But, causal prediction is useful for deciding which hospital a patient should go to. It helps understand the actual help provided by the hospital. We are **manually** choosing here the hospital here.
This is why causal prediction is useful.

### Ad-Sales

Sales and Advertisement have a high correlation. But doesn’t necessarily mean that Increasing advertisement will increase sales.
This is because, the observed advertisment could be due to previous sales. So the observed advertisment here is a natural outcome, not an active decision; ie, last year i got high sales profits, so i increased my advertisment budget this year, or i got bad profit so i increased my ad budget to somehow increase the sales. it’s like a reaction, not an active decision
However, when you actively decide to increase the ad budget, the change in sales depends on the causal effect of ad budget on sales. Why don’t companies do this causal analysis???

1. they don’t know their math and stats :laughing:
2. causal analysis is harder