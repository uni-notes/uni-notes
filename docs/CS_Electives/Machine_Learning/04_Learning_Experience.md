# Learning Experience $E$

## Learning Objectives

|                                                  |                                                              |
| ------------------------------------------------ | ------------------------------------------------------------ |
| Prediction                                       | Estimation of unseen data                                    |
| Modelling/<br />Characterization/<br />Inference | How do inputs affect output<br /><br />Obtain the Sample CEF/Conditional Distribution which closely matches the Population CEF/Conditional Distribution |
| Optimization                                     | What input values produce desired outputs (both mean and variance) |
| Control                                          | How to adjust controlled inputs to maximize control of outputs |
| Simulation                                       |                                                              |
| Causal inference                                 |                                                              |

Use ML models for developing structural models, and then let the structural models to make the predictions, not the ML models

- Why: Black swans can be predicted by theory, even if they cannot be predicted by ML
- How: Use a non-parametric ML to identify important variables and then develop a parametric structural form model.

## Learning Paradigms

| Method                         | Meaning                                                      |                                       | Application     |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------- | --------------- |
| Supervised                     | Uses labelled data, to derive a mapping between input examples and target variable. | $D_\text{train} = X, y$               |                 |
| Unsupervised                   | Learning from unlabelled data                                | $D_\text{train} = X$                  |                 |
| Semi-Supervised                | $\exists$ labelled data and large amount of unlabelled data.<br/>Label the unlabelled data using the labelled data.<br/><br/>For example, **love** is labelled as emotion, but **lovely** isn’t<br /><br />Cotraining, Semi-Supervised SVM |                                       |                 |
| Lazy/<br />Instance-Based      | Store the training examples instead of training explicit description of the target function.<br/><br/>Output of the learning algorithm for a new instance not only depends on it, but also on its neighbors.<br/><br/>The best algorithm is KNN (K-Nearest Neighbor) Algorithm.<br/><br/>Useful for recommender system. |                                       |                 |
| Active<br />AL                 | Learning system is allowed to choose the data from which it learns.<br />There exists a human annotator.<br/><br/>Useful for gene expression/cancer classification |                                       |                 |
| Multiple Instance              | Weakly supervised learning where training instances are arranged in sets.<br/>Each set has a label, but the instances don’t |                                       |                 |
| Transfer                       | Reuse a pre-trained model as the starting point for a model on a new related task |                                       |                 |
| Reinforcement Learning<br />RL | Learning in realtime, from experience of interacting in the environment, without any fixed input dataset.<br />It is similar to a kid learning from experience.<br/><br/>Best algorithm is **Q-Learning algorithm**. | $D_\text{train} = X, \text{Feedback}$ | Game playing    |
| Bayesian Learning              | Conditional-probabilistic learning tool<br/>Each observed training expmle can incrementally inc/dec the estimated probability that a hypothesis is correct.<br/><br/>Useful when there is chance of false positive.<br/>For eg: Covid +ve |                                       |                 |
| Deep<br />DL                   | Multi-Layered ANNs                                           |                                       | Computer Vision |
| Federated<br />FL              | Distributed                                                  |                                       | Privacy         |
| Online                         | Streaming                                                    |                                       |                 |

## Training Method

|                                                 |                                                              | Advantage                                                  |
| ----------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Batch                                           | $\hat f: X \to y$                                            | Better model                                               |
| Streaming/<br />Online/<br />Passive-Aggressive | $\hat f_b: X_{i \le b} \to y_{i \le b}$<br />where $b= \text{Mini-batch}$ | - Adaptive to new data points<br />- Computationally-cheap |
| Hybrid                                          | - Batch training start of day<br />- Online training intra-day |                                                            |

## Types of Learners

They are not adapted by the ML algo itself, but we can use nested learning, where other algorithms optimize the hyperparameter for the ML algo.

|                  | Eager Learner                                        | Lazy Learner                                       |
| ---------------- | ---------------------------------------------------- | -------------------------------------------------- |
| Training         | Learns relationship between class label & attributes | Stores training records                            |
| Evaluation       |                                                      | Perform computations to classify evaluation record |
| Training Speed   | Slow                                                 | Fast                                               |
| Evaluation Speed | Fast                                                 | Slow                                               |
| Example          | - Decision Tree<br />- Rule-Based Classifier         | - Nearest-neighbor classifier                      |
