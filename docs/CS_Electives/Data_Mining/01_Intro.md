# Introduction

Many times very high-quality professionals are not able to produce well, as they are usually incentivized to use complex methodologies. But data science is best when you actually solve the problem at hand, and help make decisions.

## Data Professionals

![img](./assets/data_professionals.jpg)

## Decision-Making

Iterative process with feedback loops; not linear

### Knowledge Hierarchy

Chain of increasing value

```mermaid
flowchart LR
Data -->
Information -->
Knowledge -->
Decision
```

|             |                                                              |
| ----------- | ------------------------------------------------------------ |
| Data        | Collection of numbers with known context and uncertainty estimates |
| Information | Right data at right time in right context, organized for access |
| Knowledge   | Interpretation of information based on model (understanding) of cause and effect |
| Decision    | Acting on knowledge for benefit                              |

### Process

- Preparation: Plan to turn data into information, with a specific model and decision in mind
- Testing/Experimenting/Measurement
- Analysis, with uncertainty: Use model to turn information to knowledge
- Decision
  - Using uncertainty
  - Risk/benefit analysis
- Post-mortem: Learnings to improve things

## Notes

- All models are wrong, some are useful
- Applying data mining algorithms on data that you don’t understand may result in false conclusions
- Always keep track of performed tests & analyses, to factor in data snooping

## Questions

| Question |                                                              |
| -------- | ------------------------------------------------------------ |
| Why      | Precise (not vague)<br /><br />Bad:<br />- Planning<br />- Decision-making<br /><br />Good<br />- What plans/decisions<br />- How are these plans/decisions made<br />- How would data mining help |
| What     | Goal, Level of aggregation, Forecast horizon<br /><br />Bad<br />- Sales<br />- Market share<br /><br />Good<br />- Demand |
| When     | Frequency<br />Time of day/year                              |
| Who      | Human judgement<br />Computer-generated with human judgement<br />Computer-generated<br /><br />Considerations<br />- Number & frequency of predictions<br />- Availability of historical data<br />- Relative accuracy of options |
| Where    | Predictions originate in different departments               |
| How      |                                                              |

![image-20240529002700056](./assets/image-20240529002700056.png)

![image-20240529014446419](./assets/image-20240529014446419.png)

## Fields Overview

|           | Analytics   | AI/ML                                          | Statistical Inference               |
| --------- | ----------- | ---------------------------------------------- | ----------------------------------- |
| Goal      | Descriptive | Predictive                                     | Prescriptive                        |
| Decisions |             | Large scale repetitive<br />(with uncertainty) | Small scale<br />(with uncertainty) |

![Data Roles](../assets/Data_Roles.svg)

![img](./../assets/overview_ai_ml_dl_ds.svg)

## Types of Analysis

| Type | Topic                                 | Nature               | Time   | Comment                                 | Examples|
|---                          | ---                                   | ---                  | ---    | ---                                     | ---|
|Descriptive/<br />Positive   | **What** is happening?                | Objective            | Past   | No emotions/explanations if good or bad | Increasing taxes will lower consumer spending<br />Increasing interest rate will lower demand for loans<br />Raising minimum wage will increase unemployment|
|Diagnostic                   | **Why** is it happening?              | Objective/Subjective | Past   | Helps in understanding root cause       | |
|Predictive                   | **What will happen** if condition happens | Subjective           | Future | Understanding future, using history     | |
|Prescriptive/<br />Normative | **What** to do                        | Subjective           | Future | what actions to be taken                | Taxes must be increased|

The complexity increases as we go down the above list, but the value obtained increases as well

## Project Lifecycle

```mermaid
flowchart TB

subgraph Scoping
	dp[Define<br/>Project] -->
	me["Define Metrics<br/>(Accuracy, Recall)"] -->
	re[Resources<br/>Budget] -->
    ba["Establish<br />Baseline"]
end

subgraph Data
	d[(Data Source)] -->
	l[Label &<br />Organize Data]
end

subgraph Modelling
  pre[Preprocessing] -->
	s[Modelling] -->
	train[Training] -->
  pp[Post<br />Processing] -->
	vt[Validation &<br />Testing] -->
	e[Error Analysis] -->
	pre
end

subgraph Deploy
	dep[Deploy in<br />Production] -->
	m[Monitor &<br />Maintain] & dss[Decision<br />Support System]
end

Scoping --> Data --> Modelling --> Deploy
```

https://www.youtube.com/watch?v=UyEtTyeahus&list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK&index=5

## Data Mining

Generate Decision Support Systems

> Non-trivial extraction of implicit, previously-unknown and potentially useful information from data

> Automatic/Semi-automatic means of discovering meaningful patterns from large quantities of data

## Predictive Tasks

Predict value of target/independent variable using values of independent variables

- Regression - Continuous
- Classification - Discrete

## Descriptive Tasks

Goal is to find

- Patterns
- Associations/Relationships

### Association Analysis

Find hidden assocations and patterns, using association rules

#### Applications

- Gene Discovery
- Market Baset Data Analysis
  Find items that are bought together

### Clustering/Cluster Analysis

Grouping similar customers

#### Metrics

- Similarity
- Dissimilarity/Distance Metrics

#### Applications

- Grouping similar documents

- Clustering documents

  1. Vocabulary - All terms(key words) from all docs

  2. Generate document-term frequency matrix

     | Document \vert  Term | T1   | T2   | …    | Tn   |
     | ---------------- | ---- | ---- | ---- | ---- |
     | D1               |      |      |      |      |
     | D2               |      |      |      |      |
     | …                |      |      |      |      |
     | Dm               |      |      |      |      |


### Deviation/Outlier/Anomaly Detection

Outlier is a data point that does not follow the norms.

Don’t mistake outlier for noise.

#### Application

- Credit Card Fraud Detection
    - Collect user profile such as Name, Age, Location
    - Collect user behavior data

- Network Intrusion Detection
- Identify anomalous behavior from surveillance camera videos

## Misconceptions

- All forecasts will be inaccurate, so no point
- If we had the latest forecasting technology, all problems would be solved

## IDK

Ensure you are looking at the correct scale

Model $y_t/y_0$ instead of $y_t$ to standardize all time series

## Common problems with analysis

- Poorly-defined goals
- Data doesn’t meet needs of analysis objectives
- Analysis makes unwarranted assumptions
- Model is wong
- Data doesn’t support conclusion

Learning Process

1. Model building: Functional form
2. Identify parameter weights
3. Distribution of random errors

Each of them can have different levels of generalizability

For eg: Ohm’s Law

1. $V=IR$ remains constant for all materials (under certain conditions)
2. $R$ Changes for different materials
3. Errors are dependent on measurement and experimental methods, and are independent of materials

