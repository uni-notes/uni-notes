# Production

Stage after deploying the model to work with live data

Refer to https://www.youtube.com/watch?v=2wXA1jQqJJ4&list=PLdfopzFjkPz9shHCeH9poe9sbAn0pIojX

## Drift

| Type          | Meaning                                      | Identification         | Solution          |
| ------------- | -------------------------------------------- | ---------------------- | ----------------- |
| Concept Drift |                                              |                        |                   |
| Data Drift    | Train & Test data not from same distribution | Adversarial Validation | Anomaly Detection |

### Adversarial Validation

Create a new feature in the dataset as “Set”, which signifies if the data belongs to training/test set

Train a classifier to predict which set

ROC-AUC signifies how accurately the classifier can distinguish between the sets. Higher values $\ge 0.8$ imply that Train & Test data **not** from same distribution.

## Deployment Checklist

- Realtime or batch training
- Cloud vs Edge/Browser
- Computer resources (CPU/GPU/Memory)
- Latency, throughput (QPS)
- Logging
- Security & Privacy

## Scenarios of Deployment

- New product/capability
- Automate/assist with manual task
- Replace previous ML system

## Types of Deployment

| Type       |                                                              |
| ---------- | ------------------------------------------------------------ |
| Canary     | Roll out to small fraction of traffic initially<br />Monitor system and ramp up traffic gradually |
| Blue-Green | Fully deploy new version (green)<br />Keep old model dormant, and rollback to it if required (blue) |

## Degrees of Automation

|                    |      |
| ------------------ | ---- |
| Human-Only         |      |
| Shadow Mode        |      |
| AI Assistance      |      |
| Partial Automation |      |
| Full automation    |      |

## Monitoring

- Brainstorm potential problems
- Brainstorm appropriate metrics to identify the problems
  - Software Metrics
    - Memory
    - Compute
    - Latency
    - Throughput
    - Server load
  - Input Metrics
    - Average Input length
    - Fraction of rows with missing values
    - Average image brightness
  - Output metrics
    - Missing outputs
    - No of times user redoes search
    - CTR (ClickThrough Rate): No of clicks that your ad receives divided by the number of times your ad

## Model Serving

![image-20240118224856092](./assets/image-20240118224856092.png)