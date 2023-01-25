## Drift

| Type          |      |
| ------------- | ---- |
| Concept Drift |      |
| Data Drift    |      |

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