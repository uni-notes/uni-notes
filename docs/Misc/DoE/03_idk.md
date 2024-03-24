## Experiment Pointers

- Always perform experiment with both trial & control samples
  
- Always get the raw data; processing should be done by analyst, not data providers
  
- Every data point should have central tendency & uncertainty associated
  - Incorporate all potential uncertainty associated with collecting the data & use [Uncertainty Propagation](../Machine_Learning/09_Uncertainty.md#Uncertainty-Propagation)
  - Preferably use robust [summary statistics](05_Data_Exploration.md#Summary-Statistics) such as median and IQR, rather than mean & variance
- Every data point fed to model should be iid observation

![doe](./assets/doe.svg)

## Data Template

### For Collection

| Type  | Category_ID | Subcategory_ID | Reading_ID | Value |
| ------- | ------- | ---------- | :--------: | ----: |
| Control | Product A | Sample 1  | 1          | x     |
| Control | Product A | Sample 1  | 2          | x     |
| Control | Product A | Sample 1  | 3          | x     |
| Control | Product A | Sample 2  | 1          | x     |
| Control | Product A | Sample 2   | 2          | x     |
| Control | Product A | Sample 2   | 3          | x     |
| Control | Product B | Sample 1   | 1          | x     |
| Control | Product B | Sample 1   | 2          | x     |
| Control | Product B | Sample 1   | 3          | x     |
| Control | Product B | Sample 2   | 1          | x     |
| Control | Product B | Sample 2   | 2          | x     |
| Control | Product B | Sample 2   | 3          | x     |
| Trial | … | … | … | … |

### For Modelling

We cannot use the collection data directly for modelling as each row is not iid observation. Hence aggregation is required to obtain the central tendency & uncertainty for each iid observation.

| Type    | Category_ID | Subcategory_ID | Central Tendency<br />(Median) | Uncertainty<br />(IQR) |
| ------- | ----------- | -------------- | -----------------------------: | ---------------------: |
| Control | Product A   | Sample 1       |                              x |                      x |
| Control | Product A   | Sample 2       |                              x |                      x |
| Control | Product B   | Sample 1       |                              x |                      x |
| Control | Product B   | Sample 2       |                              x |                      x |
| Trial   | …           | …              |                              … |                      … |



