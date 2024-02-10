# Project Management

Project Management is the planning, organization, securing, motivation and control of resources to successfully complete a project, while minimizing unproductive efforts

## Project Management Models

|                 |                                                              | Advantage                             | Disadvantages                                                |
| --------------- | ------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------ |
| Waterfall       | - Concept: Setting the goal<br/>- Design: Acquire the requirements, such as programmers<br/>- Pre-Production: Verify if the concept and design are feasible<br/>- Production<br/>- Testing<br/>  - Alpha<br/>  - Beta<br/>- Shipping<br/>- Maintenance |                                       | Most of the actual feedback will occur at beta testing, which is too far ahead in the timeline, and hence it will be too late to respond. |
| Classical Agile | - Individuals and interactions > processes and tools<br/>- Working software > comprehensive documentation<br/>- Customer collaboration > contract negotiation<br/>- Responding rather > over following a plan | Focus on features that consumers want | Assumes interchangeable tasks & team members<br />Depends on good communication |
| Scrum           | 1. Transparency<br />- Everyone on the team related to a decision is involved<br />- Shared responsibility<br />2. Inspection: Anyone on the team curious on why something is the way it is, can<br />3. Adaptability |                                       |                                                              |

## Scrum

### Keywords

|               |                                                              |
| ------------- | ------------------------------------------------------------ |
| Sprint        | Group of tasks to be completed by all teams concurrently     |
| Sprint Length | Duration of sprint<br />Usually 1 week                       |
| Task          | Atomic chunk of work an individual can do independently that you can estimate duration for<br />Usually $\le 8 \text{hrs}$<br />Usually assigned to 1 person |

### Artifacts

|                                       |                                                              |
| ------------------------------------- | ------------------------------------------------------------ |
| Feature                               | Requirement in User Story format:  “As the user/designer/artist, I want `something testable` so that `explain reason`" |
| Feature List                          | List of features                                             |
| Backlog<br />(Product/Project/Sprint) | Prioritized list of functionality which a product should contain, or a project/sprint should achieve<br />Maintained by Product Owner |
| Tasklist                              | Exhaustive list of tasks                                     |
| Scrumboard                            | Task list visualized based on status like on Trello          |
| Target                                | Goal<br />“Has to be done in 2days”                          |
| Estimation                            | “It’ll take about 2 days to do it”                           |
| Plan                                  | “I’ll need 2 days to do it”                                  |
| Reality                               | “It took 2 days to do it”                                    |

### Meetings

| Meeting               | Goal                                                         | Details                                                      | Timebox |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------- |
| Sprint Planning       | Plan sprint backlog by converting project backlog into tasks |                                                              | 1hr     |
| Daily Scrum/Standups  | Accountability & Communication                               | What did you do since last meeting<br />What will you do until next meeting<br />What is blocking you. Just note this down. Create a new meeting if someone wants to discuss this block | 10min   |
| Sprint Review         | Review product                                               | Demonstrate working product<br />Review & evaluate product<br />Review & update product backlog | 1hr     |
| Retrospective Meeting | Review process                                               | Things to keep doing<br />Things to stop doing<br />New things to try | 30min   |

#### Time-boxing

Every thing should be assigned a particular max-duration

If time gets over, go ahead with the best solution possible rather than wasting more time trying to obtain the “perfect” solution

#### Clear goal & agendas

If you don’t know why exactly you are having a meeting, you shouldn’t be having it

#### Involved participants

If someone does not need to be there, don’t force them to attend

### People

| Person        |                                                              |
| ------------- | ------------------------------------------------------------ |
| Product Owner | Has complete understanding of the product<br />Guides the team |
| Scrum Master  | Administrative<br />Runs the meetings<br />Facilitator<br />Mediator |
| Team Member   |                                                              |

Ideally Product Owner and Scrum master should always be different, as each of their roles is very tiring as is.

## Feature Size

Categorize features as S, M, L

IDK

- Discuss as an entire team whenever possible
- Each role should try to lend its perspective
- Listen & try to make sense of what others are saying
- Offer alternate solutions & point out problems
- Divide huge features into atomic features
- Everyone related to the decision should be there
- Everyone should agree on the feature size
  - Don’t average out the opinions if someone says S & another person says L, don’t conclude it’s M
- Should not take too long. If you get stuck on a feature, put it aside & come back to it

## Estimation

Get a clear view of project reality to make informed decisions to control project to hit targets. You need to identify what the team is capable of, not what you can/want to happen

```mermaid
---
title: Estimation
---

flowchart LR

subgraph pb[Product Backlog]
	direction LR
	f1[Feature 1]
	f2[Feature 2]
end

subgraph sb[Sprint Backlog]
	direction LR
	t1[Task 1]
	t2[Task 2]
end


pb --> Size --> Effort --> People --> Time --> sb
```

- Don’t estimate in “ideal work hours”
- Include breaks, distractions & meetings in the estimate

Need to know the __ of your estimates

- Uncertainty
- Accuracy
- Precision

You need to commit to a point estimate; don’t use the average of a range.

### Estimation $\ne$ Planning

|                        | Planning              | Estimation                                                   |
| ---------------------- | --------------------- | ------------------------------------------------------------ |
| Goal                   | Achieve a target      | Verify if plan is realistic                                  |
| Commitment occurs when | Person accepts a task | when estimation is accurate<br />(not necessary person takes up the task) |
| Nature                 | Fixed                 | Variable<br />Re-estimate as you work on the task            |

### Tracking Estimates

| Feature | Task | Original Estimate | Elapsed | Remaining | Current Estimate |
| ------- | ---- | ----------------- | ------- | --------- | ---------------- |
|         |      |                   |         |           |                  |

Get summary statistics of feature’s resources

Use this for better estimating in the future

Compare against feature size

## Backlog to Tasklist

![image-20240127231326318](./assets/image-20240127231326318.png)

By breaking up story into tasks, you can understand all dependencies & potential overloading of individuals

## Scrum Board

|                  |
| ---------------- |
| Doing            |
| Todo-This Sprint |
| Todo-Long Term   |
| Discussion       |
| Testing          |
| Completed        |
| Cancelled        |

![image-20240127232714105](./assets/image-20240127232714105.png)

## References

- https://www.youtube.com/playlist?list=PLUl4u3cNGP61V4W6yRm1Am5zI94m33dXk
- [IIT Roorkee July 2018 | Project Management](https://www.youtube.com/playlist?list=PLLy_2iUCG87CBuNhvti0h6W54ZmqrSDMJ)