## Phases of Software Engineering

```mermaid
flowchart LR
a[Requirement Gathering] -->
b[Requirement Analysis] -->
d["Design<br />(UML Diagrams)"] -->
Coding -->
Testing -->
Deployment -->
Support
```

## UML

Unified Modeling Language

## Use Case Diagram

### Actors

classes

could be

1. users of the system
2. external system
3. physical environment

Properties

1. unique name
2. description (optional)

### Use Cases

basically like functions

properties

1. unique name
2. participating actors
3. entry conditions
4. exit conditions
5. event flow
6. exceptional cases

### `<<extends>>`

for exceptions/showing use cases that are rarely used

The direction of a `<<extends>>` relationship is to the extended use case

### `<<includes>>`

for use cases that require/depend on another use case

The direction of a `<<includes>>` relationship is to the using use case (unlike `<<extends>>` relationships).

![](img/useCase.svg){ loading=lazy }

## Class Diagram

### Access Specifiers

- (nothing) default
- `-` private
- `+` public
- `#` protected

### Connections

- association
    - can be 1-way or 2-way
    - can be one-one or many-many
    $1-1, \quad 5\ldots* - *, \quad *- 3\ldots *, \quad * - *$
    - arrow from a towards b, means that a depends on b
  
- aggregation
- composition (strong aggregation)
- inheritance
    - class inheritance
    - interface inheritance
### Example

``` mermaid
classDiagram
Interface <|.. Base: implements
Base <|-- Teacher
Base <|-- Student
Base <.. Tester
ClassRoom *-- Teacher
ClassRoom o-- Student
class Interface {
	<<interface>>
	+func() void
}
class Base {
	<<abstract>>
	#var : int
	+func() void
}
class Tester {
	+main() void
}
```

## Sequence Diagram

Shows the interactions bw the classes/objects

calls are solid, returns are dashed

```mermaid
sequenceDiagram
autonumber

activate d

d ->>+ p: Discharge Advice

activate r

p ->>- r: Discharge Request

r ->> + pd: Check Details
pd -->> - r: Detailed Summary
r ->> r:  Prepare Bill
r -->>+ p: Send Bill
p ->>- r: Request Discount
r ->> d: Check Possibility of discount
d -->> r: Approve Discount

deactivate d

r ->> r: Update bill
r ->>+ p: Send bill
p -->>- r: Pay bill
r ->> +pd: Update Bill
deactivate pd

activate p
r ->> p: Discharge note

deactivate r

p ->> + Ward: Show note
Ward -->> - p: Discharge
deactivate p

%% participant
participant d as Doctor
participant p as Patient
participant r as Reception
participant pd as Patient Database
```

## State Diagrams

```mermaid
stateDiagram-v2
[*] --> dr
dr --> cp: Patient Register
cp --> er: Resign
er --> [*]

state "Doctor Registration" as dr
state "Check Patient" as cp
state "End Service" as er
```

