## Topics

1. Stacks
2. Queues
3. Linked List and Pointers

## Linear Data Structure

group of elements accessed in a sequential order

### Types

|                             | Primitive                      | Non-Primitive                  |
| --------------------------- | ------------------------------ | ------------------------------ |
| Definition                  | Basic data types of a language | complex collection of elements |
| can be broken down further? | ✅                              | ❌                              |
| Example                     | int, float, void, char         | arrays, structures, classes    |

## Stacks

eg: people in a list

**real life application:** calculators use stacks for processing

- is a collection of elements
- linear data structures
- index data structures
  - each element has its own index value referring to its position
- can be implemented as a 
  - static data structure, using array
  - dynamic data structure, using linked list

Works with the principle of LIFO(Last in First out)

only 2 processes

- push - insertion
- pop - deletion

### top

is a variable used to manipulate a stack

- when push operation takes place, top is assigned to top+1
- when operation takes place, top is assigned to top-1

Different