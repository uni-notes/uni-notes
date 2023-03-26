Object Oriented design must be

- specific to the problem, and
- general to adress future problems and requirements

## Design Patterns

> Descriptions of communicating objects and classes, that are ==customized== to solve a general design problem, in a particular context.

allows

- re-usability of design
- faster production of projects
- more accessible
- easier documentation and maintenance

### Parts

1. Pattern Name
2. Problem
3. Solution
4. Context
5. Class Diagram

## Types of Patterns

1. Creational Patterns
   deal with object creation
   1. Singleton
      Single object of a class is created, and all other objects can access it globally
2. Structural Patterns
   deal with relationship between entities
   1. Composite
      when you put components inside containers
      eg: JPanel
   2. Decorator
      something that *surrounds* component
      eg: scroll bars
   3. Adapter Pattern
      a middle interface ‘adapts’ main interface based on the requirement
   4. Proxy Pattern
      a class acts a proxy to access another class, to keep that hidden
3. Behavioural Patterns
   communication between objects
   1. Iterator
      access the elements of an aggregate object sequentially without exposing its underlying implementation
   1. Observer
      eg: ActionListener
