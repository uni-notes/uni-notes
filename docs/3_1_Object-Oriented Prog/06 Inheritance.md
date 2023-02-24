## Encapsulation

put data and related functions in a single capsule, using classes and access specifiers

helps in data-hiding

default is the default access specifier

- trying to call a private member incorrectly will give a ==run-time error==
- if a class has private constructor, then you cannot create the object of the class from outside that class; this will be a ==compile-time error==

if a class/interface is of private, then you cannot access that class at all; that’s why don’t pick this option

## Inheritance

Inheritance creates an ==is-a== relation

Constructors are not inherited

`extends` keyword is used for inheritance in Java

Base class constructor==s== are called before the current class constructor

``` java
public class Derived extends Base //public inheritance
{
  
}

Base obj = new Derived(); // will have characteristic of the Derived class
// this is like what we did for
// - String and Object
// - List and ArrayList/LinkedList
```

## Types of Inheritance

java does not support multiple inheritance using classes

we need to use interfaces

``` mermaid
graph TD

subgraph Single
a --- b
end

subgraph Multi-level
c --- d --- e
end

subgraph Hierarchical
f --- g & h & i
g --- j & k
end

subgraph Multiple
x & y --- z
end
```

## `super` keyword

refer to the base class

1. `super()` base class constructor
   1. I guess this is why classes don’t support multiple inheritance
   2. cuz Java won’t know which base class to refer to when using `super()`
2. `super.var` base class property
3. `super.func()` base class function

**not** valid - `super.super(), super.super.method(), super.super.var`

## Function Overriding

Base and derived classes have a function with the same name, but with different functionality

private, static and final methods cannot be over-ridden

doubt: private methods won’t even be inherited, so it’s not considered as over-riding, right?

``` java
class Derived extends Base
{
  void function1() // run-time binding
  {
    
  }
  @Override
  void function2() // compile-time binding
  {
    
  }
}
```

Compile-time overriding is better for performance and bug prevention

- if you keep everything as runtime, execution of program will be slow
- Therefore, it is better to make everything as compile-time

## Abstract Method

Method that only has function prototype (declared, but not defined)

``` java
public abstract void func();
```

## Abstract Class

conceptual class which acts a bridge bw class and interface

Abstract class is a class containing abstract method

- can be inherited
- can contain constructor
    - called when creating objects of **child** classes
    - we cannot create objects of the abstract class itself
- can also have final methods

``` java
abstract class Vehicle
{
  Vehicle()
  {
  	System.out.println("This comes under Vehicle class"); // gets printed when creating object of any Vehicle subclasses
  }
  abstract public void sound();
  
  String name; // gets inherited for all Vehicle subclasses
  public String getName() 
  {
    return name;
  }
}

class Car extends Vehicle
{ 
  public void sound() // NOT OVERRIDING, as sound was just an abstract method in base class
  {
    System.out.println("Woof");
  }
}

Car c = new Car();
```

## Interface

all types of inheritances are possible using interfaces. Methods from interface cannot use `protected`

In new versions, we can have

- `static/default` methods in interfaces with their definition also
- `private ` methods

### Automatic

1. variables are automatically
   1. public
   2. static
   3. final
2. all functions are automatically
   1. public
   2. abstract
3. functions can only have prototype - declared, but not defined

### Conditions
- A class that `implements ` an interface must have function definition for ==all== the functions of the interface (and extended interfaces)
- a single interface can extend multiple interfaces (multiple inheritance)
- can have `default` methods
- can have `static` methods
    - can be called just by using the interface name

We use both `extends` and `implements` here

``` java
interface i1
{
  void f1();
}

interface i2
{
  void f2();
}

interface Vehicle extends i1, i2
{
  void f();
}
  
class Bicycle implements Vehicle
{
  void f1(){;}
  void f2(){;}
  void f(){;}
}

// array of instruments of various data types
Instrument[] orchestra = {
  new Wind(),
  new Percussion(),
  new Brass()
};
```

## Abstract vs Interfaces

|                      | Abstract               | Interface             |
| -------------------- | ---------------------- | --------------------- |
| Supported methods    | abstract, concrete     | abstract              |
| Supported variables  | all types of variables | only static and final |
| multiple inheritance | N                      | Y                     |
| `extends`            | only classes           | only interfaces       |
| `implements`         | Y                      | N                     |
| creation of objects  | N                      | N                     |
| member access        | any                    | only public           |