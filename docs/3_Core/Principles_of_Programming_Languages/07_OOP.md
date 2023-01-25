We did an entire course on this, so not much details have been written here. If any doubts, refer to `Sem 3 > OOP` notes

## OOP

Object-Oriented Programming

- Classes are collection of properties/functions
- Objects are instances of that class
- Subclass, Superclass
- Method: Procedure body implementing operation
- Message: Procedure call; request to execute method

## Purpose of OOP

- Data abstraction
- Data encapsulation
- Data hiding
- Polymorphism
- Inheritance
  `is-a` relationships

## Classes

- Class declaration and definition
- Access specifiers
- Constructors
- Destructors (`~Class()`)

### Binary scope resolution operator

used for defining member function outside class

```c++
return_type ClassName::function_name()
{
  
}

// constructor
Line::Line() {
   ;
}

// destructor
Line::~Line() {
   ;
}

// regular functions
void Line::setLength() {
   ;
}
```

## Inheritance

- Base/Parent/Super class
- Derived/Child/Sub class

### Purpose

It allows a class to __ properties/functions of another class

- Re-use
- Extend
- Modify

```c++
class derived_class:access_specifier base_class {
  
};

class Rectangle: public Shape {
   public:
      int func() { 
        ;
      }
};
```

### Access Specifiers

- Public
- Protected
- Private

### Types

- Single
- Multiple
- Multi-Level
- Hierarchical

## Polymorphism

### Function Overloading

based on different argument list

### Function overriding

Derived class function definition overrides parent class definition

### Operator Overloading

Overloaded operators are functions with special names.

```c++
class className {
  public:
    returnType operator symbol (arguments) {
    	;
    } 
};
```

```c++
class Person {
  int age;
  Person()
  {
    age = 0;
  }
  void operator ++ () {
    ++age;
  }
};

void main() {
  Person p;

  ++p;
  // Calls ++ ()" that I defined
}
```

Following operators cannot be overloaded

- $::$
- $.*$
- $?:$

### Virtual Function

Declared using `virtual` in base class

Member function of base class that is overriden by derived class

When you refer to a derived class object using a pointer or a reference to the base class, you can call a virtual function for that object and execute the derived class’s version of the function.

Used to achieve runtime polymorphism

Useful when you want a function to exist in the base class, but have no meaningful definition in the base class.

## `mutable`

… is used for making data member of a object that is declared as a constant, to be changeable

### Explanation

If we have an object that is declared as a constant, then by default, all the data members of this object now become constant. However, if you want one/more specific data members to be changeable, use `mutable` next to the data member in the class defintion.

```c++
class Student
{
  public:
  	x = 0;
  	mutable y = 0;
};

void main()
{
  const Student s;

  s.x = 10; // not possible
  s.y = 10; // possible due to mutable keyword
}
```

## Order of invokation of nested objects

1. Nested object constructor
2. Outer object constructor
3. Outer object destructor
4. Nested object destructor