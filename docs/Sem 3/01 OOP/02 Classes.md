## Class

collection of data and related functions into a single entity

contains

- fields/properties - variables
- methods - functions
  - constructor
  - custom
- nested classes

Naming convention is TitleCase

## Object

Instance of a class

declared using `new` keyword

. operator is called object reference operator / relationship operator

``` java
Classname var = new Constructor();
```

``` java
class Student
{
  String name;
  int age;
  Student(String aName, int aAge) // constructor
  {
    name = aName;
    age = aAge;
  }
  void display()
  {
    System.out.println(name + age); // abc10
  }
}
public class StudentTester
{
  public static void main(String args[])
  {
		Student s1 = new Student("abc", 10);
		s1.display(); 
  }
}
```

## `instanceof ` operator

Checks if an object belongs to a particular class

returns Boolean true/false

Syntax: `(object reference var) instanceof (class/interface type)`

``` java
boolean result = varName instanceof String;
boolean result = varName instanceof CustomClass;
```

## Object reference

Just assigning one object name to another object name just assigns the pointer location; doesn't copy the data over

``` java
Student s1 = new Student("abc" , 10);
s1.display(); // abc10

Student s2 = s1;

s2.setAge(20); 
s1.display(); // abc20
```

## Constructor

function that gets invoked during object creation

no return type
Not even void, as constructor kinda returns object of class

If the formal and actual parameter have the same name, then it will output the default values

- null for String
- 0 for int
- 0.0f for float

### Types of constructors

- default constructor (by compiler)
- Non-parameterized constructor
- parameterized contructor
- copy constructor

### Copy Constructor

Truly copy data from one object to another

``` java
class Student
{
  private int year;
  Student(int year)
  {
    this.year = year;
  }
  Fruit(Student source)
  {
    this.year = source.year;
  }
}
class StudentTester
{
  public static void main()
  {
    Student s1 = new Student(2020);
    Student s2 = new Student(s1); // will have the same values of s1
  }
}
```

## Constructor Overloading

multiple constructors having the same name but different functionality. They differ in their function signature

## Custom Print Message

In order to get a custom output for `System.out.println(objectName)`, we can create a custom `public String toString()` for the class.

``` java
class Student
{
  int roll;
  String name;
  
  public String toString()
  {
    String text = "Name is " + this.name + " Roll no is " + this.roll;
    return text;
  }
}
```

## this Keyword

### refer to current object

useful when the actual and formal parameter have the same name

``` java
class Student
{
  int rno;
  String name;
  
	Student(int rno, String name)
	{
  	this.rno = rno;
	  this.name = name;
	} 
}
```

### invoke current class method

``` java
this.m();
//equivalent to
m(); // compiler adds this.
```

### invoke current class constructor

the invoked contructor should have already been defined
useful for chaining of constructors to avoid redundancy

**Note: ** `this()` cannot be at the end

```java
class Student
{
  int rno;
  String name;
  boolean student;
  float fee;
  
  Student()
  {
		student = true;
  }
  
  Student(int rno, String name)
  {
		this(); // calls Student()
    this.rno = rno;
    this.name = name;
    // this(); here will give error
    // Thanks Firas
  }
  
	Student(int rno, String name, float fee)
	{
  	this(rno, name); // calls Student(int rno)
    this.fee = fee;
	} 
}
```

### pass current obj as argument in method call

``` java
class Student
{
	void display(Student s1)
  {
    System.out.println("blah");
  }
  void m()
  {
		display(this);
  }
}
```

### passed as argument in constructor call

``` java
class Student
{
  School sch;
	Student(School sch)
  {
    this.obj = obj;
  }
  void display()
  {
    System.out.println(sch.city);
  }
}
class School
{
  int year = 2000;
  String city = "Dubai";
  School()
  {
    Student s1 = new Student(this);
  }
}
```

### return current object

``` java
class Student
{
  Student getStudent()
  {
    return this;
  }
  void msg()
  {
    System.out.println("hello");
  }
}
class main
{
  public static void main(String args[])
  {
    new Student().getStudent().msg();
    //equivalent to
    new Student().msg();
  }
}
```

## Access Modifier

| Accessibility             | `private` | `default` | `protected` | `public` |
| ------------------------- | :-------: | :-------: | :---------: | :------: |
| Same class                |     Y     |     Y     |      Y      |    Y     |
| Same package subclass     |     N     |     Y     |      Y      |    Y     |
| Same package non-subclass |     N     |     Y     |      Y      |    Y     |
| Diff package subclass     |     N     |     N     |      Y      |    Y     |
| Diff package non-subclass |     N     |     N     |      N      |    Y     |

`private` does not get inherited hence not accessible even in subclass(child class); it is only accessible in the same class/nested class

## Other Types

### Nested Classes

is a class that is inside a function or another class. 

Nested Inner Class can access any private instance of outer class

```java
Outer.Inner obj = new Outer().new Inner();

//or

Outer objo = new Outer();
Outer.Inner obji = objo.new Inter();

//or

// create a function in the outer class that creates objects of outer class
```

### Anonymous Classes

class that does one or more of the following, without even creating the class

- implements an interface
- inherits a class

```java
Student s = new HelloWorld()
{
  int x = 5;
  public void func()
  {
    x++;
  }
};
```

the entire  statement ends with `;` (just like any other java statement)
