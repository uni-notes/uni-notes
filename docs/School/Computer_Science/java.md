# Java Notes - Basics

## Hello World

1. Print Line: 
System.out.println() can print to the console:
System is a class from the core library provided by Java
out is an object that controls the output
println() is a method associated with that object that receives a single argument

    ```
    System.out.println("Hello, world!");
    // Output: Hello, world!
    ```

2. Comments: Comments are bits of text that are ignored by the compiler. They are used to increase the readability of a program.
Single line comments are created by using // . Multi-line comments are created by starting with /* and ending with */ .

    ```
    // I am a single line comment!
    /*
    And I am a
    multi-line comment!
    */
    ```

3. main() Method : In Java, every application must contain a main() method, which is the entry point for
the application. All other methods are invoked from the main() method.
The signature of the method is public static void main(String[] args) { } . It accepts a single argument: an array of elements of type String. 

    ```
    public class Person {
    public static void main(String[]
    args) {
        System.out.println("Hello,
    world!"); }
    }
    ```

4. Classes: A class represents a single concept. A Java program must have one class whose name is the same as the program filename. In the example, the Person class must be declared in a program file named Person.java.
    ```
    public class Person {
    public static void main(String[]
    args) {
        System.out.println("I am a person,
    not a computer.");
    }
    }
    ```

5. Compiling Java: In Java, when we compile a program, each individual class is converted into a .class file, which is known as byte code.
The JVM (Java virtual machine) is used to run the byte code.

    ```
    # Compile the class file:
    javac hello.java
    # Execute the compiled file:
    java hello
    ```

6. Whitespace: Whitespace, including spaces and newlines, between statements is ignored.

7. Statements: In Java, a statement is a line of code that executes a task and is terminated with a ";". 

    ```
    System.out.println("Java Programming");
    ```

## Variables

1. Boolean Data Type: In Java, the boolean primitive data type is used to store a value, which can be either
true or false. 

    ```
    boolean result = true;
    boolean isMarried = false;
    ```

2. Strings: A String in Java is a Object that holds multiple characters. It is not a primitive datatype. A String can be created by placing characters between a pair of double quotes ( " ). To compare Strings, the equals() method must be used instead of the primitive equality comparator == .

    ```
    // Creating a String variable
    String name = "Bob";

    // The following will print "false"
    because strings are case-sensitive

    System.out.println(name.equals("bob"));
    ```

3. int Data Type: In Java, the int datatype is used to store integer values. This means that it can store all positive and negative whole numbers and zero.

    ```
    int num1 = 10;
    int num2 = -5;
    int num3 = 0;
    int num4 = 12.5; // not allowed
    ```

4. char Data Type: In Java, char is used to store a single character. The character must be enclosed in single quotes.

    ```
    char answer = 'y';
    ```

5. Primitive Data Types: Java’s most basic data types are known as primitive data types and are in the system by default.The available types are as follows:
    - int
    - char
    - boolean
    - byte
    - long
    - short
    - double
    - float

null is another, but it can only ever store the value null.

    int age = 28;
    char grade = 'A';
    boolean late = true;
    byte b = 20;
    long num1 = 1234567;
    short no = 10;
    float k = (float)12.5;
    double pi = 3.14;

6. Static Typing: In Java, the type of a variable is checked at compile time. This is known as static typing. It has the advantage of catching the errors at compile time rather than at execution time. Variables must be declared with the appropriate data type or the program will not compile.

    ```
    int i = 10;
    char ch = 'a';
    j=20;
    no type is given
    char name = "Lil"; // won't compile, wrong data type
    ```

7. final Keyword: The value of a variable cannot be changed if the variable was declared using the final keyword. Note that the variable must be given a value when it is declared as final . final variables cannot be changed; any attempts at doing so will result in an error message.

    ```
    // Value cannot be changed:
    final double PI = 3.14;
    ```

8. double Data Type: The double primitive type is used to hold decimal values.

    ```
    double PI = 3.14;
    double price = 5.75;
    ```

9. Math Operations: Basic math operations can be applied to int, double and float data types:

    ```
    int a = 20;
    int b = 10;
    int result; result=a+b; //30 result=a-b; //10 result=a*b; //200 result=a/b; //2 result=a%b; //0
    ```

10. Comparison Operators: Comparison operators can be used to compare two values. They are supported for primitive data types and the result of a comparison is a boolean value
true or false.

    ```
    int a = 5;
    int b = 3;
    boolean result = a > b;
    // result now holds the boolean value
    true
    ```

11. Compound Assignment Operators: Compound assignment operators can be used to change and reassign the value of a variable using one line of code. Compound assignment operators include += , -= , *= , /= , and
%=.

    ```
    int number = 5;
    number += 3; // Value is now 8
    number -= 4; // Value is now 4
    number *= 6; // Value is now 24
    number /= 2; // Value is now 12
    number %= 7; // Value is now 5
    ```

12. Increment and Decrement Operators: The increment operator, ( ++ ), can increase the value of a number-based variable by 1 while the decrement operator, ( -- ), can decrease the value of a variable by 1.

```
int numApples = 5;
numApples++; // Value is now 6
int numOranges = 5;
numOranges--; // Value is now 4
```

13. Order of Operations: The order in which an expression with multiple operators is evaluated is determined by the order of operations: parentheses -> multiplication -> division -> modulo -> addition - > subtraction.

## Java Objects’ State and Behavior:

1. Java Instance: Java instances are objects that are based on classes. For example, Bob may be an instance of the class Person. Every instance has access to its own set of variables which are known as instance fields, which are variables declared within the scope of the instance. Values for instance fields are assigned within the constructor method.

    ```
    public class Person {
    int age;
    String name;
    
    // Constructor method
    public Person(int age, String name) {
        this.age = age;
        this.name = name;
    }
    
    public static void main(String[] args) {
        Person Bob = new Person(31, "Bob");
        Person Alice = new Person(27, "Alice");
    }
    }
    ```

2. Java Dot Notation: In Java programming language, we use . to access the variables and methods of an object or a Class. This is known as dot notation and the structure looks like this: 
    ```
    public class Person {
    int age;
    
    public static void main(String [] args) {
        Person p = new Person();
        
        // here we use dot notation to set age
        p.age = 20; 
        
        // here we use dot notation to access age and print
        System.out.println("Age is " + p.age);
        // Output: Age is 20
    }
    }
    ```
3. Constructor Method in Java: Java classes contain a constructor method which is used to create instances of the class. The constructor is named after the class. If no constructor is defined, a default empty constructor is used.

    ```
    public class Maths {
    public Maths() {
        System.out.println("I am constructor");
    }
    public static void main(String [] args) {
        System.out.println("I am main");
        Maths obj1 = new Maths();
    }
    }
    ```

4. Creating a new Class instance in Java: In Java, we use the new keyword followed by a call to the class constructor in order to create a new instance of a class. The constructor can be used to provide initial values to instance fields.

    ```
    public class Person {
    int age;
    // Constructor:
    public Person(int a) {
        age = a;
    }
    
    public static void main(String [] args) {
        // Here, we create a new instance of the Person class:
        Person p = new Person(20);
        System.out.println("Age is " + p.age); // Prints: Age is 20
    }
    }
    ```

5. Reference Data Types: A variable with a reference data type has a value that references the memory address of an instance. During variable declaration, the class name is used as the variable’s type.

    ```
    public class Cat {
    public Cat() {
        // instructions for creating a Cat instance
    }  

    public static void main(String[] args) {
        // garfield is declared with reference data type `Cat` 
        Cat garfield = new Cat();
        System.out.println(garfield); // Prints: Cat@76ed5528
    }
    }
    ```

6. Constructor Signatures: A class can contain multiple constructors as long as they have different parameter values. A signature helps the compiler differentiate between the different constructors. A signature is made up of the constructor’s name and a list of its parameters.

    ```
    // The signature is `Cat(String furLength, boolean hasClaws)`.
    public class Cat {
    String furType;
    boolean containsClaws;

    public Cat(String furLength, boolean hasClaws) {
        furType = furLength;
        containsClaws = hasClaws;
    }
    public static void main(String[] args) {
        Cat garfield = new Cat("Long-hair", true);
    }
    }
    ```

7. null Values: null is a special value that denotes that an object has a void reference.
    ```
    public class Bear {
    String species;
    public Bear(String speciesOfBear;) {
        species = speciesOfBear;
    }
    
    public static void main(String[] args) {
        Bear baloo = new Bear("Sloth bear"); 
        System.out.println(baloo); // Prints: Bear@4517d9a3
        // set object to null
        baloo = null;
        System.out.println(baloo); // Prints: null
    }
    }
    ```

8. The Body of a Java Method: In Java, we use curly brackets {} to enclose the body of a method. The statements written inside the {} are executed when a method is called.

    ```
    public class Maths {
    public static void sum(int a, int b) { // Start of sum
        int result = a + b;
        System.out.println("Sum is " + result);
    } // End of sum
    
    
    public static void main(String [] args) {
        // Here, we call the sum method
        sum(10, 20);
        // Output: Sum is 30
    }
    }
    ```

9. Method Parameters in Java: In java, parameters are declared in a method definition. The parameters act as variables inside the method and hold the value that was passed in. They can be used inside a method for printing or calculation purposes. In the example, a and b are two parameters which, when the method is called, hold the value 10 and 20 respectively.

    ```
    public class Maths {
    public int sum(int a, int b) {
        int k = a + b;
        return k;
    }
    
    public static void main(String [] args) {
        Maths m = new Maths();
        int result = m.sum(10, 20);
        System.out.println("sum is " + result);
        // prints - sum is 30
    }
    }
    ```

10. Java Variables Inside a Method: Java variables defined inside a method cannot be used outside the scope of that method.

    ```
    //For example, `i` and `j` variables are available in the `main` method only:

    public class Maths {
    public static void main(String [] args) {
        int i, j;
        System.out.println("These two variables are available in main method only");
    }
    }
    ```

11. Returning Info from a Java Method: A Java method can return any value that can be saved in a variable. The value returned must match with the return type specified in the method signature. The value is returned using the return keyword.

    ```
    public class Maths { 
    
    // return type is int
    public int sum(int a, int b) {
        int k;
        k = a + b;
        
        // sum is returned using the return keyword
        return k;
    }
    
    public static void main(String [] args) {
        Maths m = new Maths();
        int result;
        result = m.sum(10, 20);
        System.out.println("Sum is " + result);
        // Output: Sum is 30
    }
    }
    ```

12. Declaring a Method: Method declarations should define the following method information: scope (private or public), return type, method name, and any parameters it receives.

    ```
    // Here is a public method named sum whose return type is int and has two int parameters a and b. 

    public int sum(int a, int b) {
    return(a + b);
    }
    ```

## Conditionals and Control Flow: 

1. else Statement: The else statement executes a block of code when the condition inside the if statement is false. The else statement is always the last condition.

    ```
    boolean condition1 = false;

    if (condition1){
        System.out.println("condition1 is true");
    }
    else{
        System.out.println("condition1 is not true");
    }
    // Prints: condition1 is not true
    ```

2. else if Statements: else-if statements can be chained together to check multiple conditions. Once a condition is true, a code block will be executed and the conditional statement will be exited. There can be multiple else-if statements in a single conditional statement.

    ```
    int testScore = 76;
    char grade;

    if (testScore >= 90) {
    grade = 'A';
    } else if (testScore >= 80) {
    grade = 'B';
    } else if (testScore >= 70) {
    grade = 'C';
    } else if (testScore >= 60) {
    grade = 'D';
    } else {
    grade = 'F';
    }

    System.out.println("Grade: " + grade); // Prints: C
    ```

3. if Statement: An if statement executes a block of code when a specified boolean expression is evaluated as true.

    ```
    if (true) {
        System.out.println("This code executes");
    }
    // Prints: This code executes

    if (false) {
        System.out.println("This code does not execute");
    }
    // There is no output for the above statement
    ```

4. Nested Conditional Statements: A nested conditional statement is a conditional statement nested inside of another conditional statement. The outer conditional statement is evaluated first; if the condition is true, then the nested conditional statement will be evaluated.

    ```
    boolean studied = true;
    boolean wellRested = true;

    if (wellRested) {
    System.out.println("Best of luck today!");  
    if (studied) {
        System.out.println("You are prepared for your exam!");
    } else {
        System.out.println("Study before your exam!");
    }
    }

    // Prints: Best of luck today!
    // Prints: You are prepared for your exam!
    ```

5. AND Operator: The AND logical operator is represented by &&. This operator returns true if the boolean expressions on both sides of the operator are true; otherwise, it returns false.

    ```
    System.out.println(true && true); // Prints: true
    System.out.println(true && false); // Prints: false
    System.out.println(false && true); // Prints: false
    System.out.println(false && false); // Prints: false
    ```

6. NOT Operator: The NOT logical operator is represented by !. This operator negates the value of a boolean expression.

    ```
    boolean a = true;
    System.out.println(!a); // Prints: false

    System.out.println(!false) // Prints: true
    ```

7. The OR Operator: The logical OR operator is represented by ||. This operator will return true if at least one of the boolean expressions being compared has a true value; otherwise, it will return false.

    ```
    System.out.println(true || true); // Prints: true
    System.out.println(true || false); // Prints: true
    System.out.println(false || true); // Prints: true
    System.out.println(false || false); // Prints: false
    ```

8. Conditional Operators - Order of Evaluation: If an expression contains multiple conditional operators, the order of evaluation is as follows: Expressions in parentheses -> NOT -> AND -> OR.

    ```
    boolean foo = true && (!false || true); // true
    /* 
    (!false || true) is evaluated first because it is contained within parentheses. 

    Then !false is evaluated as true because it uses the NOT operator. 

    Next, (true || true) is evaluation as true. 

    Finally, true && true is evaluated as true meaning foo is true. */
    ```

## Arrays and ArrayLists: 

1. Index: An index refers to an element’s position within an array. The index of an array starts from 0 and goes up to one less than the total length of the array.

    ```
    int[] marks = {50, 55, 60, 70, 80};

    System.out.println(marks[0]);
    // Output: 50

    System.out.println(marks[4]);
    // Output: 80
    ```

2. Arrays: In Java, an array is used to store a list of elements of the same datatype. Arrays are fixed in size and their elements are ordered.

    ```
    // Create an array of 5 int elements
    int[] marks = {10, 20, 30, 40, 50};
    ```

3. Array Creation in Java: In Java, an array can be created in the following ways:

- Using the {} notation, by adding each element all at once.
- Using the new keyword, and assigning each position of the array individually.

    ```
    int[] age = {20, 21, 30};

    int[] marks = new int[3];
    marks[0] = 50; 
    marks[1] = 70;
    marks[2] = 93;
    ```

4. Changing an Element Value: To change an element value, select the element via its index and use the assignment operator to set a new value.

    ```
    int[] nums = {1, 2, 0, 4};
    // Change value at index 2
    nums[2] = 3;
    ```

5. Java ArrayList: In Java, an ArrayList is used to represent a dynamic list. While Java arrays are fixed in size (the size cannot be modified), an ArrayList allows flexibility by being able to both add and remove elements.

    ```
    // import the ArrayList package
    import java.util.ArrayList;

    // create an ArrayList called students
    ArrayList<String> students = new ArrayList<String>();
    ```

6. Modifying ArrayLists in Java: An ArrayList can easily be modified using built in methods.
- To add elements to an ArrayList, you use the add() method. The element that you want to add goes inside of the ().
- To remove elements from an ArrayList, you use the remove() method. Inside the () you can specify the index of the element that you want to remove. Alternatively, you can specify directly the element that you want to remove.

    ```
    import java.util.ArrayList;

    public class Students {
    public static void main(String[] args) {
        
        // create an ArrayList called studentList, which initially holds []
            ArrayList<String> studentList = new ArrayList<String>();
        
        // add students to the ArrayList
        studentList.add("John");
        studentList.add("Lily");
        studentList.add("Samantha");
        studentList.add("Tony");
        
        // remove John from the ArrayList, then Lily
        studentList.remove(0);
        studentList.remove("Lily");
        
        // studentList now holds [Samantha, Tony]
        
        System.out.println(studentList);
    }
    }
    ```

## Loops: 

1. For-Each Statement in Java: In Java, the for-each statement allows you to directly loop through each item in an array or ArrayList and perform some action with each item.When creating a for-each statement, you must include the for keyword and two expressions inside of parentheses, separated by a colon. These include:
- The handle for an element we’re currently iterating over.
- The source array or ArrayList we’re iterating over.

    ```
    // array of numbers
    int[] numbers = {1, 2, 3, 4, 5};

    // for-each loop that prints each number in numbers
    // int num is the handle while numbers is the source array
    for (int num : numbers) {  
        System.out.println(num);
    }
    ```

## String Methods: 

1. length() String Method in Java: In Java, the length() string method returns the total number of characters – the length – of a String.

    ```
    String str = "Codecademy";  

    System.out.println(str.length());
    // prints 10
    ```

2. concat() String Method in Java: In Java, the concat() string method is used to append one String to the end of another String. This method returns a String representing the text of the combined strings.

    ```
    String s1 = "Hello";
    String s2 = " World!";

    String s3 = s1.concat(s2);
    // concatenates strings s1 and s2

    System.out.println(s3);
    // prints "Hello World!"
    ```

3. String Method equals() in Java: In Java, the equals() string method tests for equality between two Strings.

- equals() compares the contents of each String. If all of the characters between the two match, the method returns true. If any of the characters do not match, it returns false.
- Additionally, if you want to compare two strings without considering upper/lower cases, you can use .equalsIgnoreCase().

    ```
    String s1 = "Hello";
    String s2 = "World";

    System.out.println(s1.equals("Hello"));
    // prints true

    System.out.println(s2.equals("Hello"));
    // prints false 

    System.out.println(s2.equalsIgnoreCase("world"));
    // prints true 
    ```

4. indexOf() String Method in Java: In Java, the indexOf() string method returns the first occurence of a character or a substring in a String. The character/substring that you want to find the index of goes inside of the (). If indexOf() cannot find the character or substring, it will return -1.

    ```
    String str = "Hello World!";

    System.out.println(str.indexOf("l"));
    // prints 2

    System.out.println(str.indexOf("Wor"));
    // prints 6

    System.out.println(str.indexOf("z"));
    // prints -1
    ```

5. charAt() String Method in Java: In Java, the charAt() string method returns the character of a String at a specified index. The index value is passed inside of the (), and should lie between 0 and length()-1.

    ```
    String str = "This is a string";

    System.out.println(str.charAt(0));
    // prints 'T'

    System.out.println(str.charAt(15));
    // prints 'g'
    ```

6. toUpperCase() and toLowerCase() String Methods: In Java, we can easily convert a String to upper and lower case with the help of a few string methods:
- toUpperCase() returns the string value converted to uppercase.
- toLowerCase() returns the string value converted to lowercase.

    ```
    String str = "Hello World!";

    String uppercase = str.toUpperCase();
    // uppercase = "HELLO WORLD!"

    String lowercase = str.toLowerCase();
    // lowercase = "hello world!"
    ```

## Access, Encapsulation, and Static Methods: 

1. The Private Keyword: In Java, instance variables are encapsulated by using the private keyword. This prevents other classes from directly accessing these variables.

    ```
    public class CheckingAccount{
    // Three private instance variables
    private String name;
    private int balance;
    private String id;
    }
    ```

2. Accessor Methods: In Java, accessor methods return the value of a private variable. This gives other classes access to that value stored in that variable. without having direct access to the variable itself. Accessor methods take no parameters and have a return type that matches the type of the variable they are accessing.

    ```
    public class CheckingAccount{
    private int balance;
  
        //An accessor method
        public int getBalance()
        {
        return this.balance;
        }
    }
    ```

3. Mutator Methods: In Java, mutator methods reset the value of a private variable. This gives other classes the ability to modify the value stored in that variable without having direct access to the variable itself.

- Mutator methods take one parameter whose type matches the type of the variable it is modifying. 
- Mutator methods usually don’t return anything.

    ```
    public class CheckingAccount{
    private int balance;
    
    //A mutator method
    public void setBalance(int newBalance){
        this.balance = newBalance;
    }
    }
    ```

4. Local Variables: In Java, local variables can only be used within the scope that they were defined in. This scope is often defined by a set of curly brackets. Variables can’t be used outside of those brackets

    ```
    public void exampleMethod(int exampleVariable){
    // exampleVariable can only be used inside these curly brackets.
    }
    ```

5. The this Keyword with Variables: In Java, the this keyword can be used to designate the difference between instance variables and local variables. Variables with this. reference an instance variable.

    ```
    public class Dog{
    public String name;

    public void speak(String name){
        // Prints the instance variable named name
        System.out.println(this.name);

        // Prints the local variable named name
        System.out.println(name);
    }
    }
    ```

6. The this Keyword with Methods: In Java, the this keyword can be used to call methods when writing classes.

    ```
    public class ExampleClass{
    public void exampleMethodOne(){
        System.out.println("Hello");
    }

    public void exampleMethodTwo(){
        //Calling a method using this.
        this.exampleMethodOne();
        System.out.println("There");
    }
    }
    ```
7. Static Methods: Static methods are methods that can be called within a program without creating an object of the class.

    ```
    // static method
    public static int getTotal(int a, int b) {
    return a + b;
    }

    public static void main(String[] args) {
    int x = 3;
    int y = 2;
    System.out.println(getTotal(x,y)); // Prints: 5
    }
    ```

8. Calling a Static Method: Static methods can be called by appending the dot operator to a class name followed by the name of the method.

    ```
    int largerNumber = Math.max(3, 10); // Call static method
    System.out.println(largerNumber); // Prints: 10
    ```

9. The Math Class: The Math class (which is part of the java.lang package) contains a variety of static methods that can be used to perform numerical calculations.

    ```
    System.out.println(Math.abs(-7.0)); // Prints: 7

    System.out.println(Math.pow(5, 3)); // Prints: 125.0

    System.out.println(Math.sqrt(52)); // Prints: 7.211102550927978
    ```

10. The static Keyword: Static methods and variables are declared as static by using the static keyword upon declaration.

    ```
    public class ATM{
    // Static variables
    public static int totalMoney = 0;
    public static int numATMs = 0;

    // A static method
    public static void averageMoney(){
        System.out.println(totalMoney / numATMs);
    }
    }
    ```

11. Static Methods and Variables: Static methods and variables are associated with the class as a whole, not objects of the class. Both are used by using the name of the class followed by the "." operator.

```
public class ATM{
  // Static variables
  public static int totalMoney = 0;
  public static int numATMs = 0;

  // A static method
  public static void averageMoney(){
    System.out.println(totalMoney / numATMs);
  }

  public static void main(String[] args){

    //Accessing a static variable
    System.out.println("Total number of ATMs: " + ATM.numATMs); 

    // Calling a static method
    ATM.averageMoney();
  }

}
```

12. Static Methods with Instance Variables: Static methods cannot access or change the values of instance variables.

    ```
    class ATM{
    // Static variables
    public static int totalMoney = 0;
    public static int numATMs = 0; 

    public int money = 1;

    // A static method
    public static void averageMoney(){
        // Can not use this.money here because a static method can't access instance variables
    }

    }
    ```
13. Methods with Static Variables: Both non-static and static methods can access or change the values of static variables.

    ```
    class ATM{
    // Static variables 
    public static int totalMoney = 0; 
    public static int numATMs = 0; 
    public int money = 1;

    // A static method interacting with a static variable 
    public static void staticMethod(){ 
        totalMoney += 1;
    } 

    // A non-static method interactingwith a static variable 
    public void nonStaticMethod(){
        totalMoney += 1; 
    } 
    }
    ```

14. Static Methods and the this Keyword: Static methods do not have a this reference and are therefore unable to use the class’s instance variables or call non-static methods.

    ```
    public class DemoClass{

    public int demoVariable = 5;

    public void demoNonStaticMethod(){
        
    }
    public static void demoStaticMethod(){
        // Can't use "this.demoVariable" or "this.demoNonStaticMethod()"
    }
    }
    ```

## Inheritance and Polymorphism: 

1. Inheritance in Java: Inheritance is an important feature of object-oriented programming in Java. It allows for one class (child class) to inherit the fields and methods of another class (parent class). For instance, we might want a child class Dog to inherent traits from a more general parent class Animal. When defining a child class in Java, we use the keyword extends to inherit from a parent class.

    ```
    // Parent Class
    class Animal {
    // Animal class members
    }

    // Child Class
    class Dog extends Animal {
    // Dog inherits traits from Animal 
    
    // additional Dog class members
    }
    ```

2. Main() method in Java: If the Java file containing our Shape class is the only one with a main() method, this is the file that will be run for our Java package.

    ```
    // Shape.java file 
    class Shape {
    public static void main(String[] args) {
        Square sq = new Square();
    }
    }

    // Square.java file 
    class Square extends Shape {
    
    }
    ```

3. super() in Java: In Java, a child class inherits its parent’s fields and methods, meaning it also inherits the parent’s constructor. Sometimes we may want to modify the constructor, in which case we can use the super() method, which acts like the parent constructor inside the child class constructor. Alternatively, we can also completely override a parent class constructor by writing a new constructor for the child class.

    ```
    // Parent class
    class Animal {
    String sound;
    Animal(String snd) {
        this.sound = snd;
    }
    }

    // Child class
    class Dog extends Animal { 
    // super() method can act like the parent constructor inside the child class constructor.
    Dog() {
        super("woof");
    } 
    // alternatively, we can override the constructor completely by defining a new constructor.
    Dog() {
        this.sound = "woof";
    }
    }
    ```

4. Protected and Final keywords in Java: When creating classes in Java, sometimes we may want to control child class access to parent class members. We can use the protected and final keywords to do just that. Protected keeps a parent class member accessible to its child classes, to files within its own package, and by subclasses of this class in another package. Adding final before a parent class method’s access modifier makes it so that any child classes cannot modify that method - it is immutable.

    ```
    class Student {
    protected double gpa;
    // any child class of Student can access gpa 
    
    final protected boolean isStudent() {
        return true;
    }
    // any child class of Student cannot modify isStudent()
    }
    ```

5. Polymorphism in Java: Java incorporates the object-oriented programming principle of polymorphism. Polymorphism allows a child class to share the information and behavior of its parent class while also incorporating its own functionality. This allows for the benefits of simplified syntax and reduced cognitive overload for developers.

    ```
    // Parent class
    class Animal {
    public void greeting() {
        System.out.println("The animal greets you.");
    }
    }

    // Child class
    class Cat extends Animal {
    public void greeting() {
        System.out.println("The cat meows.");
    }
    }

    class MainClass {
    public static void main(String[] args) {
        Animal animal1 = new Animal();  // Animal object
        Animal cat1 = new Cat();  // Cat object
        animal1.greeting(); // prints "The animal greets you."
        cat1.greeting(); // prints "The cat meows."
    }
    }
    ```

6. Method Overriding in Java: In Java, we can easily override parent class methods in a child class. Overriding a method is useful when we want our child class method to have the same name as a parent class method but behave a bit differently. In order to override a parent class method in a child class, we need to make sure that the child class method has the following in common with its parent class method:

- Method name
- Return type
- Number and type of parameters

    Additionally, we should include the @Override keyword above our child class method to indicate to the compiler that we want to override a method in the parent class.

    ```
    // Parent class 
    class Animal {
    public void eating() {
        System.out.println("The animal is eating.");
    }
    }

    // Child class 
    class Dog extends Animal {
    // Dog's eating method overrides Animal's eating method
        @Override
    public void eating() {
        System.out.println("The dog is eating.");
    }
    }
    ```

7. Child Classes in Arrays and ArrayLists: In Java, polymorphism allows us to put instances of different classes that share a parent class together in an array or ArrayList. For example, if we have an Animal parent class with child classes Cat, Dog, and Pig we can set up an array with instances of each animal and then iterate through the list of animals to perform the same action on each.

```
// Animal parent class with child classes Cat, Dog, and Pig. 
Animal cat1, dog1, pig1;

cat1 = new Cat();
dog1 = new Dog();
pig1 = new Pig();

// Set up an array with instances of each animal
Animal[] animals = {cat1, dog1, pig1};

// Iterate through the list of animals and perform the same action with each
for (Animal animal : animals) {
  
  animal.sound();
  
}
```

## Two-Dimensional Arrays: 

1. Nested Iteration Statements: In Java, nested iteration statements are iteration statements that appear in the body of another iteration statement. When a loop is nested inside another loop, the inner loop must complete all its iterations before the outer loop can continue.

    ```
    for(int outer = 0; outer < 3; outer++){
        System.out.println("The outer index is: " + outer);
        for(int inner = 0; inner < 4; inner++){
            System.out.println("\tThe inner index is: " + inner);
        }
    }
    ```

2. Declaring 2D Arrays: In Java, 2D arrays are stored as arrays of arrays. Therefore, the way 2D arrays are declared is similar 1D array objects. 2D arrays are declared by defining a data type followed by two sets of square brackets. 

    ```
    int[][] twoDIntArray;
    String[][] twoDStringArray;
    double[][] twoDDoubleArray;
    ```

3. Accessing 2D Array Elements: In Java, when accessing the element from a 2D array using arr[first][second], the first index can be thought of as the desired row, and the second index is used for the desired column. Just like 1D arrays, 2D arrays are indexed starting at 0.

    ```
    //Given a 2d array called `arr` which stores `int` values
    int[][] arr = {{1,2,3},
                {4,5,6}};

    //We can get the value `4` by using
    int retrieved = arr[1][0];
    ```

4. Initializer Lists: In Java, initializer lists can be used to quickly give initial values to 2D arrays. This can be done in two different ways.
- If the array has not been declared yet, a new array can be declared and initialized in the same step using curly brackets.
- If the array has already been declared, the new keyword along with the data type must be used in order to use an initializer list. 

    ```
    // Method one: declaring and intitializing at the same time
    double[][] doubleValues = {{1.5, 2.6, 3.7}, {7.5, 6.4, 5.3}, {9.8,  8.7, 7.6}, {3.6, 5.7, 7.8}};

    // Method two: declaring and initializing separately:
    String[][] stringValues;
    stringValues = new String[][] {{"working", "with"}, {"2D", "arrays"}, {"is", "fun"}};
    ```

5. Modify 2D Array Elements: In Java, elements in a 2D array can be modified in a similar fashion to modifying elements in a 1D array. Setting arr[i][j] equal to a new value will modify the element in row i column j of the array arr.

    ```
    double[][] doubleValues = {{1.5, 2.6, 3.7}, {7.5, 6.4, 5.3}, {9.8,  8.7, 7.6}, {3.6, 5.7, 7.8}};

    doubleValues[2][2] = 100.5;
    // This will change the value 7.6 to 100.5
    ```

6. Row-Major Order: “Row-major order” refers to an ordering of 2D array elements where traversal occurs across each row - from the top left corner to the bottom right. In Java, row major ordering can be implemented by having nested loops where the outer loop variable iterates through the rows and the inner loop variable iterates through the columns. Note that inside these loops, when accessing elements, the variable used in the outer loop will be used as the first index, and the inner loop variable will be used as the second index.

    ```
    for(int i = 0; i < matrix.length; i++) {
        for(int j = 0; j < matrix[i].length; j++) {
            System.out.println(matrix[i][j]);
        }
    }
    ```

7. Column-Major Order: “Column-major order” refers to an ordering of 2D array elements where traversal occurs down each column - from the top left corner to the bottom right. In Java, column major ordering can be implemented by having nested loops where the outer loop variable iterates through the columns and the inner loop variable iterates through the rows. Note that inside these loops, when accessing elements, the variable used in the outer loop will be used as the second index, and the inner loop variable will be used as the first index.

```
for(int i = 0; i < matrix[0].length; i++) {
    for(int j = 0; j < matrix.length; j++) {
        System.out.println(matrix[j][i]);
    }
}
```

8. Traversing With Enhanced For Loops: In Java, enhanced for loops can be used to traverse 2D arrays. Because enhanced for loops have no index variable, they are better used in situations where you only care about the values of the 2D array - not the location of those values. 

    ```
    for(String[] rowOfStrings : twoDStringArray) {
        for(String s : rowOfStrings) {
            System.out.println(s);
        }
    }
    ```