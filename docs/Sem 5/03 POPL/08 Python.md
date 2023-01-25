## Characteristics of Scripting Languages
- Both Batch and Interactive use
- Economy of Expression
- Lack of declarations; simple scoping rules
- Flexible dynamic typing
- Easy access to other programs
- Sophisticated Pattern matching
- High-level data types

## Python

is a general-purpose interpreted, interactive, object-oriented, and high-level programming language.

It was created by Guido van Rossum during 1985-1990.

Philosophy: Easy-to-read, easy-to-learn

## Variables

No explicit declaration; declaration happens the first time you initialize variable

## Data Types

```python
x = True
x = 5 ## int
x = 5.4 ## float
x = 'a' #string
x = 1+10j ## complex number
x = "Ahmed Thahir" ## string
x = ("a", "b", "c") ## tuple 
x = ["a", "b", "c"] ## list
range(5) ## range from 0-4
x = { ## dict
  "a": "x",
  "b": "y"
}
x = {"a", "b", "c"}
x = frozenset({"a", "b", "c"}) ## frozenset
x = b"Hello" ## bytes
x = bytearray(5) ## bytearray
x = memoryview(bytes(5)) ## memoryview
```

## Idk

```python
x = str(3)
x = int(3)
x = float(3)

print(x)
print(type(x))

fruits = ["apple", "banana", "cherry"]
x, y, z = fruits
print(x, y, z)
## apple banana cherry
```

## Conditional Statements

```python
if x < 10 or y < 20:
  print(1)
elif x > 20 and y < 20:
  print(2)
else
	print(3)
  
print(1) if x > 10 else print(2) if x>20 else print(3)
```

## Loops

```python
i = 0
while i < 10:
  if(i==3):
    continue
  
  if(i==5):
    break
  
  print(i)
  i += 1
```

```python
for i in range(5):
  print(i)
  
for i in range(1, 5, 3): ## starting, ending, updation
  print(i)

for i in [3, 5, 7]:
  print(i)
  
for ch in "Hello":
  print(ch)
```

### Pass vs Comments

| `pass`                                  | `#comment`   |
| --------------------------------------- | ------------ |
| No operation CPU instruction<br />(NOP) | Skipped over |

```python
def func(test):
	## comment
  pass
```

## List

```python
a = [10, 20, 30, 40, 50]

print(a)
print(a[0])
print(a[-1])

print(a[3:5])
## prints list with elements of index 3 and 4
## tip to remember: no of elements = 5-3 = 2

if 10 in a:
  print("Yes")
if "hi" in a:
  print("Yes")
```

```python
a.append("hi")
a.clear()
b = a.copy()
a.count("hi")
a.extend(another_list)
a.index("hi")

a.insert("hi", 3)
a.pop(3) ## index
a.remove("hi") ## element
a.reverse()
a.sort()
```

### List Comprehension

```python
new_list = [x for x in fruits if x > 10]
new_list = [x for x in fruits if "h" in x]
```

## Tuple

is non-mutable

```python
a = [10, 20, 30, 40, 50]

print(a)
print(a[0])
print(a[-1])

print(a[3:5])
## prints tuple with elements of index 3 and 4
## tip to remember: no of elements = 5-3 = 2

if 10 in a:
  print("Yes")
if "hi" in a:
  print("Yes")
```

```python
a.count("hi")
a.index("hi")
```

### Modifying a tuple

#### Add a tuple

```python
x = (10, 20, 30, 40, 50)
x += (60)
```

#### You have to convert it to a list and then revert it to a tuple

```python
x = (10, 20, 30, 40, 50)
x = list(x)

x.append(40)
x[1] = 100
x.remove(10)
x += 50

x = tuple(x)
```

### Advantages

- Better performance than list, as they are immutable
- Tuple ensures write-protection of elements
- General convention
    - Tuples for different data types
    - Lists for similar data types
- Tuples that contain immutable elements can be used as key for dictionary

## Dictionaries

key-value pairs

==**keys have to be unique**==

```python
student_marks = {
  "thahir": 10,
  "ahmed": 20
}

student_marks = dict(
  thahir = 10,
  ahmed = 20
)

print(student_marks["thahir"])
print(student_marks["ahmed"])

print(students_marks.get("thahir"))

for key in students_marks.keys():
  print(students_marks[key])

for key, value in student_marks:
  print(key, value)
```

### Functions

```python
student_marks.clear()
s = student_marks.copy()

new_dict = dict.fromkeys(("key1", "key2"), 0)
## gives a dictionary with the specified keys and specified value
## {'key1': 0, 'key2': 0, 'key3': 0}
new_dict = dict.fromkeys(("key1", "key2"))
## {'key1': None, 'key2': None, 'key3': None}

student_marks.get("thahir")
student_marks.items() ## list of tuples
student_marks.keys() ## list
student_marks.values() ## list

student_marks.pop("thahir")
student_marks.popitem() #remove last inserted key-value pair

x = thisdict.setdefault("azhar", 10)
## Returns the value of the specified key. If the key does not exist: insert the key, with the specified value

student_marks.update({
  "thahir": 100
})
```

### Nested

```python
student_marks = {
  "thahir": {
    "math": 10,
    "science": 20
  },
  "ahmed": {
    "math": 10,
    "science": 20
  }
}
```

## Set

unordered and mutable collection of elements, which are unique and immutable

```python
my_set = {"thahir", "ahmed", 10}

for x in my_set:
  print(x)
```

## Strings

```python
a = 'hello world'
a = "hello world"
a = """
hello world
"""

print(a)
print(a[0])
print(a[1:3])

print(a[-3:-1])
```

## Functions

Values are passed by value, just like C++. However, pandas dataframes behave like pointers, and hence are passed by reference.

```python
def func(name, age):
  print(name, age)

  return name
  
name = func("ahmed", 20)
name = func(age=20, name="ahmed")
```

### Arbitrary Arguments

When you are unsure how many parameters will be passed

```python
def func(*args):
  print(args) ## tuple
  print(args[0], args[1])
```

### Default Argument

```python
def func(name, age=20):
  print(name, age)
  
func("ahmed")
```

### Keyword Arguments

```python
def func(**kwargs):
  print(kwargs) ## dictionary
  print(args["name"], args["age"])
  
func(name = "thahir", age = 20)
```

### Recursion

```python
def fact(n):
  if(n==0):
    return 1
  else:
    return n * fact(n-1)
  
f = fact(10)
```

## Classes

The first argument to any function/constructor in a class is the class object itself

```python
class Student:
  def __init__(self): ## self can be anything (blah, bruh)
    self.age = 20

class Student:
  def __init__(bruh, name, age):
    bruh.name = name
    bruh.age = age
  def print_name(bruh):
    print(bruh.name)
    
## object creation
s = Student("thahir", 20)

## printing
print(s.age)
print(s.print_name())

## modification
s.age += 10

## deletion
del s.age
```

## Inheritance

```python
class Student:
  def __init__(self, name): ## self can be anything (blah, bruh)
    self.name = name
    self.age = 20
    
class Math_Student(Student):
  pass

class Sci_Student(Student):
  def __init__(self, name):
    Student.__init__(self, name)
    
class Eng_Student(Student):
  def __init__(self, name):
    super().__init__(name) ## no need of self
    
## Multiple inheritance
class CS_Student(Sci_Student, Math_Student):
  def __init__(self, name):
    Sci_Student.__init__(self, name)
    Math_Student.__init__(self, name)
```