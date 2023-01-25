## Cloning

creating an exact copy of an existing object in the memory

`clone()` from class `java.lang.Object`

Only objects of classes which implement `Cloneable` interface are eligible for cloning

By default, shallow copy occurs

|                         | Shallow                 | Deep                                  |
| ----------------------- | ----------------------- | ------------------------------------- |
| definition              |                         | custom `clone()`                      |
| original and clone      | dependent on each other | independent                           |
| changes                 | affect each other       | no effect                             |
| preferred if object has | only primitive fields   | references to other objects as fields |
| performance             | faster and cheaper      | slower and costlier                   |

## Deep Copy

```java
class Course implements Cloneable
{
  String sub1, sub2, sub3;

    // constructor

  protected Object clone() throws CloneNotSupportedException
  {
    return super.clone();
  }
}

class Student implements Cloneable
{
  int id;
  String name;
   Course c;

    // constructor

  // this is what defines a deep copy
  // without this, it will just be a shallow copy
  protected Object clone() throws CloneNotSupportedException
  {
    Student s = (Student) super.clone();
    student.c = (Course) c.clone();
    return s;
  }
}

class Tester
{
  public static void main(String args[])
  {
    Course c = new Course("Phy", "Chem", "Bio");

    Student s1 = new Student(111, "John", c);
    Student s2 = null;

    try {
      s2 = (Student) s1.clone();
    } catch (CloneNotSupportedException e) {
      e.printStackTrace();
    }

    System.out.println(s1.c.sub3); // Bio

    s2.course.sub3 = "Math"; // will not affect s1

  }
}
```
