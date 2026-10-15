## Comparable and Comparator

they both are interfaces

useful for elements of `Collection`

- sorting
- comparing

``` java
Collections.sort(list); // sort() is an abstract function

Collections.sort( list, new NameComparator() );
```

### Comparable
``` java
// Comparable
class Student implements Comparable<Student>
{
  String name;
  int age;
  
  Student(String n, int a)
  {
    name = n;
    a = age
  }
  
  public int compareTo(Student s)
  {
    if(age < s.age)
      return -1;
    else if(age > s.age)
      return 1;
    else
      return 0;
  }
}

// ArrayList

Collections.sort(al); // sorting based on age
//display
```

### Comparator

``` java
// Comparable
class Student implements Comparable<Student>
{
  String name;
  int age;
  
  Student(String n, int a)
  {
    name = n;
    a = age
  }
}

class AgeComparator implements Comparator<Student>
{
  public int compare(Student s1, Student s2)
	{
    if(s1.age < s2.age)
      return -1;
    else if(s1.age > s2.age)
      return 1;
    else
      return 0;
	}
}

class NameComparator implements Comparator<Student>
{
  public void compare(Student s1, Student s2)
  {
    return s1.name.compareTo(s2.name);
  }
}

// ArrayList

Collections.sort(al, new AgeComparator());
// display

Collections.sort(al, new NameComparator());
// display
```

### Difference

|                       | Comparable                | Comparator                                        |
| --------------------- | ------------------------- | ------------------------------------------------- |
| package               | `java.lang`               | `java.util`                                       |
| method to implement   | `compareTo()`             | `compare()`                                       |
| Sorting Sequence      | Single                    | Multiple                                          |
| Affect original class | Y                         | N                                                 |
| Sorting               | `Collections.sort(list);` | `Collections.sort( list, new NameComparator() );` |

