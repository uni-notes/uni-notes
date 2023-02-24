## Lists

``` mermaid
flowchart BT
Vector-Class & ArrayList-Class & LinkedList-Class -->
|implements|List -->
|extends|Collection
```

## `Vector`

obsolete/not recommended to use

```java
import java.util.Vector;

Vector<TypeClass> arrL = new Vector<TypeClass>();
Vector<Integer> arrL = new Vector<Integer>();

//constructors
Vector();
Vector(int initialCapacity);
Vector(int initialCapacity, int capacityIncrement);

// E is element type
void add(int index, E element);
boolean add(E e);
void addElement(E obj);
boolean addAll(Collection C);
boolean addAll(int index, Collection C);

void setElementAt(Object element, int index);
boolean removeElement(Object element);
boolean removeAll(Collection c);

int capacity();
void clear();

v.get(int index);

boolean contains(Object element);
boolean containsAll(Collection c);

Object elementAt(int index);
Object firstElement();
Object lastElement();

boolean isEmpty();
```

## `ArrayList`

As opposed to arrays, array lists are

- mutable
- size is dynamic
- only for non-primitive data types
- you have to access element using `arrL.get(index)` rather than `[]`

``` java
import java.util.ArrayList;

ArrayList<TypeClass> arrL = new ArrayList<TypeClass>(); // size is not compulsory
ArrayList<Integer> arrL = new ArrayList<Integer>();

List<Integer> l = new ArrayList<>();
// this is also allowed
// List is the parent class of LinkedList
// no need to specify type for the second <>

// constructors
ArrayList(); // build an empty array list
ArrayList(int capacity); // build an array list with initial capacity
ArrayList(Collection c); // build an array list intialized with the elements from collection c
// you can pass linkedlist as a parameter (collection class)

arrL.add(Object e);
arrL.add(20); // 20
arrL.add(50); // 20 50

arrL.add(int index, Object e);
arrL.add(index, 130); // 20 50 130
arrL.add(2, "hello");

al.addAll(Collection c);

al.set(int index, Object newData);

arrL.remove(int index);
arrL.remove(2); // 20 50
arrL.remove(Object data); // String, Integer, Float

arrL.get(index);
arrL.get(1); // 50

arrL.size(); // 2

Collections.sort(list); // ascending

// display
System.out.println(arrL); // [20, 50]

for(int i = 0; i<arrL.size(); i++)
  System.out.print(arrL.get(i) + " "); // 20 50

// enhanced for loop
for(String str:arrL)
  System.out.println(str);
for(Integer i:arrL)
  System.out.println(i);

al.indexOf(Object o);
al.contains(Object o); // searches for element and returns true/false
al.clear(); // delete all elements of the arraylist

ArrayList<String> newal = (ArrayList<String>) al.clone();

al.ensureCapacity(int minCapacity); // increases the size of the arraylist to the minCapacity
// it is overriden method as it is available for StringBuffer also
```

## `LinkedList`

``` java
import java.util.LinkedList;

LinkedList<Type> l = new LinkedList<Type>();
LinkedList<Integer> l = new LinkedList<Integer>();

// constructors
LinkedList();
LinkedList(Collection c);
// you can pass arraylist as a parameter (collection class)

l.size();

l.add(Object e) // returns boolean - true/false - after doing the thingy to indicate if the element was added or not
l.add(3);

l.add(int index, Object e); // returns void
l.add(2, 3);

l.addAll(Collection c); // take all elements of arraylist and it to the linked list 

l.addFirst(Object e);
l.addLast(Object e); // most performance-efficient

l.remove(); // removes the first element, ie the default index passed is 0
l.remove(int index);
l.revove(Object o);

l.removeFirst();
l.removeLast();
l.removeFirstOccurence(Object e);
l.removeLastOccurence(Object e);

l.get(index);
l.get(2);

l.set(int index, Object newdata);
l.set(2, "hello");

l.clear(); // delete all elements
Object obj = l.clone();
boolean contains(Object item);

l.indexOf(Object item); //returns null if it doesn't find the element
l.lastIndexOf(Object item);
Object poll(); // removes and returns the last element
// more efficent than remove() as there is only instance of it
Object pollFirst();
Object pollLast();

Collections.sort(list); // ascending
```

## Differences

|                         | `Vector`                                       | `ArrayList`                                                  | `LinkedList`                                                 |
| ----------------------- | ---------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Each location contains  | data                                           | data                                                         | 1. data<br />2. pointer to previous node<br />3. pointer to next node |
| Type                    |                                                | Dynamic Array                                                | double linked list                                           |
| setting data in between | slowest                                        | intermediate                                                 | fastest                                                      |
| reason for performance  |                                                | insertion/deletion involves affecting all succeeding elements | only pointers and data of few nodes are affected; other nodes are left unaffected |
| getting data            |                                                | fastest                                                      | slowest                                                      |
| mincapacity             | 10                                             | 10                                                           | -                                                            |
| Increase in size        | 2x (if not mentioned)<br />else user increment | 1.5x                                                         | -                                                            |

## `Iterator`

can only iterate in the forward direction

``` java
import java.util.Iterator;

boolean hasNext();
(Type) itr.next(); // returns the current element and moves the pointer to the next position
// same like p++ in C++

// the default return type of next() is String, so we have to type cast

void remove(); // removes the last element returned by the iterator

// Integer
Iterator<Integer> itr = al.iterator(); // al is the arraylist
// itr is an indirect pointer/cursor to the location
// JUST BEFORE THE 0TH INDEX

while(itr.hasNext())
{
  int i = (Integer) itr.next();
  
	if(i%2 == 0)
	  itr.remove(); // removes odd
}

// String
Iterator<String> itr = l.iterator();
while(itr.hasNext())
{
  itr.remove();
  Iterator<String> itr2 = itr.iterator(); // not sure
  while(itr2.hasNext()) // to go through each element
  {
    System.out.println(itr2.next() + "");
  }
}
```

## `ListIterator`

iterate forward and backward

``` java
import java.util.ListIterator;

ListIterator<String> litr = null;
litr = al.listIterator();

while(litr.hasNext())
{
  System.outprintln(litr.next());
}

while(litr.hasPrevious())
{
  System.outprintln(litr.previous());
}

boolean hasNext();
E next();
int nextIndex();

boolean hasPrevious();
E previous(); // returns the current element and moves the pointer to the previous location 
// same like p-- in C++
int previousIndex();

void remove(); // remove the last element which is returned by the next() or previous()
void set(E e); // replace the last element which is returned by the next() or previous()
void add(E e); // insert before next element returned by next() method
```
