## Question

Write a C/C++/JAVA program to perform the following actions on a DOUBLY LINKED LIST:

1. Implement insertLast(o) operation in a DLL for N (say N= 5) RECORDS of DATA SET (Student Record). You have to read each record of data set from an input file “studentin.dat” stored locally in your directory and the insert it into the DLL, one at a time. (You can use vi/joe editor to create input data file. Make sure that the input data file contains at least 5 records).
2. Implement the remove(p) operation for the DLL for any one record, by interactively asking for its position p from standard input (keyboard).
3. Traverse the List in forward direction (beginning to end) and display all records on the standard output (display).
4. Traverse the List in reverse direction (end to beginning) and display all records on the standard output (display).

## Algorithm

### Pseudocode

```pseudocode
Algorithm insertLast(d)
	INPUT read student records from file
	OUTPUT student records to DLL
	
	if size = 0
		start = inserted element
	else
		inserted element's back link = existing element
		existing element's front link = inserted element
	
	end = n
	size = size + 1
	
Algorithm remove(p)
	INPUT student records of DLL
	INPUT position of element to be removed
	OUTPUT deleted element
	
	if size = 0
		empty list
	else if p > size
		position out of bounds
	else
		current element = start
		while current element is not null
		if current element index = p
			Print deleted element
			previous element's front link = next element
			next element's back link = previous element
				
			current element's back and front link = null
		else
			current element = current element's front link
			increment i
				
Algorithm traverseForward
	INPUT student records of DLL
	OUTPUT forward traversed list
	
	if size = 0
		List Empty
	else
		current element = start
		while current element is not null
			Print current element
			current element = current element's front link
			
Algorithm traverseBackward
	INPUT student records of DLL
	OUTPUT backward traversed list
	
	if size = 0
		List Empty
	else
		current element = end
		while current element is not null
			Print current element
			current element = current element's back link
```

### Time Complexity

|    Algorithm     | Complexity |
| :--------------: | :--------: |
|    insertLast    |   $O(n)$   |
|      remove      |   $O(n)$   |
| traverseForward  |   $O(n)$   |
| traverseBackward |   $O(n)$   |

### Extra Note

I was initially thinking of assigning an index for each node, but that will increase the amount of steps for each process, because the processor has to update the index of elements any time a change happens to the list.

## Source Code

```java
// Ahmed Thahir 2020A7PS0198U

package Programs;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

class Node
{
  Node bp; // back pointer
  String d; // data
  Node fp; // front pointer
  
  Node()
  {
    bp = null;
    fp = null;
  }

  Node(String val)
  {
    this();
    d = val;
  }
  
  Node(Node bptr, String data, Node fptr)
  {
    bp = bptr;
    d = data;
    fp = fptr;
  }

  void setBp(Node ptr)
  {
    bp = ptr;
  }

  void setFp(Node ptr)
  {
    fp = ptr;
  }

  void setData(String data)
  {
    d = data;
  }

  Node getBp()
  {
    return bp;
  }

  String getData()
  {
    return d;
  }

  Node getFp()
  {
    return fp;
  }
}

class DLL // Double Linked List
{
  static Node start;
  static Node end;
  static int size;

  DLL()
  {
    start = null;
    end = null;
    size = 0;
  }

  public static void insertLast(String d)
  {
    Node n = new Node(d);
    if(size == 0)
    {
      start = n;
    }
    else
    {
      n.setBp(end); // link new node's back to the existing node
      end.setFp(n); // link existing node's front to the new node
    }

    end = n;
    size++;
  }

  public static void remove(int pos)
  {
    int index = pos-1;

    if(size == 0)
    {
      System.out.println("Empty");
    }
    else if (index >= size)
    {
      System.out.println("Index out of bounds");
    }
    else
    {
      Node n = start;
      int i = 0;
      while( n!=null )
      {
        if(i == index)
        {
          System.out.println( "Deleting: " + n.getData() );

          // link the previous and next one with each other
          Node prev = n.getBp();
          Node next = n.getFp();
          prev.setFp(next);
          next.setBp(prev);

          // unlink the deleted node
          n.setBp(null);
          n.setFp(null);

          break;
        }
        else
        {
          n = n.getFp();
          i++;
        }
      }
    }
  }

  public static void traverseForward()
  {
    if(size == 0)
    {
      System.out.println("List Empty");
    }
    else
    {
      System.out.println("\n" + "Traversing Forward");
      
      Node n = start;
      while( n!=null )
      {
        System.out.println( n.getData() );
        n = n.getFp();
      }
    }
  }

  public static void traverseBackward()
  {
    if(size == 0)
    {
      System.out.println("List Empty");
    }
    else
    {
      System.out.println("\n" + "Traversing Backward");

      Node n = end;
      while( n!=null )
      {
        System.out.println( n.getData() );
        n = n.getBp();
      }
    }
  }
}

public class p03
{
  static String rel = "h:\\My Drive\\Notes\\Sem 4\\02 DSA\\Practicals\\Programs\\";
  static String inputFile = rel +
    "studentin.dat";
  static String outputFile = rel +
    "studentout.dat";

  DLL students = new DLL();

  public static void main(String[] args) throws FileNotFoundException
  {
    Scanner readMyFile = new Scanner( new File(inputFile) );
    System.out.println("Reading from File");
    while (readMyFile.hasNext()) 
    {
      String o = readMyFile.nextLine();
      System.out.println(o);
      DLL.insertLast(o);
    }
    readMyFile.close();
    
    System.out.println("\n" + "Enter position (1, 2, ...) to remove");
    Scanner inp = new Scanner(System.in);
    int p = inp.nextInt();
    inp.close();
    DLL.remove(p);

    DLL.traverseForward();
    DLL.traverseBackward();
  }
}
```

## Test Cases

### Input

```
2021A7PS001 AAAA 1/1/2000 7.50
2021A7PS002 BBBB 2/1/2000 9.20
2021A7PS003 CCCC 3/1/2000 9.60
2021A7PS004 DDDD 4/1/2000 8.75
2021A7PS005 EEEE 5/1/2000 9.25
```

### Output

```
Reading from File
2021A7PS001 AAAA 1/1/2000 7.50
2021A7PS002 BBBB 2/1/2000 9.20
2021A7PS003 CCCC 3/1/2000 9.60
2021A7PS004 DDDD 4/1/2000 8.75
2021A7PS005 EEEE 5/1/2000 9.25

Enter position (1, 2, ...) to remove
2
Deleting: 2021A7PS002 BBBB 2/1/2000 9.20

Traversing Forward
2021A7PS001 AAAA 1/1/2000 7.50
2021A7PS003 CCCC 3/1/2000 9.60
2021A7PS004 DDDD 4/1/2000 8.75
2021A7PS005 EEEE 5/1/2000 9.25

Traversing Backward
2021A7PS005 EEEE 5/1/2000 9.25
2021A7PS004 DDDD 4/1/2000 8.75
2021A7PS003 CCCC 3/1/2000 9.60
2021A7PS001 AAAA 1/1/2000 7.50
```