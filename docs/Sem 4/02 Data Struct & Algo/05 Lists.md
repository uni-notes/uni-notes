## LinkedList

better than arrays ***in some aspects***, because

- you can add/delete elements in runtime
- capacity can be modified in runtime

### Logical Address

index in the array, visible to user

### Physical Address

address in the memory

## Singly LinkedList

consists of

- data
- pointer to the next location

$$
\fbox{D} \fbox{P} \quad
\fbox{D} \fbox{P} \quad
\fbox{D} \fbox{/}
$$

In the above diagram

- D = data
- P = pointer
- / = null pointer

### Inserting at tail

1. allocate a new node
2. enter element
3. set node point to null
4. make previous node point to current node

### Removing at tail

1. set 2nd last node point to null
2. de-allocate the memory (Java takes care of this automatically)

### Implementation

```java
class Node
{
  int d; // data
  Node p; // pointer
  
  Node()
  {
    d = 0;
    p = null;
  }
  
  Node(int data, Node ptr)
  {
    d = data;
    p = ptr;
  }
  void setLink(Node ptr)
  {
    p = ptr;
  }
  void setData(int data)
  {
    d = data;
  }
  Node getLink()
  {
    return p;
  }
  int getData()
  {
    return d;
  }
}

class LinkedList
{
  static Node start;
  static Node end;
  static int size;
  
  LinkedList()
  {
    start = null;
    end = null;
    size = 0;
  }
  
  int getSize()
  {
    return size;
  }

  boolean isEmpty()
  {
    return (getSize() == 0);
  }
  
  void insertAtStart(int val)
  {
    Node n = new Node(val, null);
    
    if(size == 0) // inserting for the first time
    {
      end = n;
    }
    else
    {
      n.setLink(start); // set the link to the previous start
    }
    
    start = n; // this is the new start
    size++;
  }
  
  void insertAtEnd(int val)
  {
    Node n = new Node(val, null);
    if(size == 0)
    {
      start = n;
    }
    else
    {
      end.setLink(n);
    }
    
    end = n;
    size++;
  }
  
  void insertAtIndex(int val, int index)
  {
    Node n = new Node(val, null);
    
    // traversal
    Node cur = start; // current node

    int i = 0;
    while(n != null)
    {
      if(i == index)
      {
        n.p = cur.p;
        cur.p = n;
        break;
      }
      else
      {
        cur = cur.getLink();
        i++;
      }
      
      size++;
    }
    
    void deleteAtIndex(int index)
    {
      Node n = start;
      int i = 0;
      
      while(n!=null)
      {
        if(i == index-1)
        {
          n.p = ?????????????
        }
        else
        {
          n = n.getLink();
          i++;
        }
      }
      
      size++;
    }
    public void display()
    {
      Node n = start;
      while(n!=null)
      {
        System.out.println(n.data);
        n = 
      }
    }
  }
}
```

## Stacked LL

implementing stack using linked list, rather than arrays

```java
class StackedLL
{
  Node top;
  StackedLL()
  {
    top = null;
  }
  void push(int data)
  {
    insertAtEnd(data);
  }
  void pop()
  {
		deleteAtEnd();
  }
}
```

## Queued LL

```java
class QueuedLL
{
	Node f, r; 
  StackedLL()
  {
    f = null;
    r = null;
  }
  void enqueue(int data)
  {
    insertAtEnd(data);
  }
  void pop()
  {
		deleteAtStart();
  }
}
```

## Double Linked List

[03 DLL.md](Practicals/03 DLL.md) 

## Circular Linked List

Used for dynamic circular queues to schedule tasks in OS

### Single

Tail points to head

### Double

`TailFrontPointer` points to head

`HeadBackPointer` points to tail
