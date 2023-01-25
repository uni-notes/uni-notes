## Question

Write an algorithm and C/C++/JAVA program for the following problem:

1. Create a Binary Search Tree (Ordered Binary Tree) to store `<IDNo, Name, CGPA>` for $n$ students (say $n=20$ record at least)
   - Read from a text file
   - Copy each record into nodes in the tree
   - Assume IDNO is the primary key.
2. Perform INORDER Traversal of the above tree and show output.
3. Perform PREORDER Traversal of the above tree and show output.
4. Perform POSTORDER Traversal of the above tree and show output.

## Algorithm

### Pseudocode

```pseudocode

```

### Time Complexity

| Algorithm | Complexity  |
| :-------: | :---------: |
| insert()  | $O(\log n)$ |
|  inFix()  |   $O(n)$    |
|  preFix   |   $O(n)$    |
|  postFix  |   $O(n)$    |

## Source Code

```java
// Ahmed Thahir 2020A7PS0198U

import java.util.*;
import java.io.*;

class Node { 
	String key; 
	String name;
	float CGPA;
	Node left, right; 
	
	public Node(String data, String n, float c){ 
		key = data; 
		name = n;
		CGPA = c;
		left = right = null; 
	} 
}

class BST
{ 
	Node root; 
	
	BST(){ 
		root = null; 
	} 
	
	void insert(String key, String name, float c)  { 
		root = insertRecursive(root, key, name, c); 
	} 
	
	Node insertRecursive(Node root, String key, String name, float c) { 
		if (root == null) { 
			root = new Node(key, name, c);
		} 
		else if (key.compareTo(root.key)<0)
			root.left = insertRecursive(root.left, key, name, c);
		else if (key.compareTo(root.key)>0)    
			root.right = insertRecursive(root.right, key, name , c);
    
    return root; 
	} 
	
	
	void inFix() { 
		inFixRecursive(root); 
	} 
	
	void inFixRecursive(Node root) { 
		if (root != null) { 
			inFixRecursive(root.left); 
			System.out.print(root.key + " "); 
			System.out.print(root.name + " "); 
			System.out.println(root.CGPA + " "); 
			inFixRecursive(root.right); 
		} 
	} 
	
	void postFix() { 
		postFixRecursive(root); 
	} 
	
	void postFixRecursive(Node root) { 
		if (root != null) { 
			postFixRecursive(root.left); 
			postFixRecursive(root.right); 
			System.out.print(root.key + " "); 
			System.out.print(root.name + " "); 
			System.out.println(root.CGPA + " "); 
		} 
	}
	
	void preFix() { 
		preFixRecursive(root); 
	} 
	
	void preFixRecursive(Node root) { 
		if (root != null) { 
			System.out.print(root.key + " "); 
			System.out.print(root.name + " "); 
			System.out.println(root.CGPA + " "); 
			preFixRecursive(root.left); 
			preFixRecursive(root.right); 
		} 
	}
}
class p07
{
	public static void main(String[] args) throws FileNotFoundException
	{
		Scanner readMyFile = new Scanner(new File("input.txt"));
		BST bst = new BST(); 
		
		while(readMyFile.hasNext())
		{
			String key = readMyFile.next();
			String name = readMyFile.next();
			float CGPA = readMyFile.nextFloat();
			bst.insert(key, name, CGPA); 
		}
		System.out.println("InFix traversal:"); 
		bst.inFix(); 
		System.out.println("\n\nPreFix traversal:"); 
		bst.preFix(); 
		System.out.println("\n\nPostFix traversal:"); 
		bst.postFix();
	} 
}
```

## Test Cases

### Input

```
2019A7PS096U AA 7.6
2019A7PS103U BB 7.5
2019A7PS107U CC 7.4
2019A7PS140U DD 7.3
2019A3PS135U EE 8.5
2019A3PS410U FF 8.4
2019A7PS001U GG 8.3
2019A7PS003U HH 8.2
2019A7PS023U II 8.1
2019A7PS034U JJ 8.0
2019A7PS042U KK 7.9
2019A7PS054U LL 7.8
2019A7PS091U MM 7.7
2019A7PS281U NN 9.1
2019A7PS424U OO 9.0
2019A3PS019U PP 8.9
2019A3PS080U QQ 8.8
2019A7PS153U RR 7.2
2019A7PS209U SS 7.1
2019A3PS113U TT 8.6
```

### Output

```:
InFix traversal:
2019A3PS019U PP 8.9 
2019A3PS080U QQ 8.8 
2019A3PS113U TT 8.6 
2019A3PS135U EE 8.5 
2019A3PS410U FF 8.4 
2019A7PS001U GG 8.3 
2019A7PS003U HH 8.2 
2019A7PS023U II 8.1 
2019A7PS034U JJ 8.0 
2019A7PS042U KK 7.9 
2019A7PS054U LL 7.8 
2019A7PS091U MM 7.7 
2019A7PS096U AA 7.6 
2019A7PS103U BB 7.5 
2019A7PS107U CC 7.4 
2019A7PS140U DD 7.3 
2019A7PS153U RR 7.2 
2019A7PS209U SS 7.1 
2019A7PS281U NN 9.1 
2019A7PS424U OO 9.0 
PreFix traversal:
2019A7PS096U AA 7.6 
2019A3PS135U EE 8.5 
2019A3PS019U PP 8.9 
2019A3PS080U QQ 8.8 
2019A3PS113U TT 8.6 
2019A3PS410U FF 8.4 
2019A7PS001U GG 8.3 
2019A7PS003U HH 8.2 
2019A7PS023U II 8.1 
2019A7PS034U JJ 8.0 
2019A7PS042U KK 7.9 
2019A7PS054U LL 7.8 
2019A7PS091U MM 7.7 
2019A7PS103U BB 7.5 
2019A7PS107U CC 7.4 
2019A7PS140U DD 7.3 
2019A7PS281U NN 9.1 
2019A7PS153U RR 7.2 
2019A7PS209U SS 7.1 
2019A7PS424U OO 9.0 
PostFix traversal:
2019A3PS113U TT 8.6 
2019A3PS080U QQ 8.8 
2019A3PS019U PP 8.9 
2019A7PS091U MM 7.7 
2019A7PS054U LL 7.8 
2019A7PS042U KK 7.9 
2019A7PS034U JJ 8.0 
2019A7PS023U II 8.1 
2019A7PS003U HH 8.2 
2019A7PS001U GG 8.3 
2019A3PS410U FF 8.4 
2019A3PS135U EE 8.5 
2019A7PS209U SS 7.1 
2019A7PS153U RR 7.2 
2019A7PS424U OO 9.0 
2019A7PS281U NN 9.1 
2019A7PS140U DD 7.3 
2019A7PS107U CC 7.4 
2019A7PS103U BB 7.5 
2019A7PS096U AA 7.6 
```