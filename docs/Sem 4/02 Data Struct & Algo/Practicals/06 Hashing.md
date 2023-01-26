## Question

Write an algorithm and C/C++/JAVA program to perform the following.

It is required to store various strings in a HASH TABLE. The hash function is defined as follows:

Read in strings from an input text file (source.txt) and calculate hash value for each string using the hash function given below. You can permit collisions, in case if they occur [i.e. one or more strings can map to the same hash value; you can store them in the same sub-list corresponding to the computed hash value]. 

Assume that the input string has English alphabets (upper case and lower case) and digits. Note the range of ASCII values for A-Z is 65-90, a-z is 97-122 and digits 0-9 is 48-57.

HASH FUNCTION for an input string is defined as follows:

$$
\Bigg( \left(\sum \text{alphabets' ASCII} + 2 \sum \text{digits' ASCII} \right) * 17 + 5 \Bigg) \% 6

$$
**Note:** MOD(%) denotes modulus operator (i.e. remainder after division)

**Example**
Input String : Az9
Hash Value

$$
\begin{align}
&= \Big((65 + 122 + 2*57) *17 + 5 \Big) \% 6 \\&= (301*17 +5) \% 6 \\&= 5122 \% 6 \\&= 4
\end{align}
$$

1. Compute the hash values for each of the following twenty input strings and display the values. Note: you can read each input string from a text file (one string in each line).
1. Display the contents of the Hash Table showing the elements of each sub-list.

## Algorithm

### Pseudocode

```pseudocode

```

### Time Complexity

|    Algorithm    | Complexity |
| :-------------: | :--------: |
|       sum       |   $O(n)$   |
|      hash       |   $O(1)$   |
| displaySublists |   $O(n)$   |

## Source Code

```java
// Ahmed Thahir 2020A7PS0198U

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class p06
{
	static String[] subList = new String[6];
	
	public static void init()
	{
		for(int i=0; i<6; i++)
		subList[i] = "";
	}
	
	public static int sum(String str, char c)
	{
		int sum = 0;
		int n = str.length();
		if(c == 'c')
		{
			for (int i = 0; i < n; i++)
			{
				if( (str.charAt(i) >= 65 && str.charAt(i) <= 90) || (str.charAt(i) >= 97 && str.charAt(i) <= 122) )
				sum += str.charAt(i);
			}
		}
		else if (c == 'n')
		{
			for (int i = 0; i < n; i++)
			{
				if( (str.charAt(i) >= 48) && (str.charAt(i) <= 57) )
				sum += str.charAt(i);
			}
		}
		
		return sum;
	}
	public static int hash(String record, int csum, int nsum)
	{
		int hash = ( (csum + 2*nsum) * 17 + 5 ) % 6;
		return hash;
	}
	
	public static void input() throws FileNotFoundException
	{
		String inputFile = "source.txt";
		Scanner readMyFile = new Scanner( new File(inputFile) );
		
		while (readMyFile.hasNext()) 
		{
			String record = readMyFile.nextLine();
			int csum = sum(record, 'c');
			int nsum = sum(record, 'n');
			
			int hash = hash(record, csum, nsum);
			subList[hash] += record + " ";
			
			System.out.println("The hash value of " + record + " is " + hash);
		}
		
		readMyFile.close();
	}
	
	public static void displaySubsets()
	{
		System.out.println("\n\n");
		for(int i = 0; i<6; i++)
		{
			System.out.println("The subset of " + i + " is " + subList[i]);
		}
	}
	
	public static void main(String[] args) throws FileNotFoundException
	{
		init();
		input();
		displaySubsets();
	}
}
```

## Test Cases

### Input

```
M2y
N3x
F4w
O5v
D2u
A2t
K5y
M6z
N7a
Y3w
b2Y
e3X
f4W
c5V
d2U
a2T
J5Y
m6Z
n7A
y3W
```

### Output

```
The Hash value of M2y is 1
The Hash value of N3x is 5
The Hash value of F4w is 0
The Hash value of O5v is 2
The Hash value of D2u is 2
The Hash value of A2t is 0
The Hash value of K5y is 3
The Hash value of M6z is 4
The Hash value of N7a is 2
The Hash value of Y3w is 1
The Hash value of b2Y is 0
The Hash value of e3X is 2
The Hash value of f4W is 0
The Hash value of c5V is 2
The Hash value of d2U is 2
The Hash value of a2T is 0
The Hash value of J5Y is 0
The Hash value of m6Z is 4
The Hash value of n7A is 2
The Hash value of y3W is 1

The Subset of 0 : F4w A2t b2Y f4W a2T J5Y
The Subset of 1 : M2y Y3w y3W
The Subset of 2 : O5v D2u N7a e3X c5V d2U n7A
The Subset of 3 : K5y
The Subset of 4 : M6z m6Z
The Subset of 5 : N3x
```