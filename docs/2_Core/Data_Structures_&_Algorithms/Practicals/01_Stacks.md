## Question

Write a C/C++/JAVA program to perform the following actions on a STACK implemented using arrays / array of structures:

1. Implement PUSH operation in a STACK for N (N >= 5) STUDENT RECORDS. Each STUDENT RECORD should store <IDNO, NAME, DOB,CGPA&gt;. You have to read each STUDENT record from an input file “studentin.dat” stored locally in your directory and the PUSH it into the stack, one at a time. (You can use vi editor to create input data file. Make sure that the input data file contains at least 5 records).
2. Implement the POP operation for the STACK in LIFO order and display all the records on the standard output (screen). Also, write the output results into an external file “studentout.dat”.

## Algorithm

### Pseudocode

```pseudocode
Algorithm push
		INPUT read student records from file
		OUTPUT student records to stack
		
		while inputFile has records
      if top = n
        overflow
      else
        top <- top + 1
        studentArray[top] <- record

Algorithm pop
		INPUT student records from stack
		OUTPUT write student records to file
		
		while studentArray has records
        print record
        write to outputFile
        top <- top - 1
```

### Time Complexity

| Algorithm | Complexity |
| :-------: | :--------: |
|   push    |   $O(n)$   |
|    pop    |   $O(n)$   |

## Source Code

```java
// Ahmed Thahir 2020A7PS0198U

package Programs;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Scanner;

public class p01 
{
  static int top = -1;
  static int capacity = 10;
  static String[] students = new String[capacity];

  public static void push() throws FileNotFoundException
  {
    String inputFile = "h:\\
My Drive\\
Notes\\
Sem 4\\
02 DSA\\
Practicals\\
Programs\\
"
      + "studentin.dat";
    Scanner readMyFile = new Scanner( new File(inputFile) );

    while (readMyFile.hasNext()) 
    {
      if(top != capacity)
      {
        top = top + 1;
        students[top] = readMyFile.nextLine();
      }
    }

    readMyFile.close();
  }

  public static void pop() throws FileNotFoundException
  {
    String outputFile = "h:\\
My Drive\\
Notes\\
Sem 4\\
02 DSA\\
Practicals\\
Programs\\
"
      + "studentout.dat";
    PrintWriter writeToMyFile = new PrintWriter( new File(outputFile) );

    while(top != -1)
    {
      System.out.println(students[top]);
      writeToMyFile.format("%s \n", students[top]);

      top = top-1;
    }

    writeToMyFile.close();
  }

  public static void main(String[] args) throws FileNotFoundException
  {
    push();
    pop();
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
2021A7PS005 EEEE 5/1/2000 9.25
2021A7PS004 DDDD 4/1/2000 8.75
2021A7PS003 CCCC 3/1/2000 9.60
2021A7PS002 BBBB 2/1/2000 9.20
2021A7PS001 AAAA 1/1/2000 7.50
```
