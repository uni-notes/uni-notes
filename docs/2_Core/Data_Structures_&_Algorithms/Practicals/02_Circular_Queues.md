## Question

Write a C/C++ program to perform the following actions on a QUEUE implemented using arrays / array of structures / array-lists (which is viewed circularly):

1. Implement ENQUEUE(o) operation in a QUEUE for N (N &gt;= 5) STUDENT RECORDS. Each STUDENT RECORD should store `<IDNO, NAME, DOB, CGPA>`. You have to read each STUDENT record from an input file “studentin.dat” stored locally in your directory and insert it into the queue, one at a time. (You can use vi editor to create input data file. Make sure that the input data file contains at least 5 records).
2. Implement the DEQUEUE() operation for the QUEUE in FIFO order and display all the records on the standard output (screen display). Also, write the output results into an external file “studentout.dat”.
3. Display the student names (NAME field) whose CGPA is less than 9.

## Algorithm

### Pseudocode

```pseudocode
Algorithm enqueue
		INPUT read student records from file
		OUTPUT student records to queue
	
		while inputFile has records
			if (F = 0 and R = n-1) or (F=R+1)
				overflow
			else if (F = -1 and R = -1)
				F <- 0
				R <- 0
			else if (R = n-1)
				R <- 0
			else
				R <- R+1

			a[R] <- element

Algorithm dequeue
		INPUT student records from queue
		OUTPUT write student records to file
	
		while studentArray has records
				record <- a[F]
				print record
				write to outputFile

				F <- F+1

Algorithm displayNames
		INPUT student records from queue
		OUTPUT write student records to file

		while studentArray has records
				record <- a[F]

				if(cgpa < 9.0)
					print name

				F <- F+1
```

### Time Complexity

|  Algorithm  | Complexity |
| :----------: | :--------: |
|   enqueue   |  $O(n)$  |
|   dequeue   |  $O(n)$  |
| displayNames |  $O(n)$  |

## Source Code

```java
// Ahmed Thahir 2020A7PS0198U

package Programs;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Scanner;

public class p02
{
  static int f, r, n = 10;
  static String[] students = new String[n];

  public static void enqueue() throws FileNotFoundException
  {
    f = -1;
    r = -1;
  
    String inputFile = "h:\\
My Drive\\
Notes\\
Sem 4\\02 DSA\\
Practicals\\
Programs\\"
      + "studentin.dat";
    Scanner readMyFile = new Scanner( new File(inputFile) );

    while (readMyFile.hasNext()) 
    {
      if( (f == 0 && r == n-1) || f == r+1 )
      {
        // overflow
      }
      else if (f == -1 && r == -1)
      {
        f = 0;
        r = 0;
      }
      else if (r == n-1)
        r = 0;
      else
        r++;
    
      students[r] = readMyFile.nextLine();
    }

    readMyFile.close();
  }

  public static void dequeue() throws FileNotFoundException
  {
    String outputFile = "h:\\
My Drive\\
Notes\\
Sem 4\\02 DSA\\
Practicals\\
Programs\\"
      + "studentout.dat";
    PrintWriter writeToMyFile = new PrintWriter( new File(outputFile) );
   
    while( !(f == -1 && r == -1) )
    {
      System.out.println(students[f]);
      writeToMyFile.format("%s \n", students[f]);
    
      if (F == n-1)
				F = 0;
      else if (F == R)
      {
        F = -1;
        R = -1;
      }
      else
				F = F+1;
    }

    writeToMyFile.close();
  }

  public static void displayNames() throws FileNotFoundException
  {
    enqueue();
  
    while( !(f == -1 && r == -1) )
    {
      String[] strArray = students[f].split(" ");  
      float cgpa = Float.parseFloat( strArray[strArray.length -1] );
      String name;
    
      if(cgpa < 9f)
      {
        name = strArray[1];
        System.out.println(name);
      }
    
      if (F == n-1)
				F = 0;
      else if (F == R)
      {
        F = -1;
        R = -1;
      }
      else
				F = F+1;
    }
  }

  public static void main(String[] args) throws FileNotFoundException
  {
    enqueue();
    dequeue();
    displayNames();
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
2021A7PS001 AAAA 1/1/2000 7.50
2021A7PS002 BBBB 2/1/2000 9.20
2021A7PS003 CCCC 3/1/2000 9.60
2021A7PS004 DDDD 4/1/2000 8.75
2021A7PS005 EEEE 5/1/2000 9.25
AAAA
DDDD
```
