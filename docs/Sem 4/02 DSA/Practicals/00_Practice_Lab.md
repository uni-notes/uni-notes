## Java File Handling

```java
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Scanner;
public class f {
  public static void main(String[] args) throws
  FileNotFoundException {
    //Scanner method of reading files (Since JDK 4)
    //Open file for reading contents
    System.out.println( & quot; Your file should be placed at: & quot; +
      System.getProperty( & quot; user.dir & quot;));
    Scanner readMyFile = new Scanner(new File(args[0]));
    //Open file for writing content
    System.out.println( & quot; Output file will be created at: & quot; +
      System.getProperty( & quot; user.dir & quot;));
    PrintWriter writeToMyFile = new PrintWriter(new File(args[1]));
    while (readMyFile.hasNext()) {
      // Read the content of input file
      // Read 3 integers
      int a = readMyFile.nextInt();
      int b = readMyFile.nextInt();
      int c = readMyFile.nextInt();
      //Read the string
      String name = readMyFile.next(); //not
      readMyFile.nextLine()
      //Read a float
      float f = readMyFile.nextFloat();
      //Display the content of input file
      System.out.printf( & quot; % d % d % d % s % f % n & quot;, a, b, c, name, f);
      //You can also use System.out.print to display one data
      type at a time.
      /*
      * %n is a new line character appropriate to the
      platform running the application.
      * You should always use %n, rather than \n.
      */
      //Write data to file
      int result = (a * a) + (b * b) + (c * c);
      System.out.format( & quot;
        (a * a + b * b + c * c) = % d % s % f % n & quot;,
        result, name, f);
      writeToMyFile.format( & quot;
        (a * a + b * b + c * c) = % d % s %
        f % n & quot;, result, name, f);
      /*

      6

      * Again writeToMyFile.print can be use to write one
      data type at a type.
      */
    }
    readMyFile.close();
    writeToMyFile.close();
  }
}
/* now try to read inputs for these fields from keyboard */
```

## Programming Language

Both the below programs were made using Java.

## Question 1

Write a program in your favorite programming language (C/C++/JAVA) to determine if a given Input Number is PERFECT or DEFICIENT or ABUNDANT. Assume that the input number is in the range 1 – 32768 (inclusive at both sides).

A number (consider only positive integers) is **perfect** if it is equal to the sum of its proper divisors. For example, 6 is a perfect number, because its proper divisors are 1, 2, and 3(note that we do not include the number itself), and 1+2+3=6.

A number is **deﬁcient** if the sum of its proper divisors is less than the number. For example, 8 is deﬁcient, because its proper divisors are 1, 2, and 4, and 1 + 2 + 4 = 7, which is less than 8.

A number is **abundant** if the sum of its proper divisors is greater than the number. For example, 12 is abundant, because 1 + 2 + 3 + 4 + 6 = 16, which is greater than 12.

Write a program that prompts the user for a number, then determines whether the number is perfect, deﬁcient, or abundant. Your program should continue to prompt the user for numbers until a 0 is provided as input. An example session:

```
Enter an integer (0 to quit): 7
7 is deficient.
Enter an integer (0 to quit): 12
12 is abundant.
Enter an integer (0 to quit): 6
6 is perfect.
Enter an integer (0 to quit): 0
```

### Algorithm

1. Input number
2. Find factors and their sum
3. Check the various cases. If sum of factors is

   1. $=$ number, then perfect
   2. $<$ number, then deficient
   3. $>$ number, then abundant
4. Print the result

```pseudocode
pseudocode
```

Time complexity is $O(n)$, because of the `for` loop.

### Code

```java
import java.util.Scanner;

class q
{
  public static void checker(int num)
  {
    int factorSum = 0;

    for(int i = 1; i<num; i++)
      if(num % i == 0)
        factorSum += i;

    String text = "";
    if (factorSum == num)
      text = "Perfect";
    else if (factorSum < num)
      text = "Deficient";
    else if (factorSum > num)
      text = "Abundant";
    System.out.println( num + " is " + text + " number");
  }
  public static void main( String args[] )
  {
    Scanner inp = new Scanner( System.in );

    checker(7);
    checker(12);
    checker(6);

    System.out.println("\nInput a number of your wish:");
    int input = inp.nextInt();
    checker(input);
  }
}
```

### Input/Output

```
7 is Deficient number
12 is Abundant number
6 is Perfect number

Input a number of your wish:
18
18 is Abundant number
```

## Question 2

Write a program that inputs two fractions in the form a/b and c/d, and outputs their sum in the form p/q cancelled down to its simplest form. Here, you can read the values of a,b,c,d as input from keyboard and show the output in the simplest form. i.e. numerator / denominator.

```
Input: 5/6 1/10 Output: 14/15
Input: 2/3 4/6 Output: 4/3
Input: 1/2 3/4 Output: 5/4
Input: 1/2 1/2 Output: 1/1
```

### Algorithm

1. Input numbers
2. Obtain the numerator and denominator by cross-multiplication
3. Simplify the numerator and denominator
4. Print the result

```pseudocode
pseudocode
```

Time Complexity is $O(n)$, because of the `for` loop.

### Code

```java
import java.util.Scanner;

class q02 {
  public static void checker(int a, int b, int c, int d)
  {
    int p = a*d + b*c,
      q = b*d;

    int pSim = p,
      qSim = q;
    
    for(int i = Math.min(p, q); i>=2; i--)
      if(p%i == 0 && q%i == 0)
      {
        pSim = p/i;
        qSim = q/i;
        break;
      }

    System.out.println("\n" + pSim + "/" + qSim);
  }
  public static void main(String[] args) {
    Scanner inp = new Scanner( System.in );
    int a, b, c, d;
    System.out.println("\nEnter your values");
    System.out.println("a"); a = inp.nextInt();
    System.out.println("b"); b = inp.nextInt();
    System.out.println("c"); c = inp.nextInt();
    System.out.println("d"); d = inp.nextInt();

    checker(a, b, c, d);
  }
}
```

### Input/Output

```
Enter your values
a
5
b
6
c
1
d
10

14/15
```

