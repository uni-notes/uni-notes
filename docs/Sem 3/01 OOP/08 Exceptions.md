## Exceptions

warnings, not exactly errors

|                        | Checked      | Unchecked        |
| ---------------------- | ------------ | ---------------- |
| handled at             | compile-time | runtime          |
|                        |              |                  |
| inherited from (class) | Exception    | RuntimeException |
|                        |              |                  |

## Exception

For accessing out of bounds array data, JVM throws `ArrayIndexOutOfBoundsException`

1. JVM will show a warning
2. it will just skip the exception
3. then proceed with the rest of the program

## Exception-Handling

use `try-catch`

- logic in `try`
- exception will be caught by `catch`
- `finally` block contains all statements that must be executed when exception does or does not occurs

IDK

1. neither can exist independently, but not finally is not compulsory
2. nested try is possible, but nested catch is not
3. nothing can come up after `finally` - unreachable catch block error

``` java
try {
  // code to test
  
  try {
    // something
  }
  catch(Exception E)
  {
    //something
  }
}
catch(ArrayIndexOutOfBoundsException e) {
  //
}
catch(Exception2 e2) {
  //
}
catch(Exception e) { // all exceptions (checked/unchecked)
  //
}
finally {
	//
}

System.out.println("Program done"); // doesn't get executed
```

here, statement1 runs, but statement2 doesn’t. this is because, the flow of control goes to the catch after the throw

Similarly, the last statement doesn’t get executed because of `finally` block

### `throw` and `throws`

you can explicitly throw any kind of exception

can come with/without `try-catch`

|                            | `throw`                                                     | `throws`                                                     |
| -------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| no of exceptions at a time | only one                                                    | multiple                                                     |
|                            |                                                             | i’m not sure this, but<br />doesn’t actually throw - just shows that the function **might** throw |
| location                   | function definition                                         | function prototype                                           |
|                            | can come inside `throws`                                    | cannot come inside `throw`                                   |
| type of exception          | unchecked                                                   | checked/unchecked                                            |
| followed by                | exception instance                                          | exception class                                              |
| example                    | - `throw new ArithmeticException(“blah”);`<br />- `throw e` | `void test() throws IOException{}`                           |

## IDK

error

- Compile time - Syntax errors
- Runtime error - wrong constructor for initialization

exceptions

- Runtime exception 
    - unexpected values - divide by 0
    - array index out of bound

## Custom Exceptions

``` java
class CustomException extends IllegalArgumentException
{
  String message = "Blah";
  CustomException(String s)
  {
    super(s); // or super(message);
  }
  @Override
  public String toString() 
  {
    return message;
  }
}

// somewhere else
try {
  throw new CustomExcepiton("specific message"); // prints specific message
} catch(CustomException e) {
  System.out.println(e); // prints Blah
}
```

## Common Exceptions

1. `ArithmeticException`
2. `ArrayIndexOutOfBoundException`
3. `IOException`
4. `NullPointerException`
5. `StringIndexOutOfBoundsException`
6. `FileNotFoundException`
7. `NumberFormatException`

## Exception Methods

```java
e.func();

public String getMessage();
// inside System.out.println()
// details of why the exception happened
// eg: / by zero

public String toString(); // name + getMessage()
// eg: java.lang.ArithmeticException / by zero

public void printStackTrace();
// outside System.out.println()
// toString() + location of exception
// eg: java.lang.ArithmeticException / by zero at Test.main(Testjava:9)

public Throwable getCause(); // toString()
// eg: java.lang.ArithmeticException / by zero
```
