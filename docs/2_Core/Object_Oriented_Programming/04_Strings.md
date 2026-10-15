## Strings

immutable - you cannot change anything directly

`Object` and `String` are interchangeable

``` java
// both are valid
Object test1 = "hello";
String test2 = "hello"; 

// strings
String s1 = "hello world";
String s = new String("hello world");

// length
s.length(); // 4

// accessing char
s.charAt(3);
s.substring(i); // i to end of the string
s.substring(i, j); // i to j-1

//concat
s1 += s2;
// DONT FORGET s1 = 
s1 = s1.concat(s2); 
// s2 is added to s1 and returns it
// basicaly s1 is the object in focus, and it is getting modified

// search
// returns int position of the 1st char of the string
s1.indexOf("HELLO"); // the string is case-sensitive
s1.indexOf(s2); // searches for s2 inside s1
s1.indexOf(s2, 3); // (string, startIndex)

"Hello".equals("hello"); // returns false
"Hello".equalsIgnoreCase("hello"); // returns true

char[] dh = s1.toCharArray();

Integer.parseInt(numberString); // String to int
Float.parseFloat(numberString);

Integer.toString(numbervar); // int to String
Float.toString(numbervar);
Double.toString(numbervar);
Boolean.toString(numbervar);
```

## Comparing Strings

``` java
s1 = "hello";
s2 = "hello";

// equality
if(s1 == s2) // compares address
if( s1.equals("hello") ) // compares characters

if( s1.compareTo("hello") ) // compares objects including strings
```

## Char Array

``` java
char ch = 'w';
char[] dh = { ch, 'o', 'r', 'd'};

char dh[] = new char[ buffer.length() ];

//char array
dh = s1.toCharArray();
Arrays.equals(dh1, dh2); // 2 char arrays
```

### Char array vs String

Char array is the primitive strings we have in C/C++, but it doesn't have the advanced features of the String objects in java

Another difference is that String objects have automatically-included `\0`

## `StringBuffer`

gives you the best of strings and char array; they are basically mutable Strings

``` java
StringBuffer sb = new StringBuffer("hello there");

String s = sb.toString();
```

you can append, set without creating a new String each time

==default capacity = 16== (not size)
has a minimum allocation of memory for 16 characters; $\ge 16$ is valid

- length = current number of characters
- capacity = max length

### Functions

- `boolean length()` returns length of StringBuffer
- `boolean capacity()` returns capacity of 
- `setLength(length)`
- `ensureCapacity()`
- `charAt(index)` return the character at index
- `setCharAt(index, newchar)` change the character at index
- `getChars(startIndex, nOfCharacters, arrayName, 0)` takes substring and returns as character array
- `reverse()` reverses string
- `append(string, startIndex, endIndex)` append data
- `String insert(insertPos, arrayName, startIndex, endIndex)`
- `deleteCharAt(index)` remove a character
- `delete(startIndex, endIndex)` remove a substring

## `StringTokenizer`

partitions String into individual substring(s)

constructor takes 2 parameters

1. input string
2. delimiter `(eg: comma, space, colon)`

`import java.util.StringTokenizer`

``` java
String s = "https://ahmedthahir.tk";
StringTokenizer st = new StringTokenizer(s, "://."); // order doesn't matter, but the pattern matters

System.out.println(st.countTokens());
while(st.hasMoreTokens())
  System.out.println(st.nextToken());
```