## Simple Example

### `Server.java`

```java
import java.util.*;
import java.io.*;
import java.net.*;

public class Server extends Thread {
  private ServerSocket server_socket;

  public Server(int port_no) throws IOException {
    while (true) {
      try {
        server_socket = new ServerSocket(port_no);

        System.out.println("Waiting for client to connect");

        Socket server = server_socket.accept();
        System.out.println("Connected to client");

        DataOutputStream out = new DataOutputStream(server.getOutputStream());
        String server_message = "Hi there! Enter something";
        out.writeUTF(server_message);

        DataInputStream in = new DataInputStream(server.getInputStream());
        String client_reply = in.readUTF();
        System.out.println("User replied with: " + client_reply);
        server.close();
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }

  public static void main(String[] args) throws IOException {
    Thread t1 = new Server(5000);
    t1.start();
  }
}
```

### `Client.java`

```java
import java.util.*;
import java.io.*;
import java.net.*;

public class Client
{
  public Client(String host, int port) throws IOException
  {
    try
    {
      Socket client = new Socket(host, port);

      System.out.println("Connected to server");

      DataInputStream in = new DataInputStream(client.getInputStream());

      String server_message = in.readUTF();
      System.out.println(server_message);

      DataOutputStream out = new DataOutputStream(client.getOutputStream());

      Scanner user_in = new Scanner(System.in);
      String client_reply = user_in.next();
      out.writeUTF(client_reply);

      user_in.close();
      client.close();
    } catch (Exception e)
    {
      e.printStackTrace();
    }
  }
  public static void main(String[] args) throws IOException
  {
    new Client("localhost", 5000);
  }
}
```

### Execution Commands

```bash
javac Server.java
java Server

javac Client.java
java Client
```

### Output

#### Server

```

```

#### Client

```

```

## Calculator

### `Server.java`

```java
import java.util.*;
import java.io.*;
import java.net.*;

public class Server extends Thread {
	private ServerSocket server_socket;

	public Server(int port_no) throws IOException {
		while (true) {
			try {
				server_socket = new ServerSocket(port_no);

				System.out.println("Waiting for client to connect");

				Socket server = server_socket.accept();
				System.out.println("Connected to client");

				DataOutputStream out = new DataOutputStream(server.getOutputStream());
				String server_message = "Hi there! Enter something";
				out.writeUTF(server_message);

				DataInputStream in = new DataInputStream(server.getInputStream());
				
				int num1 = Integer.parseInt(in.readUTF());
				String op = in.readUTF();
				int num2 = Integer.parseInt(in.readUTF());
				
				String client_reply = "Received: " + num1 + op + num2;
				System.out.println(client_reply);
        
				int result = 0;
				
				switch(op)
				{
					case "+": {result = num1 + num2; break;}
					case "-": {result = num1 - num2; break;}
					case "*": {result = num1 * num2; break;}
					case "/": {result = num1 / num2; break;}
					case "^": {result = num1 ^ num2; break;}
					default: {System.out.println("Invalid Operator");}
				}

				out.writeUTF(Integer.toString(result));

				server.close();
			} catch (Exception e) {
        e.printStackTrace();
			}
		}
	}

	public static void main(String[] args) throws IOException {
		Thread t1 = new Server(args[0]);
		t1.start();
	}
}
```

### `Client.java`

```java
import java.util.*;
import java.io.*;
import java.net.*;

public class Client {
	public Client(String host, int port) throws IOException
  {
    try
    {
      Socket client = new Socket(host, port);

      System.out.println("Connected to server");

      DataInputStream in = new DataInputStream(client.getInputStream());

      String server_message = in.readUTF();
      System.out.println(server_message);

      DataOutputStream out = new DataOutputStream(client.getOutputStream());

      Scanner user_in = new Scanner(System.in);

	  int num1 = user_in.nextInt();
	  String op = user_in.next();
	  int num2 = user_in.nextInt();

	  out.writeUTF(Integer.toString(num1));
	  out.writeUTF(op);
	  out.writeUTF(Integer.toString(num2));

	  
	  String result = in.readUTF();
	  System.out.println(result);
      
      user_in.close();
      client.close();
    } catch (Exception e)
    {
      e.printStackTrace();
    }
  }

	public static void main(String[] args) throws IOException {
		new Client(args[0], args[1]);
	}
}
```

### Execution Commands

```bash
javac Server.java
java Server 5000

javac Client.java
java Client localhost 5000
```

### Output

#### Server

```

```

#### Client

```

```

## Working with Telnet

1. no need to implement your own client
2. change server’s parameters as command-line arguments (using `args[0], ...`)
3. Need to use `BufferedReader` and `PrintWriter` as the encoding used Telnet is different

```java
import java.util.*;
import java.io.*;
import java.net.*;

public class Server extends Thread {
  private ServerSocket server_socket;

  public Server(int port_no) throws IOException {
    while (true) {
      try {
        server_socket = new ServerSocket(port_no);

        System.out.println("Waiting for client to connect");

        Socket server = server_socket.accept();
        System.out.println("Connected to client");

        PrintWriter out = new PrintWriter(server.getOutputStream());
        String server_message = "Hi there! Enter something";
        out.print(server_message);

        BufferedReader in = new BufferedReader(new InputStreamReader(
        	server.getInputStream()
        ));
        String client_reply = in.readLine();
        
        System.out.println("User replied with: " + client_reply);
        server.close();
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }

  public static void main(String[] args) throws IOException {
    Thread t1 = new Server(5000);
    t1.start();
  }
}
```

```bash
javac Server.java
java Server 5000
telnet localhost 5000
```

