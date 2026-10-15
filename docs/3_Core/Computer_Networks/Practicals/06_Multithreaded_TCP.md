## `Server.java`

```java
// import java.util.*;
import java.io.*;
import java.net.*;

public class Server {
	private ServerSocket server_socket;

	public Server(int port_no) throws IOException {
		server_socket = new ServerSocket(port_no);

		System.out.println("Waiting for client to connect");

		while (true) {
			try {
				Socket connection = server_socket.accept();
				System.out.println("Connected to client");

				ClientHandler client_handler = new ClientHandler(connection);
				client_handler.start();
			} catch (Exception e) {
        e.printStackTrace();
			}
		}
	}

	public static void main(String[] args) throws IOException {
		new Server(5000);
	}
}

class ClientHandler extends Thread {

	public ClientHandler(Socket connection) throws IOException {
		try {
			DataOutputStream out = new DataOutputStream(connection.getOutputStream());
			String server_message = "Hi there! Enter something";
			out.writeUTF(server_message);

			DataInputStream in = new DataInputStream(connection.getInputStream());

			int num1 = Integer.parseInt(in.readUTF());
			String op = in.readUTF();
			int num2 = Integer.parseInt(in.readUTF());

			String client_reply = "Received: " + num1 + op + num2;
			System.out.println(client_reply);

			int result = 0;

			switch (op) {
				case "+": {
					result = num1 + num2;
					break;
				}
				case "-": {
					result = num1 - num2;
					break;
				}
				case "*": {
					result = num1 * num2;
					break;
				}
				case "/": {
					result = num1 / num2;
					break;
				}
				case "^": {
					result = num1 ^ num2;
					break;
				}
				default: {
					System.out.println("Invalid Operator");
				}
			}

			out.writeUTF(Integer.toString(result));

			in.close();
			out.close();
			connection.close();

		} catch (Exception e) {
      e.printStackTrace();
		}
	}
}
```

## `Client.java`

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
		new Client("localhost", 5000);
	}
}
```

