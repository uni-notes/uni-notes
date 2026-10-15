## Initial Steps

1. Go to Windows Features
2. Turn on `Internet Information Services`
3. Host the web page
   1. Go to `Windows (C:)>inetpub>wwwroot`
   2. Add a file called `index.html`
   3. Type your html code

4. Perform [Execution](#Execution)

## `Client.java`

```java
import java.util.*;
import java.io.*;
import java.net.*;

public class Client
{
	public static void main(String[] args) throws IOException
	{
		String server = args[0];
		int port = Integer.parseInt(args[1]);

		try
		{
			Socket socket = new Socket(server, port);

			DataOutputStream request = new DataOutputStream(socket.getOutputStream());
			request.writeUTF("\n");

			DataInputStream response = new DataInputStream(socket.getInputStream());
			
			String response_text = response.readUTF();
			while(response_text != null)
			{
				System.out.println(response_text);
				response_text = response.readUTF();
			}

			request.close();
			response.close();
			socket.close();
		} catch (Exception e)
		{
			e.printStackTrace();
		}
	}
}
```

## Execution

```
javac Client.java
java Client localhost /
```

