## Just the graph building + outgoing nodes part

```java
import java.util.ArrayList;

class Graph
{
	int size;
	ArrayList adj[];

	Graph(int size)
	{
		this.size = size;

		adj = new ArrayList[size];

		for(int i = 0; i<size; i++)
		{
			adj[i] = new ArrayList<Integer>();
		}
	}

	void insert(int u, int v)
	{
		int ul = u-65;
		int vl = v-65;
		adj[ul].add(vl);
	}

	void outgoing(char from)
	{
		System.out.println("Nodes outgoing from " + from);

		int u = (char)(from) - 65;
		for(int i=0; i<adj[u].size(); i++)
		{
			int v = (int) adj[u].get(i);
			char ch = (char) (v + 65);
			System.out.println(ch);
		}
	}
}

class p06
{
	public static void main(String[] args)
	{
		Graph graph = new Graph(8);

		graph.insert('A', 'B');
		graph.insert('A', 'E');
		graph.insert('B', 'C');
		graph.insert('D', 'E');
		graph.insert('A', 'D');

		graph.outgoing('A');
	}
}
```

## Output

```
Nodes outgoing from A
B
E
D
```

