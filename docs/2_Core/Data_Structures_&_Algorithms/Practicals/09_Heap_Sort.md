## Code

```java
class Heap
{
	String[] nodes = new String[100];
	int size = 0;
	
	Heap()
	{
		nodes[0] = "";
	}

	int parent(int pos)
	{
		return pos/2;
	}
	int lc(int pos)
	{
		return 2*pos;
	}
	int rc(int pos)
	{
		return 2*pos + 1;
	}

	void swap(int a, int b)
	{
		String t = nodes[a];
		nodes[a] = nodes[b];
		nodes[b] = t;
	}

	void insert(String data)
	{
		size++;
	
		nodes[size] = data;

		int cur = size;
		while(
			nodes[cur].compareTo(nodes[parent(cur)]) > 0
		)
		{
			swap(cur, parent(cur));
			cur = parent(cur);
		}
	}

	void max_heapify(int size, int root)
	{
		int largest = root,
				l = lc(root),
				r = rc(root);

		if(
			l<size && nodes[l].compareTo(nodes[largest])>0
		)
			largest = l;
		if(
			r<size && nodes[r].compareTo(nodes[largest])>0
		)
			largest = r;

		if(root != largest)
		{
			swap(root, largest);
			max_heapify(size, largest);
		}
	}

	void sort()
	{
		// build heap
		for(int i = size; i>=0; i--)
		{
			swap(i, 0);
			max_heapify(i, 0);
		}
	}

	void display()
	{
		for(int i=0; i<=size; i++)
		{
			if(nodes[i].length() > 0)
				System.out.print(nodes[i] + " ");
		}
	}
}

class p06
{
	public static void main(String[] args)
	{
		Heap heap = new Heap();

		heap.insert("CC");
		heap.insert("DF");
		heap.insert("MM");
		heap.insert("AB");
		heap.insert("ZX");
		heap.insert("PQ");
		heap.insert("LR");
		heap.display();

		System.out.println();
		heap.sort();
		heap.display();
	}
}
```

## Input

```
RR
BB
YY
GG
NN
QQ
MM
PP
BB
AA
KT
UV
VV
GG
QQ
MN
PQ
RS
TU
YM
```

## Output

```
YY YM VV UV RS TU RR QQ NN PQ KT GG BB QQ GG MM MN BB PP AA
AA BB BB GG GG KT MM MN NN PP PQ QQ QQ RR RS TU UV VV YM YY
```

