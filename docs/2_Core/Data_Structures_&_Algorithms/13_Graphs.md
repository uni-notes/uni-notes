[Graphs in Discrete Structures](../Discrete_Structures/04_Graphs.md) 

Graphs without parallel edges and self loops are called as simple graphs.

## Representations

### Adjacency Matrix (Array)

$O(n^2)$

|        | **9** | **7** | **40** | **60** |
| :----: | :---: | :---: | :----: | :----: |
| **9**  |   1   |   0   |   1    |   0    |
| **7**  |   1   |   1   |   1    |   1    |
| **40** |   0   |   0   |   1    |   1    |
| **6**  |   0   |   1   |   0    |   1    |

### Adjacency List (Linked List)

More efficient, as $O(n)$

(diagram)

## Applications

1. Networks
     - Computer Networks
     - Transportation
2. Computer Vision
   Pixels

## Connected Graph/Components

## Traversals

| BFS           | DFS         |
| ------------- | ----------- |
| Breadth First | Depth First |
| Queue         | Stack       |

### Trick to remember

If the person is a **Queue-t**, they’ll take your **breadth** away.

If it is a **stack**, it has a **depth** associated with it.

## Single Source Shortest Path

Path from a single start point to every other point in the graph

### Dijkstra’s algorithm

each step has connected components

Time complexity: $O(v^2)$

If you use minimum priority queue, it’ll be $O(v+e \log v)$

#### Disavantages

It requires

1. Simple graph
2. Connected graph
3. Positive Weights only

## Minimum Spanning Tree

Refer [Discrete Structures](./../Discrete_Structures/05_Trees.md#minimum-spanning-tree)

### Prim’s Algorithm

Refer [Discrete Structures](./../Discrete_Structures/05_Trees.md#prims-algorithm)

### Kruskal’s Algorithm

Refer [Discrete Structures](./../Discrete_Structures/05_Trees.md#kruskals-algorithm)
