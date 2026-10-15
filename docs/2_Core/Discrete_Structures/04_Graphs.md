## Graphs

Here, we are talking about undirected graphs

All undirected graphs are symmetric

Adjacency matrix of non-directed graph is a symmetric matrix $(A = A')$

## Basic Concepts

A graph is represented as $G = (V, E)$ where

- V = set of vertices
- E = set of edges; the edges are undirected

Loops are allowed; graph with no loops is called as simple/loop-free graph

maximum degree of a vertex in a simple graph $= |V| - 1$

$|V(g)| = |V| =$ order of G = no of vertices in the graph G

$|E(g)| = |E| =$ size of G = no of edges in the graph G

No of edges $= \frac{\sum deg(v_i)}{2}$

n vertices can only have $n-1$ adjacent vertices

Graphic sequence = deg sequence from which valid graph is possible
Non-graphic sequence = deg sequence from which graph is not possible

no of labelled graphs on a given set of n vertices $= 2^{nC_2}$
out of them, $(nC_2)C_m$ contain m edges

when n = 3, 4 non-isomorphic graphs are possible
when n = 4, 11 non-isomorphic graphs are possible

### Multigraph

is a graph with more than one edge b/w a pair of vertices

### Degree Sequence

is the set of degrees of the vertices

Loop is taken as an increment of two (as it starts and ends at the same place)

it is written in ascending order: from lowest degree to highest degree
$\delta(G) \to \Delta (G)$

### Regular Graph

==loop-free== graphs where every vertex has the same degree

$\delta(G) = \Delta(G)$

The name of the graph $= (|V| - 1)$ Regular graph
Eg: for 5 vertices graph, if all the 5 vertices are connected to the others, then the name will be "4 - regular graph"

$|E| = \frac{n \times d}{2}, d =$ the degree of every vertex

## Theorems

### Non-directed graph

If $V = {v_1, v_2, v_3, \dots, v_n}$ is the vertex set of a non-directed graph, then $\sum\limits_{i=1}^n deg(v_i) = 2 |E|$

Each element contributes a count of one to the degree of each of the two vertices on which it is incident

### Directed Graph

$\sum\limits_{i=1}^n deg^+(v_i) = \sum\limits_{i=1}^n deg^-(v_i) = |E|$

## Cor(1)

In any non-directional graph, there is an even number of vertices of odd degree 

If W: set of vertices of G with odd degree, U: set of vertices of G with even degree

$$
\sum\limits_i deg(V_i) = 2|E| \\
\sum\limits_{i \in W} deg(V_i) + \underbrace{ \sum\limits_{i \in U} deg(V_i)}_\text{even} = \underbrace{2|E|}_\text{even} \\
\implies \sum\limits_{i \in W} deg(V_i) \text{ is also even}
$$

But W contains all vertices with odd degree. $\therefore,$ the no of vertices in W should be even. Hence, |W| is also even.

## Cor(2)

If $k = \delta (G)$ is the minimum degree of all vertices of G, then $k|V| \le \sum\limits_{i=1}^n deg(v_i)$

In particular, if G is a k-regular graph (where the degree of all the vertices is k), then $k|V| = \sum\limits_{i=1}^n deg(v_i) = 2|E|$

## Path

In a graph G, a sequence P of zero/more edges of the form $\set{v_0, v_1}, \set{v_1, v_2}, \dots, \set{v_{n-1}, v_n}$

### Graphical Representation

``` mermaid
graph LR
v0 --- v1 --- v2 --- ... --- Vn-1 --- Vn
```

is called a path from $v_0$ to $v_n$

### Length

the number of edges in path p

### Notes

In a path, vertices and edges

1. may be repeated
2. If $v_0 = v_n$, then path p is closed
   $v_0 = v_n$, then path p is open
3. a path p is itself a graph, ie, subgraph of G
   1. $V(P) \subseteq V(G)$
   2. $E(P) \subseteq E(G)$
4. Path may have no edges at all
   1. length = 0 $(V(P) = \set{v_0})$
   2. trivial path (simple, closed path)

## Simple path

Path with all distinct edges and vertices
end points(vertices) of a ==**closed**== path are exempted from this condition

## Circuit

1. closed path
2. length $\ge 1$
3. no repeated edges
4. end points are equal (is repeated?)
5. it ***may*** have repeated vertices

## Cycle

simple circuit
no repeated vertices (except start and end points)

## Wheel

Cycle with 1 vertex connected to all other vertices
the vertex doesn’t necessarily have to be inside the cycle

## Complete Graph $k_n$

every vertex is connected with every other vertex

if $|V| = n$, deg of every vertex $= n-1$

``` mermaid
graph LR

subgraph k1
    a(( ))
end

subgraph k2
    b(( )) --- c(( ))
end

subgraph k3
    d(( )) --- e(( )) --- f(( )) --- d
end

subgraph k4
    g(( )) --- h(( )) --- i(( )) --- j(( )) --- g
    h --- j
    i --- g
end
```

## Linear graphs $L_n$

Open graph
$|V| = n$

``` mermaid
graph LR

subgraph L2
    a(( )) --- b(( ))
end

subgraph L5
    c(( )) --- d(( )) --- e(( )) --- f(( )) --- g(( ))
end
```

|               | Closed | Open | Circuit | Cycle | Wheel | Regular graph       |      Complete Graph       |
| ------------- | :----: | :--: | :-----: | :---: | :---: | ------------------- | :-----------------------: |
| \vert V \vert         |   n    |  n   |    n    |   n   |       | n                   |             n             |
| deg of vertex |        |      |         |       |       | d                   |            n-1            |
| \vert E \vert         |   n    | n-1  |    n    |   n   |       | $\frac{n \times d}{2}$ | $\frac{n(n-1)}{2} = nC_2$ |

## Theorem

In a graph G, every u-v path contains a simple u-v path

### Proof

Mathematical induction

Taking a u-v path. It can either be

- closed
  
    obviously contains a trivial path (of length 0)
  
    simple path

- open
  
    Consider an open u-v path
  
    To show it contains a simple u-v path

    ``` mermaid
    graph LR
    u((u / v0)) --- v1((v1)) --- v2((...)) --- v((v / vn))
    ```

    Proof by induction on the length of path p

    - Length = 1 (basic)

    - then path p is open as it contains only one edge

    - length = k, where $1 \le k \le n$ (induction hypothesis)

    - assume that when the length is k, then u-v path contains a simple u-v

    - length = n+1 (induction proof)

    - trying to prove that it contains a simple path (using the induction hypothesis)

        ``` mermaid
        graph LR
        u((u / v0)) --- v1((v1)) --- v2((...)) --- vn((vn)) --- v((v / vn+1))
       ```

        this path

        - has no repeated vertices $\implies$ it is simple

      - contains repeated vertices

        - Let $v_i = v_j$ be the vertices for $i < j$
          $v_0 - v_1 - \ldots - v_i - v_{i+1} - \ldots - v_j - v_{j+1} - \ldots - v_{n+1}$
        - remove $v_{i+1} - \ldots - v_j$ from P

    - now, $v_{0} - v_1 - \ldots - v_i - v_{j+1} - \ldots - v_{n+1}$ is a simple path

Hence, proved

## Isomorphism

denoted by $\cong$

2 graphs G1 and G2 are isomorphic if there is a function $f: V(G_1) \to V(G_2)$ such that

1. f is one-one
2. f is onto
3. f preserves adjacency of vertices
4. $\forall (u,v) \in E(G_1) \implies (f(u), f(v)) \in E(G_2)$

f need not be unique; there can be various mappings that preserve adjacency

### Implications of isomorphism

1. $| V(G_1) | = | V(G_2) |$
2. $| E(G_1) | = | E(G_2) |$
3. deg seq(G1) = deg seq(G2)
4. Loops: $(v,v) \in E(G_1) \implies (\ f(v), f(v) \ ) \in E(G_2)$
5. if there is a cycle of length n in G1, ie $v_0 - v_1 - \ldots - v_k (=v_0)$
   then $f(v_0) - f(v_1) - \ldots - f(v_k) (=f(v_0))$ is also a cycle of length n in G2
6. Cycle vector $\set{c_1, c_2, \dots, c_k}$ of G = cycle vector $\set{d_1, d_2, \dots, d_k}$ where
   1. cn = cycle of length n
   2. dn = cycle of length n
7. the induced subgraphs (by a set W) of isomorphic graphs are also isomorphic
8. even the complements of the graphs are isomorphic

## Incident Matrix

Let G = (V, E) be an undirected graph with n vertices and m edges

$B_{n \times m} = [b_{ij}]$ is called the incident matrix of G, where

$$
b_{ij} = \begin{cases}
1, \text{ when } e_j \text{ is incident on } v_i \\
0, \text{ otherwise}
\end{cases}
$$

## Subgraph

H is a subgraph of G $\iff V(H) \subseteq V(G)$ and $E(H) \subseteq E(G)$

## Spanning subgraph

$\iff V(H) = V(G)$ and $E(H) \subseteq E(G)$

### Minimal Spanning subgraph

spanning subgraph with minimum no of edges required to make the graph connected

removal of any ==edge== makes the subgraph disconnected

need not be a unique; there can many variations of subgraphs with the above property

## Induced Subgraph

it’s the subgraph using only vertices contained in set W and all the pre-existing edges

If $W \subseteq G$, then the subgraph induced by W in G is the one with the vertices set W and contains all edges connecting a pair of vertices in W

## Complement of graph

If H is a simple graph with n vertices, then complement denoted as $\bar H$ of H is the complement of H in $k_n$, where $k_n =$ complete graph with n vertices

$V(\bar H) = V(H)$

2 vertices in $\bar H$ are adjacent/connected only if they are not adjacent/connected in H

## Complement of subgraph

$\bar H = G - H$

- $V(\bar H) = V(H)$
- $E(\bar H) = E(G) - E(H)$

## Operations on Graphs

- $G_1 \cap G_2$ is a graph with

    - vertices set $V(G_1) \cap V(G_2)$
    - edge set $E(G_1) \cap E(G_2)$

- $G_1 \cup G_2$ is a graph with
    - vertices set $V(G_1) \cup V(G_2)$
    - edge set $E(G_1) \cup E(G_2)$
- $\bar G \cup G = k_n$
- $\bar G \cap G = N_7$, where $N_7$ is a null graph (with vertices but no edges)

## Connected graph

if there is a path from any vertex to any other vertex in that graph

ie every vertex is of degree$\ge 1$

otherwise it is disconnected

## Bipartite Graph

is a simple graph in which V(G) can be partitioned into 2 sets M and N, such that

1. if vertex $v \in M,$ then it can only be adjacent to vertices in N
2. If vertex $v \in N,$ then it can only be adjacent to vertices in M
3. $M \cap N = \phi$
4. $M \cup N = V(G)$

When drawing the graph, by convention, M comes up and N comes down

### Properties

1. $\sum\limits_{v \in M} deg(v) = \sum\limits_{v \in n} deg(v)$
2. A bipartite graph contains no odd cycles
3. Every subgraph of a bipartite graph is also bipartite
4. each edge joins a vertex in M to a vertex in N

### Complete Bipartite graph

Every vertex of M is connected to every vertex of N, and vice-versa

G is denoted as $k_{m, n}$

if $|M| = m, |N| = n$, then $|V(G)| = m+n, |E(G)| = mn$

in order to traverse a cycle, you need to traverse even no of edges
