## Planar Graph

1. either the graph itself or at least one isomorphic form of the graph is a plane graph (can be drawn on a plane surface)
2. no crossover of edges

eg:

- Complete Graphs $k_n$
  $n \le 4$
- $Q_3$
- Bipartite graph $k_{m,n}$
  either $m \le 2$ **or** $n \le 2$

Non-planar graphs eg

- $k_5$ and larger
- $k_{3,3}$
    - find longest cycle
    - draw it as a circle

### Uses

coloring, classification, analysis of graphs

Plane form helps identify different phases (connected regions) of a planar graph

### Degree of Region

$|R|$ = No of edges in the boundary of that region

cut edge is counted twice

### Dual of Planar Graph

- every region will become vertices of the dual, and vice versa
  if G is primal graph and G* is the dual,
    - $|R^*| = |V|$
    - $|V^*| = |R|$
- if 2 regions have common boundary line, then the corresponding new vertices of the dual graph will get connected to each other

### Theorems

If G is a planar graph, then

1. $\sum deg(r_i) = 2|E|$
2. $3|R| \le 2|E|$
3. if G is a connected planar graph, then $|V| - |E| + |R| = 2$
4. $|E| \le 3|V| - 6$
5. There exists a vertex v in G such that $deg(v) \le 5$

## Euler’s theorem for planar graphs

$|V| - |E| + |R|= 2$

## Polyhedral Graphs

connected planar graphs

- $deg(v_i) \ge 3$
- $deg(r_i) \ge 3$
- using degree of region theorem,
    - $3|V| \le 2|E|$
    - $3|R| \le 2|E|$

eg: $k_4, Q_3$

## Eulerian Graph

Graph with at least one Eulerian circuit

|                                       | Eulerian Circuit | Eulerian Path |
| ------------------------------------- | ---------------- | ------------- |
| Path Type                             | closed           | open          |
| passes every edge of original graph   | exactly once     | exactly once  |
| passes every vertex of original graph | at least once    | at least once |
| repeated vertices                     | allowed          | allowed       |

### Cases

- All vertices are of even degree - both possible
- Only 2 vertices are of odd degree and the rest are even degree - eulerian path possible not circuit
- All vertices are of odd degree - both not possible

## Hamiltonian Graph

Graph with at least one Hamiltonian cycle

|                                       | Hamiltonian Cycle | Hamiltonian Path |
| ------------------------------------- | ----------------- | ---------------- |
| Path Type                             | simple, closed    | simple, closed   |
| passes every edge of original graph   | exactly once      | exactly once     |
| passes every vertex of original graph | exactly once      | exactly once     |
| repeated vertices                     | not allowed       | not allowed      |
| when we have cut edge, possible?      | not possible      | possible         |

### Dirac’s Theorem

A simple graph with n vertices $(n \ge 3)$ and $deg(v_i) \ge \frac n 2$ has a Hamiltonian circuit
eg: $k_n, n \ge 3$

this is ==not== a necessacity for existence of hamiltonian circuit; the converse is ==not== necessarily true
eg: cycle of $n$ vertices each of deg 2; there obviously is hamiltonian circuit even though dirac’s theorem isn’t satisfied

## Dual graph

dual graph is a graph where the

- vertices are the regions of primal graph
- regions are the vertices of primal graph

### Properties

If $G(V,E,R) \implies G^*(V^*,E^*,R^*)$, where G is primal and G* is dual graph

1. $|V^*| = |R|$
2. $|R^*| = |V|$
3. $|E^*| = |E|$
4. $deg(r_i) = deg(r^*_i)$
5. Dual graph is always planar
6. there is a cut vertex placed in region $r \implies$ you will get a self loop at $v^*$ of G, where $v^*$ represents the corresponding vertex of $r$

## Graph Coloring

A coloring of a simple graph is the assignment of a color to each vertex of the graph such that no 2 adjacent vertices are assigned the same color.

### Chromatic number of G

$\chi (G) =$ the least number of colors need for coloring G

eg:

- Star graph requires only 2 colors
- $k_n$ requires $n$ colors
- $k_{m,n}$ requires only 2 colors
- $C_n$ (cycle of $n$ vertices) requires
    - 2 colors when n = even
    - 3 colors when n = odd

### Theorem

For a ==planar== graph,

The chromatic number is no greater than 4, ie
$\chi(G) \le 4$

no proof for this

### Coloring Rules

1. $\chi \le |V|$
2. a triangle/triangular subgraph $(C_3)$ requires 3 colors
3. if some subgraph of $G$ requires $k$ colors, then
   $X(G) \ge k$
4. if deg$(v) = d$, then d colors are required to color the vertices adjacent to v
5. $\chi(G) = max \{ \chi(C)$ where C is a connected component of G
6. every $k$ chromatic graph $(\chi(G) = k)$ has atleast $k$ vertices such that the $deg(v_i) \ge k-1$
7. For any graph $G, \chi(G) \le 1 + \Delta(G)$
   $\Delta(G)$ is the largest degree of any vertex in G
8. $\chi(G) \ge \frac{|V|}{ |V| - \delta(G) }$
   $\delta(G)$ is the largest degree of any vertex in G

## Properties of chromatic number

1. $k$-critical graph is a graph where

     - $\chi(G) = k$
     - $\chi(G-V) = k-1$

   possible only if $\delta(G) \ge k-1$

2. G is 1-chromatic, then G is totally disconnected

3. $\chi(G) = 2 \iff$ G is bipartite graph $\iff$ every cycle of G has even length

   1. otherwise it will be a triangular subgraph and hence $\chi$ has to be 3

4. $\chi(G) \le \Delta(G) + 1$

   1. For complete graphs, $\chi(G) = \Delta + 1$
   1. For other graphs, $\chi(G) < \Delta + 1$

5. If G1, G2, … Gk are disconnected components of graph G, then $\chi(G) = max\set{\chi(G_i)}$

6. Every tree with $|V| \le 2$ is 2-chromatic

7. - $\chi(G) \ge 3 \iff$ G has a cycle of odd length
     - $\chi(G) = 2 \iff$ G has **no** cycle of odd length
     (we already learnt this for bipartite graphs)

8. Every connected k-connected graph contains a critical k-chromatic graph

9. Only type of 3-critical graph is $C_{2n+1}$