## Digraph

Directed Graph

A digraph G is defined as (V, E), if $E \sube V \times V$, where V = vertices set of G, E = edge set of G

"edge incident from 1 to 3"

Every element of set is a vertex of digraph; every relation is an edge of digraph

### Indegree

no of edges incident TO vertex

### Outdegree

no of edges incident FROM vertex

## Subgraph

$G' = (V', E')$ is a subgraph of $G = (V, E)$ if $V' \subseteq V$ and $E' \subseteq E \cap(V' \times V')$ // not sure

## Graph Isomorphism

$G_1 = (V_1, E_1)$ and$G_2 = (V_2, E_2)$ are isomorphic if there is one-one and onto function f between them that preserves adjacency

$f: V_1 \to V_2$, where $f$ preserves adjacency
$E_2 = \{ (f(v), f(w)) | (v, w) \in E_1 \}$

// not sure // You have to check if they have the same properties (like reflexivity, symmetry, etc...); otherwise the 2 graphs won't maintain adjacency

### Adjacency

x and y are adjacent vertices if they are connected to each other

### Preservation of adjacency

if x adjacent to y, then f(x) and f(y) should also be adjacent to each other

## Features of isomorphism digraphs

If G1 and G2 are isomorphic

1. no of vertices equal in G1 and G2
2. no of edges equal in G1 and G2
3. degree spectrum of G1 and G2 are same

However, converse isn't necessarily true - the above 3 features don't necessarily imply isomorphism

## Degree Spectrum

... of a graph is set of indegree and outdegree for all vertices of a digraph

For every V:(i,j) where i = indegree and j = outdegree of vertex V.

Then degree spectrum of the graph $= \{ (i,j) | i=\text{indegree}, j = \text{outdegree} \}$