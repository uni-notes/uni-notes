## Gates

|  a   |  b   |  a'  | $a \cdot b$ | a + b | a nand b | a nor b | a xor b | a xnor b |
| :--: | :--: | :--: | :---------: | :---: | :------: | :-----: | :-----: | :------: |
|  0   |  0   |  1   |      0      |   0   |    1     |    1    |    0    |    1     |
|  0   |  1   |      |      0      |   1   |    1     |    0    |    1    |    0     |
|  1   |  0   |  0   |      0      |   1   |    1     |    0    |    1    |    0     |
|  1   |  1   |      |      1      |   1   |    0     |    0    |    0    |    1     |

## Verilog Codes

``` verilog
module gates(y, a, b);
  input a, b;
  output y;
  wire andg, org, notg, nandg, norg, xorg, xnorg;
  
  assign andg = a & b;
  assign org = a | b;
  assign notg = ~a;
  
  assign nandg = ~ (a&b);
  assign norg = ~ (a|b);
  
  assign xorg = a^b;
  assign xnorg = ~(a^b);
  
endmodule
```

## Bubbled Gates

inputs to the gates are negated

|               | Bubbled AND       | Bubbled OR    |
| ------------- | ----------------- | ------------- |
| Function      | $y = a' \cdot b'$ | $y = a' + b'$ |
| Another name  | Negative AND      | Negative OR   |
| equivalent to | NOR               | NAND          |

### Verilog

``` verilog
module gates(y, a, b);
  input a, b;
  output y;
  wire band, bor;
  
  assign band = ~a & ~b;
  assign bor = ~a | ~b;
  
endmodule
```

## Universal Gates

NAND & NOR are called universal gates because we can implement all other gates with just these 2

### for NOT

for NAND and OR, just split a single input into 2 wires and pass it through the gate

``` verilog
module gates(y, z, a, b);
  input a, b;
  output y, z;
  
  assign y = nand(a, b);
  assign z = nor(a, b);
  
endmodule
```

### for OR

NOR

1. use a gate
2. complement the output

NAND

1. complement a, b individually 
2. Complement the output using another gate

``` verilog
module gates(y, a, b);
  input a, b;
  output y;
  wire p, q;

  assign p = nor(a, b); // (a+b)'
  assign y = nor(p, p); // a+b
  
  assign p = nand(a, a); // a'
  assign q = nand(b, b); // b'
  assign y = nand(p, q); // (a' . b')' = a + b
  
endmodule
```

### for AND

NAND

1. use a gate
2. complement the output using another gate

NOR

1. complement a, b invidually
2. Complement the output using another gate

``` verilog
module gates(y, a, b);
  input a, b;
  output y;
  wire p, q;

  assign p = nand(a, b); // (a*b)'
  assign y = nand(p, p); // a*b
  
  assign p = nor(a, a); // a'
  assign q = nor(b, b); // b'
  assign y = nor(p, q); // (a' + b')' = a * b
  
endmodule
```