##  Types of Instructions

1. Data Transfer
2. Arithmetic
3. Logical
4. Branch and Program Control

## Steps

1. initialise segment register

   ```assembly
   mov ax, 2000h
   mov ds, ax
   ```

## `MOV`

$$
\textcolor{orange}{
\underbrace{\text{MOV}}_\text{Opcode}
}
\ \
\textcolor{green}{
\underbrace{\text{dest, src}}_\text{Operands}
}
$$

Destination/Source could be register/memory location. This is the data, and the operands of the operation

4 bits are required to refer to a register: $0000-FFFF$

MOV, ADD, etc… are called mnemonics

$$
\text{MOV dest, src}
$$

- copies contents from src to dest
- no flags affected
- size of src and dest must be same; however smaller data can be inserted into bigger register

### Possible Options

- source can be register, memory location, immediate date
- dest can be register/memory location

|   From   |    To    |
| :------: | :------: |
| Register |  Memory  |
|  Memory  | Register |
| Register | Register |
|  Index   |  Memory  |
|  Index   | Register |

You **cannot** do `MOV [1234] [5678]`

## `INC`, `DEC`

$$
\text{INC dest} \\\text{DEC dest}
$$

increments/decrements the content of the affected register by 1.

```assembly
inc ax

inc word ptr[bx]
inc byte ptr[bx] ; only low byte
```

## `ADD`

## `ADC`

First you must use `CLC` to clear the carry flag.

## `EQU`

used to assign value to a variable. It doesn’t store anything in memory.

```assembly
count equ 08h
mov cl, count
```

## `DUP`

Duplicate

```assembly
array db 5 dup(12h)
array db 5 dup('A')
```

## Flags

| Flag | Meaning   | High when                                           |
| ---- | --------- | --------------------------------------------------- |
| AF   | Auxiliary | internal carry (from lower nibble to higher nibble) |
| CF   | Carry     | carry from the entire byte                          |
| OF   | Overflow  | overflow                                            |
| PF   | Parity    | even parity (only follows low byte)                 |
| SF   | Sign      | signed number                                       |
| ZF   | Zero      | data is 0                                           |

## Branch Instructions

Jump means like `go to` in C++

|      |                   |
| ---- | ----------------- |
| JZ   | Jump on Zero      |
| JNZ  | Jump on Non-Zero  |
| JE   | Jump on Equal     |
| JNE  | Jump on Not Equal |

