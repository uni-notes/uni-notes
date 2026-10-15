## Types of instructuctions

- Data transfer
- Arithmetic
- Logical
- Branch(conditional) and program control

## Little Endian Format

In little endian format adopted by Intel and most manufacturers, first the low byte gets stored and then the high byte.

Consider a number $1234_H$. It will be stored in memory as follows

$$
\underset{40000}
{\Large
\fbox{34$\vphantom{0}$}
}
\underset{40001}
{\Large
\fbox{12$\vphantom{0}$}
}
$$

## Addressing Modes

| Addressing                 |                    |                             |
| -------------------------- | ------------------ | --------------------------- |
| Register                   | MOV AX, BX         | AX, BX are registers        |
| Immediate                  | MOV AX, 1420~H~    | 1420H = value of data       |
| Direct                     | MOV AX, [2340~H~]  | 2340 = offset address of DS |
| Register Indirect          | MOV AX, [BX]       | BX is the pointer           |
| Base-Plus-Index            | MOV AX, [BX+SI]    | BX, SI are pointers         |
| Register relative          | MOV AX, BX[10]     | BX is the pointer           |
| Base relative-plus-indexed | MOV AX, [BX+SI+10] |                             |
| Scaled Indexed             | MOV AX, [10BX]     |                             |

## Instruction Format

The assembler converts assembly code into bytecode

- Mnemonics like `MOV`, `ADD` get converted into opcode
- Variable names get converted into addresses

### Register Addressing

$$
\underbrace{ \fbox{1} \fbox{0} \fbox{0} \fbox{0} \fbox{1}\fbox{0} }
_{\text{Opcode}}
\underset{\text{D}}{ \fbox{1}}
\underset{\text{W}}{ \fbox{1}}
\underbrace{ \fbox{1} \fbox{1} }
_{\text{MOD}}
\underbrace{ \fbox{0} \fbox{1} \fbox{1} }
_{\text{Reg}}
\underbrace{ \fbox{0} \fbox{1} \fbox{1} }
_{\text{R/M}}
$$

#### Meanings

|        |                                 |    0     |   1    |
| ------ | ------------------------------- | :------: | :----: |
| Opcode | Operation Code                  |          |        |
| D      | **D**irection                   | From Reg | To Reg |
| W      | **W**ord                        |   Byte   |  Word  |
| MOD    | Addressing **Mod**e of R/M      |          |        |
| Reg    | **Reg**ister                    |          |        |
| R/M    | **R**egister/**M**emory Address |          |        |

#### Word

| W=0  | W=1  |
| :--: | :--: |
|  AL  |  AX  |
|  CL  |  CX  |
|  DL  |  DX  |
|  BL  |  BX  |
|  AH  |  SP  |
|  CH  |  BP  |
|  DH  |  SI  |
|  DH  |  DI  |

#### MOD

| MOD  |   Addressing Mode   |
| :--: | :-----------------: |
|  00  |                     |
|  01  |                     |
|  10  |                     |
|  11  | Register Addressing |

#### Reg

| Register    | Code |
| ----------- | ---- |
| EAX, AX, AL | 000  |
| EBX, BX, BL | 011  |
| EAX, CX, CL |      |

No need to learn 32bit encoding

