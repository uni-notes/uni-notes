## Rotate

ROL/ROR

Data does not get lost

also, the value gets stored in carry flag

![Screen Shot 2022-03-22 at 6.03.59 PM](assets/rotate.png)

## Shift

command is

- SAL/SHL
- SAR/SHR

Shift each bit count times

```assembly
sal dest, count
shl dest, count
```

## Multiplication

- Mul - unsigned
- Imul - signed

```assembly
mul src

mul cx ; ax
mul cl ; al
```

Src times

- AL
- AX
- EAX

Source can be a register or memory location

| Multiplication | Result Storage |
| -------------- | -------------- |
| Byte           | AX             |
| Word           | DX:AX          |
| Dword          | EDX:EAX        |

- CF and OF are zero if MSB/MSW/MSD zero
- AF, PF, SF, ZF - undefined
- CBW/CWD

## Conversion

- `CBW` converts byte to word
- `CWD` converts word to double word

When MSB is

- 0, 0s are added
- 1, 1s are added

## Division

### `div`

|           | 8bit | 16bit |
| --------- | ---- | ----- |
| dividend  | AX   | AX    |
| divisor   | BX   | BX    |
| quotient  | AL   | AX    |
| remainder | AH   | DX    |

### `idiv`

