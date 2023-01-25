## String

sequence of data bytes/words that are in **consecutive** memory locations

**Everywhere**

- does **not** affect flags
- d = 0 -> SI/DI inc
- d = 1 -> SI/DI dec

## `MOVS`

Moving Strings

copies a byte/word/double word fro a location in the **data segment** to a location in the **extra segment**

- Source - DS:SI
- Destination - ES:DI

|         | SI/DI inc/dec by |
| ------- | ---------------- |
| `MOVSB` | 1                |
| `MOVSW` | 2                |
| `MOVSD` | 4                |

## `LEA`

Load effective address

```assembly
lea si, array1
lea di, array2
```

## LODS

Loads AL/AX/EAX witht the data stored at the data segment

offset address indexed by si register

|         | Equivalent |      |
| ------- | ---------- | ---- |
| `LODSB` | AL =       |      |
| `LODSW` | AX =       |      |
| `LODSD` | EAX =      |      |

## STOS

Stores AL/AX/EAX into the extra segment memory at offset address indexed by DI register.

|         | Equivalent |      |
| ------- | ---------- | ---- |
| `STOSB` |            |      |
| `STOSW` |            |      |
| `STOSD` |            |      |
