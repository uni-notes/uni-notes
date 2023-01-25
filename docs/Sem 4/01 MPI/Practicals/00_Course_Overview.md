## Intro

We are using ‘Assembly Language’, which is a lower level language compared to C, C++, Java, Python, etc…

It uses an assembler to convert the code into machine language the processor can understand. (high level languages use compiler/interpreter).

## Installation

### Windows

### MacOS

1. Install `dosbox`
   1. https://www.dosbox.com/download.php?main=1
2. Copy `8086` files to `ahmedthahir/dosbox`; basically the root folder (next to Desktop, Documents, etc)
   1. https://www.mediafire.com/file/mm7cjztce9efj4w/8086.zip/file
3. open `dosbox`
4. `mount c ~/dosbox/8086`
5. `c:`

## Basics

### Skeleton Program

```assembly
.model small
.stack 20

.data
org 1000h
num1 db 05h
num2 db 03h

.code
start:
mov ax, @data
mov ds, ax

mov al, num1
add num2, al

int 3
end start
code ends
```

### Steps

1. Open up TurboAssembler

2. **Editing**
   
     - no
     - Dos
     1. Type `edit fileName.asm`
     1. Type your code
     1. Save your code
        [Click Here to learn how](#Saving)
   
3. **Assembling**
   Type `tasm fileName.asm`

4. **Linking**
   Type `tlink fileName.obj`

5. **Execution**
   1. Type `td fileName.exe`
   2. Click `F7` to execute the required lines

6. **Viewing results**
   1. Click `Tab` key until focus reaches the address-value thing at the bottom
   2. Click `Ctrl-G`
   3. Enter `ds:address`
      for eg `ds:1000`

### Saving

on your keyboard, click

1. `Alt+f` 
2. then `s` 