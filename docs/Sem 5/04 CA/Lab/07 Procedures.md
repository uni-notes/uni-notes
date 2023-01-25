## Question 1

Procedures to find __ of 2 inputted numbers

- Sum
- Difference
- Product
- Quotient

**Arguments**

- $x$
- $y$

```assembly
.data
newline: .asciiz "\n"

.text
main:
	addi $a0, $zero, 1
	addi $a1, $zero, 5

	jal sum
	add $a0, $zero, $v0
	addi $v0, $zero, 1
  syscall	
  la $a0, newline
	addi $v0, $zero, 4
	syscall

	addi $a0, $zero, 1
	addi $a1, $zero, 5

  jal dif
	add $a0, $zero, $v0
	addi $v0, $zero, 1
  syscall
	la $a0, newline
	addi $v0, $zero, 4
	syscall

	addi $a0, $zero, 1
	addi $a1, $zero, 5

  jal pro
	add $a0, $zero, $v0
	addi $v0, $zero, 1
  syscall
	la $a0, newline
	addi $v0, $zero, 4
	syscall
	addi $a0, $zero, 1
	addi $a1, $zero, 5

  jal quo
	add $a0, $zero, $v0
	addi $v0, $zero, 1
  syscall
	la $a0, newline
	addi $v0, $zero, 4
	syscall

	#exit
	addi $v0, $zero, 10
	syscall
.end main

sum:
	add $v0, $a0, $a1

	jr $ra

dif:
	sub $v0, $a0, $a1

	jr $ra

pro:
	mult $a0, $a1
	mflo $v0

	jr $ra

quo:
	div $a0, $a1
	mflo $v0

	jr $ra
```

## Question 2

Procedure for linear search

**Arguments**

- Array address
- Array length
- Search Element

**Return**

- Found flag
- Index, if found

```assembly
.data
array: .word 10, 20, 30, 40, 50
found_msg: .asciiz "Found at index "
not_found_msg: .asciiz "Not Found"

.text
main:
	la $a0, array
	addi $a1, $zero, 5
	addi $a2, $zero, 30

	jal linear_search
	beq $v0, $zero, print_not_found

	print_found:
		la $a0, found_msg
		addi $v0, $zero, 4
		syscall

		add $a0, $zero, $v1
		addi $v0, $zero, 1
		syscall

		j exit

	print_not_found:
		la $a0, not_found_msg
		addi $v0, $zero, 4
		syscall
  
	exit:
		addi $v0, $zero, 10
		syscall

linear_search:
	add $t0, $zero, $zero ## i = 0	
	add $t4, $zero, 4
	add $v0, $zero, $zero ## status

	loop:
		beq $t0, $a1, return
		
		mult $t0, $t4 ## i*4
		mflo $t1 ## offset = i*4
		add $t1, $t1, $a0 ## adress = array_base_address + offset
		
		lw $t2, 0($t1)
		beq $t2, $a2, found
		
		addi $t0, $t0, 1
		j loop

	j return

	found:
		addi $v0, $zero, 1
		add $v1, $zero, $t0
		j return

	return:
		jr $ra
```

