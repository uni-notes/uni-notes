## Question 1

Implement a function to find exponent of a number

```assembly
.data

.text
main:
addi $a0, $zero, 2
addi $a1, $zero, 0

jal power

display:
add $a0, $zero, $v0
addi $v0, $zero, 1
syscall

exit:
addi $v0, $zero, 10
syscall

power:
	beq $a1, 0, zero_case
  
  addi $sp, $sp, -8
	sw $ra, 4($sp)
	sw $a0, 0($sp)

	beq $a1, 1, one_case

	mult $v0, $a0
	mflo $v0
	jr $ra

	addi $a1, $a1, -1
	jal power

	zero_case:
		addi $v0, $zero, 1
		jr $ra

	one_case:
		lw $ra, 4($sp)
		lw $v0, 0($sp)
		addi $sp, $sp, 8

		add $v0, $zero, $a0
		jr $ra
```

## Question 2

