## Jumps

| Jump  | Displacement | Range           |
| ----- | ------------ | --------------- |
| Short | 8 bits       | $-128 \iff 127$ |
| Near  | 16 bits      |                 |

## Loops

```assembly
count db 09h
mov cx, count 			; initialization

repeat:
	; code
	loop repeat
	
; equivalent to
repeat:
	; code
	dec cx 						; updation
	jnz repeat 				; condition
```

