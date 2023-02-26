## Interrupts

8086 has 2 interrupts.

|                       | Maskable                                  | Non-Maskable |
| --------------------- | ----------------------------------------- | ------------ |
|                       | controlled by the interrupt flag          |              |
| checks interrupt flag | ✅                                         | ❌            |
| works when            | interrupt flag is high                    |              |
| input                 | INTR (Interrupt Request)                  | NMI          |
| output                | $\overline{INTA}$ (Interrupt Acknowledge) |              |

## Hold

Input Signal to the processor from the bus masters as a request to control the bus.

Usually by DMA controller.

## HLDA

Hold Acknowledge

Output Signal from the processor to the bus master requesting control.

When high, acknowledged