## Network Criteria

- Fault Tolerance
- Scalability
- QoS (Quality of Service)
  - High Throughput
  - High Bandwidth
  - Low Latency
- Security

## Performance Criteria

|                       |                                                              |
| --------------------- | ------------------------------------------------------------ |
| Bandwidth             | Max number of bits transferrable per unit time<br />(In analog world, it is the range of accepted frequencies) |
| Throughput            | Actual number of bits transferred per unit time              |
| Latency/<br />Delay   | Duration to send info & its earliest possible reception      |
| End-to-End<br />Delay | Duration to transmit packet along its entire path<br />- Created by application<br />- Handed over to OS<br />- Passed to NIC<br />- Encoded, transmitted over a physical medium<br />- Received by intermediate device (switch, router)<br />- Analyzed, retransmitted over another medium, etc. |
| Round-Trip-Time       | Duration to send and receive acknowledge                     |

## Types of Delays

| Delay                | Duration of                                                  | Formula                                                      |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Transmission         | Placing bits onto transmission mediaum                       | $\frac{\text{Size}}{\text{Bandwidth}}$                       |
| Propagation          | Travel for a bit from one end of medium to other             | $\frac{\text{Distance}}{\text{Speed}}$                       |
| Processing           | Error verification<br />Routing decision, ie<br />- analyze packet header<br />- decide where to send packet | No of entries inrouting table<br />Implementatio of data structures<br />Hardware specs |
| Buffer/<br />Queuing | Packet to wait until it is transmitted                       | Traffic intensity<br />Type of traffic                       |

Latency = $\sum$ all the above delays

## Mediums

| Medium |       Speed (m/s) |
| ------ | ----------------: |
| Vacuum |   $3 \times 10^8$ |
| Cable  | $2.3 \times 10^8$ |
| Fiber  |   $2 \times 10^8$ |

