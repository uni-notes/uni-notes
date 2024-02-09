| Command     | Usage                                    | Displays                                                     | Full Form                   | MacOS<br />alternative |
| ----------- | ---------------------------------------- | ------------------------------------------------------------ | --------------------------- | ---------------------- |
| `getmac`    | `getmac`                                 | Get mac address of current machine                           |                             | `ipconfig`             |
| `ipconfig`  | `ipconfig`                               | computer’s IP details                                        |                             |                        |
| `ping`      | `ping www.google.ae`                     | Check validity of connection between 2 machines<br />Sends 4 packets to the other machine and checks how many of those packets reached<br />Rount-trip time<br />Time to live |                             |                        |
| `netstat`   | `netstat`<br />`netstat -r`              | Connection information<br />Routing table                    | Network Statistics          |                        |
| `route`     | `route`                                  | Routing table                                                |                             | `netstat -nr`          |
| `arp`       | `arp -a`                                 | ARP cache of current machine                                 | Address Resolution Protocol |                        |
| `hostname`  | `hostname`                               | Gives the machine name                                       |                             |                        |
| `nslookup`  | `nslookup`<br />`nslookup www.google.ae` | Look and diagnose the DNS of a location                      | Name System                 |                        |
| `tracert`   | `tracert www.google.ae`                  | Shows the RTT from source and destination node, and also all the intermediary nodes | Trace Root                  |                        |
| `pathping`  | `pathping www.google.ae`                 | Combination of `ping` and `tracert`                          |                             |                        |

