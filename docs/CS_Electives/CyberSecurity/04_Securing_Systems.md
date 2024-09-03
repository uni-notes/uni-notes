# Securing Systems

## Network Layer

| Layer       | Protocol                               | Attacks                                                                                                                                          | Solution                                                                                                   |
| ----------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Application | HTTP                                   | Machine-in-the-Middle attacks:<br>1. Script injection<br>2. Session hijacking using cookies<br>3. SSL/TLS Stripping: Redirecting to fake website | 1. HSTS: Hint from server to browser to always use TLS<br>2. VPN<br>3. URL redirecting of suspicious links |
|             | HTTPS<br>(~~SSL~~TLS)                  |                                                                                                                                                  |                                                                                                            |
|             | SSH                                    |                                                                                                                                                  |                                                                                                            |
| Transport   |                                        | Port Scanning                                                                                                                                    | Penetration testing                                                                                        |
| Network     | Unsecured                              | Packet sniffing: Adversary can be on the same network, or using an antenna                                                                       | Firewall<br>Deep Packet Inspection<br>Proxy                                                                |
|             | Secured<br>WPA: Wi-Fi Protected Access |                                                                                                                                                  |                                                                                                            |

## Attacks

- Malware
- Virus
- Worm: virus that can transfer through port-scanning
- Botnet
- DDOS: Distributed Denial of Service
	- Attempt to disrupt normal services by flooding machines/networks with superfluous requests, typically at high volume and frequency
	- Can be directed at occupying bandwidth, drive space, or CPU
	- Source: one or distributed (botnet)
	- Features of Log
		- Duration
		- No of packets sent
		- No of received packets
		- Interval of sending packets
		- Interval of receiving packets
		- Upload speed by source
		- Download speed by destination
		- Label: Normal (0)/DDOS (1)
	- Challenges in Data
		- Labelling
			- They don't label themselves
			- Not definite if DDOS is happening
			- Labelling is expensive and time-consuming
		- Benign network traffic is variant across different infrastructures
			- University: students streaming videos
			- Enterprise: High SMTP load
- Zero-day attacks

## Solutions
- Antivirus (with auto-updates)
	- Cannot handle zero-day attacks