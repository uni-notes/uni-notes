# Securing Data

According NIST requirements, secrets must be salted and hashed

|                 | Meaning                                            | Reversible | Limitation                                                                                                                                                                                                                                                                                                                     |
| --------------- | -------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| One-way Hashing | Converting credential into fixed-length hash value | ❌          | - multiple credentials can have the same hash value; which is also good because adversary cannot easily get the credential even with hashes<br>- multiple people can have the same credential and hence, the same hash<br>	- adversary will use this information to find patterns between the people and crack the credentials |
| Salting         | Private key perturbs the hash value                | N/A        |                                                                                                                                                                                                                                                                                                                                |
| Encryption      |                                                    | ✅          |                                                                                                                                                                                                                                                                                                                                |

The hashing algorithm need not be private; reputed hashing functions are nearly impossible to reverse-engineer

## Problems

|               |                                                                                                                            |
| ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Rainbow Table | Adversaries may have already hashed all the words in the dictionary<br><br>But very unlikely for reputed hashing functions |

If an online service directly emails your password
- that means they know your password, and hence it is stored unencrypted
- do not use that service anymore

## Encryption

| Type                      | Encryption  | Decryption  | Application                                       | Algorithms                   |
| ------------------------- | ----------- | ----------- | ------------------------------------------------- | ---------------------------- |
| Secret-Key/<br>Symmetric  | Private key | Private key |                                                   | AES<br>Triple DES            |
| Assymetric<br>Public-Key  | Public key  | Private key | Messaging data transfer and storage               | Diffie-Hellman<br>MQV<br>RSA |
| Assymetric<br>Private-Key | Private key | Public key  | Digital signatures<br>Message source verification |                              |
| End-to-End                |             |             |                                                   |                              |
## Deleting Data

Deleting from storage device does not actually physically delete the files; it just frees up the pointer

- Secure deletion: just turn all the bits to 0s/1s/random
- Full-disk encryption/Encryption at rest
	- Data only decrypted when the device is on
	- Important to enable this at the start of using storage device
		- Over time, device may wear out and encryption may not happen

## Ransomeware

Adversary encrypts the victim's storage for payment

## Quantum Computing

If adversaries get more computing power, all systems will become insecure

