# Preserving Privacy


- Web browsing history
- Server-side logs
- Referer
	- Browser tells webpage from which webpage you came
	- `<meta name="referrer" content="none" />`
	- `Referrer-Policy: no-referrer`
- Fingerprinting
	- Profiling user using request characteristics
	- User-Agent
		- Browser
		- Operating System
	- Display resolution
	- Timezone
- IDK
	- Cookies
		- Session Cookies: usually harmless, but not safe from physical snooping
		- Tracking Cookies: disable
		- Third-Party Cookies: disable
		- Supercookies: no solution
	- Tracking Query Parameters: remove them
	- LocalStorage is preferred
- DNS: Domain Name System
	- ISP (Internet Service Provider) can see what domain you are looking for
	- DoH: DNS over HTTPS
	- DoT: DNS over TLS
## Solutions

- Incognito: Private Browsing
- Strict privacy permissions
- VPN
- Tor

## Data Protection
GDPR: General Data Protection Regulation



| Aspect             | Meaning                                                                                            | Solution                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Identifiability    | Reduce and safeguard identifiable data components as much as possible                              | De-Identification Techniques<br>- Anonymization<br>- Pseudonymization                 |
|                    |                                                                                                    | Safeguarding practices<br>- Data Encryption<br>- Secure servers<br>- Storage Location |
| Data Minimization  | Limit data collection and duration of storage to only what is required to fulfill specific purpose | Right to be Forgotten                                                                 |
| Notice and Consent | Prepare clear notice and consent communication to data subjects                                    | Consent<br>- Voluntary<br>- Informed<br>- Competent                                   |
## Class of Data

|              |                                                                                                       |
| ------------ | ----------------------------------------------------------------------------------------------------- |
| Sensitive    | Genetic<br>Biometric<br>Health<br>Race<br>Ethnicity<br>Political affiliation<br>Religious affiliation |
| Personal     | Name<br>Home address<br>Location<br>Email<br>IP address                                               |
| Non-Personal | Generalized data<br>Aggregated data<br>Data collected by govt bodies                                  |
