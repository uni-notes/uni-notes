# Securing Accounts

| Attack                | Description                                                                | Problem                                                                             | Solution                                                                                                    |
| --------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Dictionary            | Hacker attempts different combination of the dictionary to find credential |                                                                                     | Do not use words from dictionary                                                                            |
| Brute-Force           | Hacker attempts all possible combination of credentials                    | Even if the credential is random, if it is too short, hacker will be able to get it | NIST recommendations for credentials<br>[MFA](#MFA)<br>[SSO](#SSO)<br>[Password Manager](#Password-Manager) |
| Keylogging            | Malicious software recording everything user types                         | Adversary will get access to username, credential, OTP                              | - Avoid using public computers<br>- Avoid connecting to public networks                                     |
| Credential Stuffing   | Using list of known credentials                                            | Adversary will try these to get access                                              | Use different credentials for different services                                                            |
| Social Engineering    |                                                                            |                                                                                     |                                                                                                             |
| Phishing              |                                                                            |                                                                                     | Be careful of clicking links<br>Be careful of entering information online                                   |
| Machine-in-the-Middle | Compromise of any machines in between                                      |                                                                                     | Be careful what online services you use                                                                     |
| Deepfakes             | Audio<br>Video                                                             | Voice can be used by adversary to gain access                                       | Disable authentication using voice<br>Do not speak out any special key phrases into services                |

## Solutions

| Solution                           |                                                                |
| ---------------------------------- | -------------------------------------------------------------- |
| Good password                      | Check out NIST credential  requirements                        |
| MFA<br>Multi-Factor Authentication |                                                                |
| SSO<br>Single Sign-On              |                                                                |
| Password-Manager                   | Catch: protect the master password                             |
| Passkeys                           | Login to online services with biometrics via device's hardware |

### NIST credentials requirements
National Institute of Standards and Technology

- Should contain letters, numbers, punctuations, and unicode characters
- Should at least be 8 characters
- Should not contain
	- credentials from previously-breached corpuses
	- Dictionary words
	- Repetitive or sequential characters
	- Context-specific words
		- username
		- name of service
- Do not allow hint, that is accessible to unauthenticated entity
- Verifiers should not require memorized secrets to be changed arbitrarily/periodically
	- Keeping on changing will irritate people and they will end up making a less-secure credential
- Verifiers should enforce a rate-limiting mechanism

### MFA

Multi-Factor Authentication

| Factor     | Meaning            | Example                             |
| ---------- | ------------------ | ----------------------------------- |
| Knowledge  | Something you know | Childhood best friend<br>credential |
| Possession | Something you have | OTP (One-Time credential)           |
| Inherence  | Something you are  | Biometrics                          |
Sending OTP over SMS is not safe
- SMS is unencrypted
- SIM Swapping
