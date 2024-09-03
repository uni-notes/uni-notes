# Cyber-Security

Security: Achieving a goal against an adversary

## Cyber-Security
- Protection of important information and related systems
- All fields require cyber-security, especially financial and govt institutions
- Highly influenced by military
- There is always tradeoff between usability and security

Note: Never hack into a system you’re not authorized to do so, not even a scan.


|                |                                                                              |
| -------------- | ---------------------------------------------------------------------------- |
| Entity         | Every entity has an identification                                           |
| Resource       |                                                                              |
| Authentication | Process of proving an entity's identification                                |
| Authorization  | Whether or not an entity can have access to a resource, after authentication |

## Privileges

Whenever you’re designing a system, everyone should have the least amount of privileges as possible.

## Aspects

### Policy

**CIA**

- Confidentiality
  - Protection of data
- Integrity
  - Maintaining the data as it is
- Availabiltiy (Capacity Planning)
  - Making information available when required

### Thread Model

Assumptions about adversary

Better to be over-cautious about adversary than casual

### Mechanism

Hardware, Software, Algorithms to enforce policy against the threat model

This process is iterative, as you are trying to achieve a negative goal, so you need to keep updating the system.

A good mechanism is one that enforces the policy using as few components as possible. The more endpoints you have, the more vulnerabilities

## Cyber Security Awareness

Not a science

It is an evolving list of best practices

Instilling knowledge to people about these practices

## Cyber Security Threats

These are very easy for hackers

- Web Application
- Software Vulnerability
- Credentials Theft
- ‘Strategic Web Compromise’
- DDOS (Distributed Denial of Service)
- Malware/Ransomware
- Phishing
- Social Engineering

## Threat Creators

|                     | Skill              | Goal                                        | Example        |
| ------------------- | ------------------ | ------------------------------------------- | -------------- |
| Script Kddies       | Low                |                                             |                |
| Hacktivism          | Varied             | Public show against unethical organizations | ‘Anonymous’    |
| Crime ORganizations | Medium $\iff$ High |                                             |                |
| Nation State Actors | Very High          |                                             |                |
| Insider Threat      | Low                |                                             | Edward Snowden |

## Fields of Cyber-Security

### Risk-Management

- Security Strategies
- Risk Assessments
  - List out the possible outcomes
- Security Architecture
- Compliance Reviews
- Risk & Governance
- Policy
- Awareness

### Offensive

- Vulerability Assessment
- Penetration Testing
- Bug Bounty
- Netowrk Instrusion
- Hacking

### Defensive

Mostly white hat hackers do this

- Data Protection
- Log Monitoring
- Incident Response
- Threat Intelligence & Detection
  - Using indicators, like fraud detection in data mining
- Cyber Forensics

## Types of Hackers

- White Hat = Ethical Hacking
- Black Hat =  Cracking/Unethical Hacking
- Grey Hat = Mix of both

## Parts of Cyber-Security

### Vulnerability

Weakness in the system

- Bugs
- Misconfiguration
- Poor process or control

### Threat

Circumstance/event with potential harm

- Cyber criminals
- Nation State Actors
- Internal Threats

### Cyber Controls

#### Types

Defence-in-depth is followed, using a collection of these types

- Administrative
  You cannot proceed to the other types without first addressing this
  - Compliance to regulators
  - Standarsds
  - Policies
  - Precedures
  - Legal contracts
  - Service level agreements
- Preventative
  - Firewalls
  - Web Proxy
    - Internet Content Inspection
  - Email Gateways
  - Anti-Virus
  - Patch Management
- Detective
  - Vulnerability Scanning
  - Log Monitoring
  - Security Reviews
  - Honey Bot
    - Enticement of hackers, to detect how they would attack your main system
    - Not lure/entrap (that is illegal)
- Corrective
  - Incident Response
  - Cyber Forensics
  - Business Continuity & Disaster Recovery
- Deterrent Control
  - Just to deter any possible hackers; Doesn’t provide security
  - Such as a ‘Beware of Dog’, without actually having a dog

#### Examples

- Firewall is like Locks
- Log Monitoring is like CCTV

## Risk

Materialized threat exploiting a vulnerability
$$
\text{Inherent Risk} +
\underset{\approx \text{ Brakes}}{\text{Cyber Controls}} =
\text{Residual Risk}
$$
The organization should decide the residual risk it can tolerate.

### 4 ways to handle risk

- Risk assessment
- Risk transfer
- Risk mitigation
- (one more thing)

