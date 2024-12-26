# Securing Software

- OWASP: Open Worldwide Application Security Project
	- has documentation all kinds of attacks
- CVE: Common Vulnerabilities and Exposures
- CVSS: Common Vulnerability Scoring System
- EPSS: Exploit Prediction Scoring System
- KEV: Known Exploited Vulnerabilities Catalog

## Attacks

|        | Type of Attack                                                 | Attack                    |                                                                                                  | What could go wrong                                                                                                                                       | Solution                                                                                                                                         |
| ------ | -------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Web    | Phishing                                                       |                           | `<a href="https://g00gle.com">https://google.com</a>`                                            | Fake website                                                                                                                                              | 1. Always hover and verify destination before clicking links<br>2. Check if website has `https` (minimum requirement; not sufficient for safety) |
|        | JS Injection                                                   | XSS: Cross-Site Scripting | Trick website to execute malicious code via js/css                                               |                                                                                                                                                           | Character escape<br>HTTP Header                                                                                                                  |
|        |                                                                | Reflected attack          | Phishing + XSS<br><br>`<a href="https://g00gle.com/search?q=script_here">https://google.com</a>` | Cookie Hijacking                                                                                                                                          | Character escape<br>HTTP Header                                                                                                                  |
|        |                                                                | Stored Attack             | Email                                                                                            |                                                                                                                                                           | Character escape<br>HTTP Header                                                                                                                  |
|        | SQL Injection                                                  |                           |                                                                                                  | Extract details/credentials of other users                                                                                                                | Prepared statements                                                                                                                              |
|        | Command injection                                              |                           | Command-line<br>`os.system()`, `eval()`                                                          | Access file system                                                                                                                                        |                                                                                                                                                  |
|        | Developer Tools Tweaks                                         |                           |                                                                                                  | Bypass client-side validation                                                                                                                             | Always do server-side validation                                                                                                                 |
|        | CSRF<br>Cross-Site Request Forgery                             | GET                       | Using GET request for changing state<br><br>Eg: Buying amazon item                               | Request can be made via other websites using `<img src>`                                                                                                  | GET Request should never be used for changing state                                                                                              |
|        |                                                                | POST                      | Even POST form has issues                                                                        | Javascript can be used to automatically submit form across websites, when user visits a website with an embedded form<br><br>`document.forms[0].submit()` | Use POST with a random CSRF token for a particular user                                                                                          |
| Native | Code Execution<br><br>R/ACE<br>Remote/Arbitrary Code Execution | Buffer Overflow           |                                                                                                  |                                                                                                                                                           | Digital Signatures<br>Package managers                                                                                                           |
|        | Cracking                                                       |                           |                                                                                                  |                                                                                                                                                           |                                                                                                                                                  |
|        | Reverse Engineering                                            |                           |                                                                                                  |                                                                                                                                                           |                                                                                                                                                  |

## Solutions

Always preprocess inputs from user; do not trust any input

### Character Escape

At least escape these

| replace | with     |
| ------- | -------- |
| `<`     | `&lt;`   |
| `>`     | `&gt;`   |
| `&`     | `&amp;`  |
| `"`     | `&quot;` |
| `'`     | `&apos;` |

### HTTP Header

Instruct browsers to only allow execution of styles/scripts
- in files (not inline css/js)
- from the specified domain

```
Content-Security-Policy: script-src https://example.com
Content-Security-Policy: style-src https://example.com
```

### Prepared Statements

Let programming language/DB engine handle escaping

```python
query = f"""
select *
from users
where username = ? AND password = ?;
"""
```

### Bug Bounties

