# Form Onsubmit
- Trigger
## Email

```python
const EMAIL_SUBJECT = 'Kayal Email Test';

function onFormSubmit(e) {
  const responses = e.namedValues;

  // const email_superuser = responses['Email Address'][0].trim();
  const superuser_name = responses['Superuser'][0].trim();
  person_type = responses["Type"][0].trim();

  let registrand_id, registrand_id_split, registrand_name, registrand_email;
  
  if (person_type == "Existing Member") {
    registrand_id = responses['Person'][0].trim();
    registrand_id_split = registrand_id.split(" - ");
    registrand_name = registrand_id_split[1];

    const database = SpreadsheetApp.getActiveSpreadsheet();

    const search_string = registrand_id;
    var textFinder = database.getRange("Database!A1:A5000").createTextFinder(search_string);
    var match_row = textFinder.findNext().getRow();

    registrand_email = database.getRange("Database!B"+match_row).getValues()[0][0];
  } else if (person_type == "New Member" || person_type == "Guest") {
    registrand_name = responses['Name'][0].trim();
    registrand_email = responses['Email'][0].trim();
  } else {
    return
  }

  MailApp.sendEmail({
    to: registrand_email,
    subject: EMAIL_SUBJECT,
    htmlBody: createEmailBody(registrand_name, superuser_name),
  });

  // Append the status on the spreadsheet to the responses' row.
  const sheet = SpreadsheetApp.getActiveSheet();
  var row = sheet.getActiveRange().getRow();
  var column = e.values.length + 1;
  sheet.getRange(row, column).setValue('Email Sent');
}

function createEmailBody(registrand_name, superuser_name) {
  // Make sure to update the emailTemplateDocId at the top.
  var emailBody = "" +
  "Dear " + registrand_name +
  "<br/><br/>" +
  "You have been registered by " + superuser_name + "." +
  "<br/><br/>" +
  "Regards" +
  "<br/>" +
  "Kayal Registration Team";
  
  return emailBody;
}
```

## Document
```javascript
/**
  Replace the "<DOCID>" with your document ID, or the entire URL per say. Should be something like:
  var EMAIL_TEMPLATE_DOC_URL = 'https://docs.google.com/document/d/asdasdakvJZasdasd3nR8kmbiphqlykM-zxcrasdasdad/edit?usp=sharing';
*/

var EMAIL_TEMPLATE_DOC_URL = 'https://docs.google.com/document/d/<DOCID>/edit?usp=sharing';
var EMAIL_SUBJECT = 'This is an important email';

/**
 * Sends a customized email for every response on a form.
 *
 * @param {Object} e - Form submit event
 */
function onFormSubmit(e) {
  var responses = e.namedValues;

  // If the question title is a label, it can be accessed as an object field.
  // If it has spaces or other characters, it can be accessed as a dictionary.
  
  /** 
    NOTE: One common issue people are facing is an error that says 'TypeError: Cannont read properties of undefined'
    That usually means that your heading cell in the Google Sheet is something else than exactly 'Email address'.
    The code is Case-Sesnsitive so this HAS TO BE exactly the same on line 25 and your Google Sheet.
  */
  var email = responses['Email Address'][0].trim();
  Logger.log('; responses=' + JSON.stringify(responses));

  MailApp.sendEmail({
    to: email,
    subject: EMAIL_SUBJECT,
    htmlBody: createEmailBody(),
  });
  Logger.log('email sent to: ' + email);

  // Append the status on the spreadsheet to the responses' row.
  var sheet = SpreadsheetApp.getActiveSheet();
  var row = sheet.getActiveRange().getRow();
  var column = e.values.length + 1;
  sheet.getRange(row, column).setValue('Email Sent');
}

/**
 * Creates email body and includes the links based on topic.
 *
 * @param {string} name - The recipient's name.
 * @param {string[]} topics - List of topics to include in the email body.
 * @return {string} - The email body as an HTML string.
 */
function createEmailBody() {
  // Make sure to update the emailTemplateDocId at the top.
  var docId = DocumentApp.openByUrl(EMAIL_TEMPLATE_DOC_URL).getId();
  var emailBody = docToHtml(docId);
  return emailBody;
}

/**
 * Downloads a Google Doc as an HTML string.
 *
 * @param {string} docId - The ID of a Google Doc to fetch content from.
 * @return {string} The Google Doc rendered as an HTML string.
 */
function docToHtml(docId) {
  // Downloads a Google Doc as an HTML string.
  var url = 'https://docs.google.com/feeds/download/documents/export/Export?id=' +
            docId + '&exportFormat=html';
  var param = {
    method: 'get',
    headers: {'Authorization': 'Bearer ' + ScriptApp.getOAuthToken()},
    muteHttpExceptions: true,
  };
  return UrlFetchApp.fetch(url, param).getContentText();
}
```