# GSheets API for Custom Form

## App Script

```javascript
const DATA_ENTRY_SHEET_NAME = "Registration";
const TIME_STAMP_COLUMN_NAME = "Timestamp"; // You can edit the name of this column name or you can put blank like this : "". Ensure that the same name exist there in the sheet as well.


var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(DATA_ENTRY_SHEET_NAME);

const doPost = (request = {}) => {
  const { postData: { contents, type } = {} } = request;
  var data = parseFormData(contents);
  try {
    appendToGoogleSheet(data);
  } catch (e) {

  }

  try {
    send_confirmation_mail();
  } catch (e) {

  }


 return ContentService.createTextOutput(contents).setMimeType(ContentService.MimeType.JSON);
};

function send_confirmation_mail(data) {
  url = "https://btf.pythonanywhere.com/send-registration-confirmation?n=" + data["Name"].replace(" ", "+") + "&i=" + data["Institution"].replace(" ", "+") + "&e=" + data["Email"].replace(" ", "");
  UrlFetchApp.fetch(url);
}

function parseFormData(postData) {
  var data = [];
  var parameters = postData.split('&');
  for (var i = 0; i < parameters.length; i++) {
    var keyValue = parameters[i].split('=');
    data[keyValue[0]] = decodeURIComponent(keyValue[1]);
  }
  return data;
}

function appendToGoogleSheet(data) {
  if(TIME_STAMP_COLUMN_NAME !==""){
    data[TIME_STAMP_COLUMN_NAME]=new Date();
  }
  var headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
  var rowData = headers.map(headerFld => data[headerFld]);
  sheet.appendRow(rowData);
}
```

## Client-Side

```javascript
const API_LINK = "https://script.google.com/macros/s/AKfycbz_FzYhH1h0WZIhvpLicgxWQxqpFnkUGAvLN-oTUdMkImJe_hWqDlaQT8GdPn5MPsVmVA/exec";

const Technofest = () => {
  const handleSubmit = (event) => {
    event.preventDefault();
    document.getElementById("message").textContent = "Submitting..";
    document.getElementById("message").style.display = "block";
    document.getElementById("submit-button").disabled = true;

    // Collect the form data
    var formData = new FormData(event.target);
    var keyValuePairs = [];
    for (var pair of formData.entries()) {
      keyValuePairs.push(pair[0] + "=" + pair[1]);
    }

    var formDataString = keyValuePairs.join("&");

    // Send a POST request to your Google Apps Script
    fetch(API_LINK, {
      redirect: "follow",
      method: "POST",
      body: formDataString,
      headers: {
        "Content-Type": "text/plain;charset=utf-8",
      },
    })
      .then(function (response) {
        // Check if the request was successful
        if (response) {
          return response; // Assuming your script returns JSON response
        } else {
          throw new Error("Failed to submit the form.");
        }
      })
      .then(function (data) {
        // Display a success message
        document.getElementById("message").textContent =
          "Data submitted successfully!";
        document.getElementById("message").style.display = "block";
        document.getElementById("message").style.backgroundColor = "green";
        document.getElementById("message").style.color = "beige";
        document.getElementById("submit-button").disabled = false;
        event.target.reset();

        setTimeout(function () {
          document.getElementById("message").textContent = "";
          document.getElementById("message").style.display = "none";
        }, 2600);
      })
      .catch(function (error) {
        // Handle errors, you can display an error message here
        console.error(error);
        document.getElementById("message").textContent =
          "An error occurred while submitting the form." + ": " + error;
        document.getElementById("message").style.display = "block";
      });
  };
```

