```javascript
const SHEETID = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx';
const DOCID = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx';
const FOLDERID = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx';

function onOpen(){
  SpreadsheetApp.getUi()
  .createMenu('Generate Invoices Workflow')
  .addItem('Manually Add Month Folder/ Import New Data','xxxx')
  .addItem('Generate PDF Invoices','sender')
  .addToUi();
}

function sender () {
  const sheet = SpreadsheetApp.openById(SHEETID).getSheetByName('InvoiceData');
  const InvoiceData = sheet.getDataRange().getValues();
  const rows = InvoiceData.slice(1);
  // Logger.log(rows);

  const temp = DriveApp.getFileById(DOCID);
  const folder = DriveApp.getFolderById(FOLDERID);

  //Loop through each spreadsheet row, and for each row, create a new temp document in your drive folder
  rows.forEach((row,index)=>{
    const file = temp.makeCopy(folder);
    const doc = DocumentApp.openById(file.getId());
    const body = doc.getBody();

    //Loop through the spreadsheets heading values and populate those values into the temp document
    InvoiceData[0].forEach((heading,i)=>{
      const header1 = heading.toUpperCase();
      body.replaceText(`{${header1}}`,row[i]);
      })

      //Set a name for each document using x, y from the data
      doc.setName('INV-'+row[0]+' '+row[2]+'.doc');
      const blob = doc.getAs(MimeType.PDF);
      doc.saveAndClose();
      const pdf = folder.createFile(blob).setName('INV-'+row[0]+' '+row[2]+'.pdf');
      //The followinng code removes the temp doc file, leaving just the PDF files.
      file.setTrashed(true);
    
    //TESTING
    // console.log(header1);
    //TESTING
  })
}
```