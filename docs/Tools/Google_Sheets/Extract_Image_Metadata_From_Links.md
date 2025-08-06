# Extract Image Metadata From Links

Best used alongside in-built `IMAGE()`

## App

```javascript
/**
 * Adds a custom menu to the spreadsheet to run the script.
 */
function onOpen() {
  SpreadsheetApp.getUi()
      .createMenu('Image Metadata')
      .addItem('Get Metadata', 'getMetadataWithPrompts')
      .addToUi();
}

/**
 * Prompts the user for the URL and output ranges and then runs the main function.
 */
function getMetadataWithPrompts() {
  const ui = SpreadsheetApp.getUi();
  
  // Prompt for the URL range
  const urlRangeResponse = ui.prompt(
      'Range of image URLs ',
      '(e.g., A2:A)',
      ui.ButtonSet.OK_CANCEL);
  
  // Exit if the user clicks "Cancel"
  if (urlRangeResponse.getSelectedButton() !== ui.Button.OK) {
    return;
  }
  const sourceRange = urlRangeResponse.getResponseText();

  // Prompt for header inclusion
  const headerResponse = ui.prompt(
      'Header row included?',
      'Leave empty to include. Type "n" to omit',
      ui.ButtonSet.OK_CANCEL);

  let includeHeader = true; // Set a default value of true

  if (headerResponse.getSelectedButton() === ui.Button.OK) {
    // If the user types "no" (or "NO", "No", etc.), set includeHeader to false.
    if (headerResponse.getResponseText().toLowerCase().trim() === 'no') {
      includeHeader = false;
    }
  } else {
    // If the user clicks "Cancel," we should exit the script.
    return;
  }

  // Prompt for the output start cell
  const outputCellResponse = ui.prompt(
      'Starting output cell',
      '(e.g., B2)',
      ui.ButtonSet.OK_CANCEL);

  // Exit if the user clicks "Cancel"
  if (outputCellResponse.getSelectedButton() !== ui.Button.OK) {
    return;
  }
  const outputStartCell = outputCellResponse.getResponseText();

  // Run the main function with the collected parameters.
  getImageMetadata(sourceRange, outputStartCell, includeHeader); 
}

/**
 * Main function to get the image metadata.
 * @param {string} sourceRange A string representing the range of URLs.
 * @param {string} outputStartCell A string representing the starting cell for the output.
 * @param {boolean} includeHeader If true, a header row will be included.
 */
function getImageMetadata(sourceRange, outputStartCell, includeHeader) {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = spreadsheet.getActiveSheet();
  
  try {
    const urls = sheet.getRange(sourceRange).getValues().flat();
    const outputCell = sheet.getRange(outputStartCell);
    const allResults = [];
    
    // Add header row based on user input
    if (includeHeader) {
      allResults.push(["File Type", "Size (kB)", "Width (px)", "Height (px)", "Aspect Ratio"]);
    }
    
    const chunkSize = 100;
    
    for (let i = 0; i < urls.length; i += chunkSize) {
      const chunk = urls.slice(i, i + chunkSize);
      const requests = [];
      
      for (let j = 0; j < chunk.length; j++) {
        const url = chunk[j];
        if (!url || url.toString().trim() === '' || !is_valid_url(url)) {
          allResults.push(["Error: Invalid/Empty URL", null, null, null, null]);
          continue;
        }
        
        const encodedUrl = encodeURI(url);
        requests.push({ url: encodedUrl, muteHttpExceptions: true, originalUrl: url });
      }
      
      let responses = [];
      if (requests.length > 0) {
        responses = UrlFetchApp.fetchAll(requests);
      }
      
      let responseIndex = 0;
      for (let j = 0; j < chunk.length; j++) {
        const url = chunk[j];
        if (!url || url.toString().trim() === '' || !is_valid_url(url)) {
          // Already handled above
          continue;
        }

        let metadata = [];
        const response = responses[responseIndex];
        
        try {
          if (!response || response.getResponseCode() >= 400) {
            throw new Error(HTTP Error: ${response ? response.getResponseCode() : 'No Response'});
          }
          
          const blob = response.getBlob();
          const headers = response.getHeaders();
          const sizeBytes = blob.getBytes().length;
          const sizeKB = (sizeBytes / 1024).toFixed(2);
          const contentType = headers['Content-Type'];
          
          let fileType = null, width = null, height = null, aspectRatio = null;
          
          const match = /^(image\/(\w+));\s*width=(\d+);\s*height=(\d+)$/.exec(contentType);
          if (match) {
            fileType = match[2];
            width = parseInt(match[3], 10);
            height = parseInt(match[4], 10);
            const commonDivisor = gcd(width, height);
            aspectRatio = (width / commonDivisor) + ":" + (height / commonDivisor);
          } else {
            fileType = contentType ? contentType.split('/')[1] : null;
          }
          metadata = [fileType, sizeKB, width, height, aspectRatio];
        } catch(e) {
          metadata = [Error: ${e.message}, null, null, null, null];
        }
        
        allResults.push(metadata);
        responseIndex++;
      }
      
      if (i + chunkSize < urls.length) {
        Utilities.sleep(1 * 1000); // Sleep for 1 second
      }
    }
    
    if (allResults.length > 0) {
      const outputRange = sheet.getRange(
        outputCell.getRow(), 
        outputCell.getColumn(), 
        allResults.length, 
        allResults[0].length
      );
      outputRange.setValues(allResults);
    }
    
    SpreadsheetApp.getUi().alert('✅ Completed!');
  } catch (e) {
    SpreadsheetApp.getUi().alert('❌ Error: ' + e.message);
  }
}

// Helper function to find the greatest common divisor (GCD) using the Euclidean algorithm
function gcd(a, b) {
  return b === 0 ? a : gcd(b, a % b);
}

// Helper function to validate if a URL is in the correct format
function is_valid_url(url) {
  return url.toString().startsWith('http://') || url.toString().startsWith('https://');
}
```

## Custom Function

### Usage

```
=IMAGE_METADATA(C3) // copy-paste for all cells
=IMAGE_METADATA(C3:C) // for multiple-cells, but mostly will time-out

=ARRAYFORMULA(IMAGE(D3:D))
```

### Source Code

```javascript
/**  
 * Gets the file type, file size (kB), width (px), and height (px) of images from a list of URLs.  
 * This function is designed to process an entire range at once, avoiding the need for ARRAYFORMULA.  
 *  
 * @param {A2:A} urls A one-dimensional range of image URLs.  
 * @param {TRUE} chunkSize Optional (default = 100). # of asynchronous requests in a single batch  
 * @param {TRUE} includeHeader Optional (default = true). If true, a header row with column names will be included in the output. If false, header row will be omitted.  
 * @returns {string[][]} A two-dimensional array containing the metadata for each URL.  
 * @customfunction  
 */  

function IMAGE_METADATA(urls, chunkSize = 100, includeHeader = true) {  
  const allResults = [];  
  const urlsToProcess = Array.isArray(urls) ? urls.flat() : [urls];  

  // Add header row only for a range input with more than one row and if the includeHeader parameter is true.  
  if (Array.isArray(urls) && urls.length > 1 && includeHeader === true) {  
    allResults.push(["File Type", "Size (kB)", "Width (px)", "Height (px)", "Aspect Ratio"]);  
  }  

  for (let i = 0; i < urlsToProcess.length; i += chunkSize) {  
    const chunk = urlsToProcess.slice(i, i + chunkSize);  
    const requests = [];  
    const chunkMap = new Map();  

    // 1. Build the array of requests for the current chunk  
    for (let j = 0; j < chunk.length; j++) {  
      const url = chunk[j];  

      if (!url || url.toString().trim() === '' || !is_valid_url(url)) {  
        chunkMap.set(j, null);  
        continue;  
      }  

      const encodedUrl = encodeURI(url);  
      requests.push({ url: encodedUrl, muteHttpExceptions: true });  
      chunkMap.set(j, encodedUrl);  
    }  

    // 2. Make all valid requests in the chunk concurrently  
    let responses = [];  
    if (requests.length > 0) {  
      responses = UrlFetchApp.fetchAll(requests);  
    }  

    const encodedUrlToResponseMap = new Map(requests.map((req, index) => [req.url, responses[index]]));  

    // 3. Process the results for the current chunk  
    for (let j = 0; j < chunk.length; j++) {  
      const url = chunk[j];  

      let metadata = [];  
      // Handle empty and invalid URLs without fetching  
      if (!url || url.toString().trim() === '') {  
        metadata = [null, null, null, null, null];  
      } else if (!is_valid_url(url)) {  
        metadata = ["Error: Invalid URL", null, null, null, null];  
      } else {  

        const encodedUrl = encodeURI(url);  
        const response = encodedUrlToResponseMap.get(encodedUrl);  

        try {  
          if (!response || response.getResponseCode() >= 400) {  
            throw new Error(HTTP Error: ${response ? response.getResponseCode() : 'No Response'});  
          }  

          const blob = response.getBlob();  
          const headers = response.getHeaders();  
          const sizeBytes = blob.getBytes().length;  
          const sizeKB = (sizeBytes / 1024).toFixed(2);  
          const contentType = headers['Content-Type'];  
          const match = /^(image\/(\w+));\s*width=(\d+);\s*height=(\d+)$/.exec(contentType);  

          let fileType, width, height, aspectRatio;  
          if (match) {  
            fileType = match[2];  
            width = parseInt(match[3], 10);  
            height = parseInt(match[4], 10);  
            const commonDivisor = gcd(width, height);  
            aspectRatio = (width / commonDivisor) + ":" + (height / commonDivisor);  
          } else {  
            fileType = contentType ? contentType.split('/')[1] : null;  
            width = null;  
            height = null;  
            aspectRatio = null;  
          }  
          metadata = [fileType, sizeKB, width, height, aspectRatio];  
        } catch(e) {  
          metadata = [Error: ${e.message}, null, null, null, null];  
        }  
      }  
      allResults.push(metadata);  

    }  

    // Add a small delay between chunks to avoid rate limits  
    if (i + chunkSize < urlsToProcess.length) {  
      Utilities.sleep(1*1000); // Sleep for x seconds  
    }  
  }  
  return allResults;  

}

// Helper function to find the greatest common divisor (GCD) using the Euclidean algorithm  
function gcd(a, b) {  
  return b === 0 ? a : gcd(b, a % b);  
}  

// Helper function to validate if a URL is in the correct format  
function is_valid_url(url) {  
  return url.toString().startsWith('http://') || url.toString().startsWith('https://');  
}
```