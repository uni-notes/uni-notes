# NodeJS

Open source server environment, that allows you to run JavaScript on the server, ie allows JS to run outside the browser.

## Installation

https://nodejs.org

## Web Server

### `index.html`

```js
<html>
  <body>
  	"Hello world!"
  </body>
</html>
```

### `app.js`

```js
// imports
const http = require("http")
const fs = require("fs") // file system

const port = 3000

const server = http.createServer(function (req, res){
  
  fs.readFile(
    "index.html",
    function(error, data) {
      if (error) {
        res.writeHead(404)
        res.write("Error: File not found")
      } else {
        res.writeHead(200, {
          "Content-Type": "text/html"
        })
        res.write(data)
      }
    }
    res.end()
  )
  
  // res.end()
})

server.listen(port, function(error) {
  if (error) {
    console.log("Something went wrong: ", error)
  } else {
    console.log("Server listening on port " + port)
  }
})
```

```js
node app.js
```

