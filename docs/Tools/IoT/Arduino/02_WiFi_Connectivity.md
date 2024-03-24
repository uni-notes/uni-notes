# WiFi Connectivity

```cpp
WiFi.scanNetworks();
WiFi.SSID(i);
WiFi.RSSI(i);
WiFi.begin(ssid, pass);
WiFi.status();
WiFi.localIP();
```

```cpp
WiFiServer server(local_port);
server.begin();
```

```cpp
WiFiClient = server.available();
client.readStringUntil("\r");
```

## Show WiFi Networks

```cpp
#include <ESP8266WiFi.h>

void setup(){
	Serial.begin(9600);
}

void loop(){
  Serial.println("Scanning WiFi");

  int no_of_networks = WiFi.scanNetworks(); // only 2.4GHz for NodeMCU
  
  if (n==0) {
    Serial.println("No networks available");
  } else {
    for (int i=0; i<n; i++) {
      Serial.println(
      	String(i+1) + " " + WiFi.SSID(i) + " " + String(WiFi.RSSI(i))
      );
    }
  }
  
  delay(5000);
}
```

## Connect to Network

```cpp
#include <ESP8266WiFi.h>

// variables
// we need to use these exact types and names, to override the one in the header file
char* ssid = "WiFi Name";
char* pass = "WiFi Password";

void setup(){
	Serial.begin(9600);
  pinMode(D2, OUTPUT);
  
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, pass);
  
  
  while(WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(1000);
  }
  
  Serial.println("Connected to " + String(ssid) + " successfully");
}

void loop(){
  // code
  
  delay(5000);
}
```

## Local Server

```cpp
#include <ESP8266WiFi.h>

// variables
// we need to use these exact types and names, to override the one in the header file
char* ssid = "WiFi Name";
char* pass = "WiFi Password";

int local_port = 80;
WiFiServer server(local_port);

void setup(){
	Serial.begin(9600);
  pinMode(D2, OUTPUT);
  
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, pass);
  
  
  while(WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(1000);
  }
  
  Serial.println("Connected to " + String(ssid) + " successfully.");
  
  server.begin(local_port);
  
  local_ip = WiFi.localIP();
  
  Serial.println(
    "Started server on " + String(local_ip) + ":" + String(local_port) + " successfully."
  );
}

void loop(){
  WiFiClient client = server.available();
  if (!client) {
    return;
  }
  Serial.println("New request received!");
  String request = client.readStringUntil("\r");
  
  query_path = "/on"
  /*
  localhost/on turns on LED
  localhost/off turns off LED
  */
  if (request.indexOf(query_path) != -1){
    digitalWrite(D2, HIGH);
    Serial.println("LED turned on");
  }
  else if (request.indexOf(query_path) != -1){
    digitalWrite(D2, LOW);
    Serial.println("LED turned off");
  } else {
    Serial.println("Invalid request");
  }
  
  client_interaction_code = "" +
    "<html><body>" + 
    "<button><a href='/on'>On</a></button>" +
    "<button><a href='/off'>Off</a></button>" +
    "</body></html>";
  
  delay(5000);
}
```

## Client

```cpp
#include <ESP8266WiFi.h>

// variables
// we need to use these exact types and names, to override the one in the header file
char* ssid = "WiFi Name";
char* pass = "WiFi Password";

int local_port = 80;
WiFiClient client;

char* api_key = "";
int id_server = ;
char ip_server[] = "";

void setup() {
  Serial.begin(9600);
  
  Serial.println("Connecting to WiFi ...");
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED)
  {
    Serial.print(".");
    delay(1000);
  }
  Serial.println("WiFi connected");
  
  // ThingSpeak.begin(client);
}

void loop() {
  if (client.connect(ip_server, local_port)) {
    ThingSpeak.setField(1, data);
    ThingSpeak.writeFields(id_server, api);
  }
  
  delay(5000);
}
```

## HTTP Requests

### Send

```cpp
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

// variables
// we need to use these exact types and names, to override the one in the header file
char* ssid = "WiFi Name";
char* pass = "WiFi Password";

HTTPClient client;
String api;
int data;
int status_code;
String response;

void setup(){
	Serial.begin(9600);
  pinMode(D2, OUTPUT);
  
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, pass);
  
  
  while(WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(1000);
  }
  
  Serial.println("Connected to " + String(ssid) + " successfully");
}

void loop(){
  data = 100;
  api = "http://.../insert.php?data=" + String(data);
  
  client.begin(api);
  
  status_code = client.GET();
  response = client.getString();
     
  delay(5000);
}
```

