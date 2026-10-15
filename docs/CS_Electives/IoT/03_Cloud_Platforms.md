# Cloud Platforms

## Arduino IoT Cloud

HTTP

- Install Arduino create agent
- Go to [create.arduino.cc/iot/](create.arduino.cc/iot/)

### Things

1. Create thing
2. Create variables
3. Select devices
4. Select network
5. Set timezone
6. Full code editor

### Dashboards

1. Create dashboard
2. Add widgets

### Mobile App

Arduino IoT Cloud Remote

### Uploading code locally

Install `ArduinoIoTCloud` library

## Blynk

HTTP

[blynk.cloud](blynk.cloud)

1. Create account
2. Create new template
3. Create new data stream > Virtual Pin
   - Digital: Input only
   - Analog: Input only
   - Virtual Pin: Input/Output
4. Install Blynk library

## ThingSpeak

Developed by MathWorks, the same company that developed Matlab

Uses HTTP Read and Write requests

## AdaFruit IO

Uses MQTT

### Subscribe

```cpp
Adafruit_MTT_Client mqtt(&client, server, port, user, key);
Adafruit_MTT_Subscribe toggle = Adafruit_MTT_Subscribe(
  &mqtt,
  user"/feeds/led"
);

void mqtt_connect() {
  if(mqtt.connected()){
    return ;
  }
  
  Serial.println("Connecting to MQTT...");
  
  int retries = 3, status;
  
  while(
    (status = mqtt.connect()) != 0
  ) {
    Serial.println(mqtt.connectErrorString(status));
    Serial.println("Retrying after 5sec");
    
    delay(5000);
    retries--;
    
    if(retries == 0) {
      while(1); // reset NodeMCU
    }
  }
  
  Serial.println("MQTT connected");
  mqtt.subscribe(&toggle);
}

void setup() {
  Serial.begin(9600);
  pinMode(D2, OUTPUT);
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, pass);
  while(WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(1000);
  }
  Serial.println("WiFi connected!")
}

void loop() {
  mqtt_connect();
  Adafruit_MTT_Subscribe *subscription;
  
  while(subscription = mqtt.readSubscription(5000)) {
    if(subscription == &toggle) {
      char* data = (char*) toggle.lastread;
      Serial.println(data);
    }
  }
}
```

### Publish

```cpp
Adafruit_MTT_Client mqtt(&client, server, port, user, key);
Adafruit_MTT_Publish gauge = Adafruit_MTT_Publish(
  &mqtt,
  user"/feeds/sensor"
);

int data;

void mqtt_connect() {
  if(mqtt.connected()){
    return ;
  }
  
  Serial.println("Connecting to MQTT...");
  
  int retries = 3, status;
  
  while(
    (status = mqtt.connect()) != 0
  ) {
    Serial.println(mqtt.connectErrorString(status));
    Serial.println("Retrying after 5sec");
    
    delay(5000);
    retries--;
    
    if(retries == 0) {
      while(1); // reset NodeMCU
    }
  }
  
  Serial.println("MQTT connected");
}

void setup() {
  Serial.begin(9600);
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, pass);
  while(WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(1000);
  }
  Serial.println("WiFi connected!")
}

void loop() {
  mqtt_connect();
  
  data = 100;
  
  if(gauge.publish(data)) {
    Serial.println("Published successful: " + String(data));
  } else {
    Serial.println("Published failed: " + String(data));
  }
  delay(5000);
}
```

## IFTTT

- WebHooks for HTTP Requests

## AWS IoT Core
