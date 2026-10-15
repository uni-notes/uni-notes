# Digital Humidity and Temperature

Measure humidity and temperature

2 variants

|                                          | DHT 11             | DHT 22             |
| ---------------------------------------- | ------------------ | ------------------ |
| Temperature Range                        | 0 - 50C            | -40 - 125C         |
| Temperature Accuracy                     | &pm; 2C            | &pm; 0.5C          |
| Humidity Range                           | 20-80%             | 0-100%             |
| Humidity Accuracy                        | &pm; 5%            | &pm; 2-5%          |
| Sampling Rate<br />(readings per second) | 1Hz                | 0.5Hz              |
| Body Size                                | 15.5 x 12 x 5.5 mm | 15.1 x 25 x 7.7 mm |
| Operating Voltage                        | 3-5V               | 3-5V               |
| Max Current during measurement           | 2.5mA              | 2.5mA              |

## Working

$$
\text{HH} \ \text{LH} \ \text{HT} \ \text{LT} \ \text{CP}
$$

where

- $HH=$ High Humidity -> Humidity Reading in %
- $LH=$ Low Humidity
- $HT=$ High Temperature -> Temperature Reading
- $LT=$ Low Temperature
- $CP=$ Checksum Parity

## Code

```cpp
#include <DHT.h> // not in-built

DHT dht(pin_name, type_of_sensor);

dht.begin();

dht.readTemperature(); // returns Temperature in C
dht.readTemperature(True); // returns Temperature in F
// returns nan for invalid value 

dht.readHumidity() // returns Humidity %
// returns nan for invalid value
```

```cpp
#include <DHT.h>

DHT dht(D1, DHT11);
float hum, temp;

void setup() {
  dht.begin();
  Serial.begin(9600);
}
void loop() {
  hum = dht.readHumidity();
  temp = dht.readTemperature();
  
  Serial.println(
    String(hum) + " " + String(temp)
  );
  
  delay(5000);
}
```

### Dependencies

- AdaFruit Unified Sensor
- DHT_Sensor
- Time
- TinyGSM

## Sensor

3 pins

- VCC
- GND
- DOUT/Data/Signal
