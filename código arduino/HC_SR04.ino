/*
 * HC-SR04 example sketch with time
 * based on this post
 * https://create.arduino.cc/projecthub/Isaac100/getting-started-with-the-hc-sr04-ultrasonic-sensor-036380
 * by Isaac100
 */
const int trigPin = 9;
const int echoPin = 10;

float duration, distance;
unsigned long previousMillis = 0;
const long interval = 200; // Change this value to set the interval in milliseconds

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    duration = pulseIn(echoPin, HIGH);
    // Print data in CSV format
    Serial.print(currentMillis);
    Serial.print(",");
    Serial.print(duration);
    Serial.println();
  }
}
