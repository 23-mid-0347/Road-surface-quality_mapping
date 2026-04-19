#include <Wire.h>
#include <TinyGPSPlus.h>
#include <HardwareSerial.h>
#include <WiFi.h>
#include <math.h>

// ================= WIFI =================
const char* ssid = "Nothing";          // Phone hotspot SSID
const char* password = "xxxxxxxxx";  // Phone hotspot password
WiFiServer server(8080);             // TCP server for Termux

#define LED_PIN 2   // Built-in LED (try 5 if yours differs)

// ================= GPS =================
TinyGPSPlus gps;
HardwareSerial GPS_Serial(1);
#define GPS_RX 16
#define GPS_TX 17

// ================= IMU (MPU6050) =================
#define MPU_ADDR 0x68
#define PWR_MGMT_1 0x6B
#define ACCEL_XOUT_H 0x3B

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);

  // ---------- WIFI ----------
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(500);

  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN)); // Blink while connecting
    delay(500);
  }

  digitalWrite(LED_PIN, HIGH); // Solid ON = WiFi connected
  Serial.println("WiFi connected");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  server.begin();

  // ---------- GPS ----------
  GPS_Serial.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);

  // ---------- IMU ----------
  Wire.begin(21, 22);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(PWR_MGMT_1);
  Wire.write(0x00);   // Wake MPU6050
  Wire.endTransmission();

  Serial.println("Waiting for GPS fix...");
}

void loop() {

  // ---------- UPDATE GPS ----------
  while (GPS_Serial.available()) {
    gps.encode(GPS_Serial.read());
  }

  // Do not log until GPS is valid
  if (!gps.location.isValid()) {
    return;
  }

  // ---------- WAIT FOR PHONE (nc) ----------
  WiFiClient client = server.available();
  if (!client) return;

  Serial.println("Client connected, sending data...");

  // ---------- SEND CSV HEADER ONCE ----------
  client.println("time,lat,lon,speed,ax,ay,az,acc_mag,gx,gy,gz,label");

  // ---------- STREAM DATA ----------
  while (client.connected()) {

    // Keep GPS updated
    while (GPS_Serial.available()) {
      gps.encode(GPS_Serial.read());
    }

    if (!gps.location.isValid()) continue; // Pause if GPS lost

    // ---- Read IMU ----
    int16_t ax, ay, az, gx, gy, gz;

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(ACCEL_XOUT_H);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 14);

    ax = (Wire.read() << 8) | Wire.read();
    ay = (Wire.read() << 8) | Wire.read();
    az = (Wire.read() << 8) | Wire.read();

    Wire.read(); Wire.read(); // Temperature (ignored)

    gx = (Wire.read() << 8) | Wire.read();
    gy = (Wire.read() << 8) | Wire.read();
    gz = (Wire.read() << 8) | Wire.read();

    // ---- Features ----
    float acc_mag = sqrt(ax * ax + ay * ay + az * az);
    float speed = gps.speed.kmph();
    int label = 0;   // Set later during ML preprocessing

    // ---- CSV ROW ----
    client.print(millis()); client.print(",");
    client.print(gps.location.lat(), 6); client.print(",");
    client.print(gps.location.lng(), 6); client.print(",");
    client.print(speed); client.print(",");
    client.print(ax); client.print(",");
    client.print(ay); client.print(",");
    client.print(az); client.print(",");
    client.print(acc_mag); client.print(",");
    client.print(gx); client.print(",");
    client.print(gy); client.print(",");
    client.print(gz); client.print(",");
    client.println(label);

    delay(200); // ~5 Hz sampling
  }

  Serial.println("Client disconnected");
}
