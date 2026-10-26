/*
  Hack in Hills '26 robot - Arduino Uno firmware
  UNTESTED on hardware: test with the wheels off the ground first.

  Hardware assumed:
    - L293D motor shield (Adafruit v1 style) + "AFMotor" library (Library Manager: "Adafruit Motor Shield V1")
      Left motors on M1+M2, right motors on M3+M4  (swap in code if a side runs backwards)
    - HC-05 Bluetooth on the shield's servo headers:
        pin 10 (Arduino RX <- HC-05 TXD)
        pin 9  (Arduino TX -> HC-05 RXD, through a 1k/2k voltage divider)
        HC-05 VCC -> 5V pin of a servo header, GND -> GND pin of a servo header
    - HC-SR04 ultrasonic (optional): TRIG = A0, ECHO = A1
    - MPU6050 gyro (optional) on I2C: SDA = A4, SCL = A5, address 0x68

  Commands (text lines, over Bluetooth at 9600 baud or USB serial at 115200):
    M <left> <right>   speeds -255..255. If left == right (> 0) the robot HOLDS ITS HEADING with the gyro.
    S                  stop
    RESET              zero the yaw angle
  The motors stop by themselves if no valid command arrives for 0.6 s (watchdog), so keep re-sending.

  Telemetry over USB serial (10 Hz):  STAT:<yaw_deg>:<front_cm>:<turn_rate_dps>:<mode>   mode = I idle, S straight, C curving
  Over Bluetooth only READY and ESTOP are sent (see BT_TELEMETRY).
*/
#include <AFMotor.h>
#include <SoftwareSerial.h>
#include <Wire.h>

// ---------- set these to 1 only once the part is really wired ----------
#define HAVE_ULTRASONIC 0   // 0 = no HC-SR04 connected (a floating ECHO pin would cause false emergency stops)
#define HAVE_GYRO       0   // 0 = no MPU6050 connected
#define BT_TELEMETRY    0   // 1 = also send STAT lines over Bluetooth. SoftwareSerial is half-duplex, so
                            //     sending while the laptop is sending can make the robot miss commands.

AF_DCMotor ML1(1), ML2(2), MR1(3), MR2(4);
SoftwareSerial bt(10, 9);             // RX = pin 10 (SER1 header), TX = pin 9 (SER2 header)

const int TRIG = A0, ECHO = A1;
const uint8_t MPU = 0x68;
const float GYRO_SIGN = 1.0;          // if heading hold corrects the WRONG way, change to -1.0
const float KP = 3.0;                 // motor PWM added per degree of heading error
const int STOP_CM = 15;

int cmdL = 0, cmdR = 0;
bool hold = false, gyroOk = false, estop = false;
float yaw = 0, target = 0, rate = 0, bias = 0;
int frontCm = 999;
byte closeCount = 0;
unsigned long tPrev, tStat, tDist, tCmd = 0;

struct LineBuf { char b[24]; byte n; bool over; };
LineBuf btBuf = {{0}, 0, false}, usbBuf = {{0}, 0, false};

float readGz() {
  Wire.beginTransmission(MPU);
  Wire.write(0x47);                    // GYRO_ZOUT_H
  Wire.endTransmission(false);
  if (Wire.requestFrom(MPU, (uint8_t)2) != 2) return bias;   // read failed: report "no rotation"
  int16_t raw = (Wire.read() << 8) | Wire.read();
  return raw / 131.0;                  // deg/s at +-250 dps range
}

int readFrontCm() {
  digitalWrite(TRIG, LOW); delayMicroseconds(3);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  unsigned long us = pulseIn(ECHO, HIGH, 12000UL);   // ~2 m max
  return us == 0 ? 999 : (int)(us / 58);
}

void driveSide(AF_DCMotor &a, AF_DCMotor &b, int v) {
  int s = abs(v); if (s > 255) s = 255;
  a.setSpeed(s); b.setSpeed(s);
  uint8_t d = v > 0 ? FORWARD : (v < 0 ? BACKWARD : RELEASE);
  a.run(d); b.run(d);
}

void sayEvent(const __FlashStringHelper *s) { bt.println(s); Serial.println(s); }

void handle(char *s) {
  int l, r;
  if (s[0] == 'M' && sscanf(s + 1, "%d %d", &l, &r) == 2) {
    tCmd = millis();
    cmdL = constrain(l, -255, 255);
    cmdR = constrain(r, -255, 255);
    bool straight = (cmdL == cmdR && cmdL > 0);
    if (straight && !hold) target = yaw;           // lock the current heading
    hold = straight;
  } else if (s[0] == 'S' && s[1] == 0) {
    tCmd = millis();
    cmdL = cmdR = 0; hold = false;
  } else if (strncmp(s, "RESET", 5) == 0) {
    tCmd = millis();
    yaw = 0; target = 0;
  }
}

void feed(Stream &st, LineBuf &lb) {
  while (st.available()) {
    char c = st.read();
    if (c == '\n' || c == '\r') {
      if (lb.n && !lb.over) { lb.b[lb.n] = 0; handle(lb.b); }
      lb.n = 0; lb.over = false;
    } else if (lb.n < sizeof(lb.b) - 1) {
      lb.b[lb.n++] = c;
    } else {
      lb.over = true;                              // line too long: ignore it completely
    }
  }
}

void setup() {
  Serial.begin(115200);
  bt.begin(9600);                      // HC-05 default data baud
#if HAVE_ULTRASONIC
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
#endif
#if HAVE_GYRO
  Wire.begin();
  Wire.beginTransmission(MPU); Wire.write(0x6B); Wire.write(0);       // wake MPU6050
  gyroOk = (Wire.endTransmission() == 0);
  if (gyroOk) {
    delay(200);
    for (int i = 0; i < 300; i++) { bias += readGz(); delay(3); }     // keep the robot still for ~1 s at power-on
    bias /= 300.0;
  } else {
    Serial.println(F("GYRO NOT FOUND - heading hold disabled"));
  }
#endif
  tPrev = micros(); tStat = tDist = millis();
  sayEvent(F("READY"));
}

void loop() {
  unsigned long now = micros();
  float dt = (now - tPrev) / 1e6; tPrev = now;
  if (gyroOk) {
    rate = GYRO_SIGN * (readGz() - bias);
    yaw += rate * dt;
  }

#if HAVE_ULTRASONIC
  if (millis() - tDist >= 60) {
    tDist = millis();
    frontCm = readFrontCm();
    closeCount = (frontCm < STOP_CM) ? min(closeCount + 1, 10) : 0;   // need 2 close readings in a row (ignores glitches)
  }
#endif

  feed(bt, btBuf); feed(Serial, usbBuf);
  if (millis() - tCmd > 600) { cmdL = cmdR = 0; hold = false; }       // no command for 0.6 s -> stop (laptop crashed / link lost)

  int l = cmdL, r = cmdR;
  if (hold && gyroOk) {                             // heading hold: gyro corrects drift
    float corr = constrain(KP * (target - yaw), -80, 80);
    l = constrain(l - corr, -255, 255);
    r = constrain(r + corr, -255, 255);
  }
  bool block = (closeCount >= 2) && (cmdL + cmdR) > 0;   // emergency stop: something close and we are moving forward
  if (block) {
    l = r = 0;
    if (!estop) sayEvent(F("ESTOP"));               // announce once, not every loop (spamming would stall the loop)
  }
  estop = block;
  driveSide(ML1, ML2, l);
  driveSide(MR1, MR2, r);

  if (millis() - tStat >= 100) {
    tStat = millis();
    char mode = (abs(cmdL) + abs(cmdR) < 10) ? 'I' : (fabs(rate) < 6 ? 'S' : 'C');
    Serial.print(F("STAT:")); Serial.print(yaw, 1);
    Serial.print(':'); Serial.print(frontCm);
    Serial.print(':'); Serial.print(rate, 1);
    Serial.print(':'); Serial.println(mode);
#if BT_TELEMETRY
    bt.print(F("STAT:")); bt.print(yaw, 1);
    bt.print(':'); bt.print(frontCm);
    bt.print(':'); bt.print(rate, 1);
    bt.print(':'); bt.println(mode);
#endif
  }
}
