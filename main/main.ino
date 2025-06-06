const int analogPin = A0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int voltage = analogRead(analogPin);
  char str[5]; // 4자리 + null 문자
  sprintf(str, "%04d", voltage);

  Serial.println(str);  // 문자열 출력
  delay(100);                 // 500ms마다 갱신
}