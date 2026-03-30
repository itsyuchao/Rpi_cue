import time
from lcd_i2c import LCD_I2C

lcd = LCD_I2C(0x27, 16, 2)

print("Backlight ON for 5 seconds")
lcd.backlight.on()
time.sleep(5)

print("Backlight OFF for 5 seconds")
lcd.backlight.off()
time.sleep(5)

print("Backlight ON again")
lcd.backlight.on()
time.sleep(5)
