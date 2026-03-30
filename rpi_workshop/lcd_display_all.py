import time
from lcd_i2c import LCD_I2C

lcd = LCD_I2C(0x27, 16, 2)
lcd.backlight.on()
lcd.clear()

# Fill both lines with text
lcd.cursor.setPos(0, 0)
lcd.write_text("1234567890123456")  # Fill entire first line
lcd.cursor.setPos(1, 0)
lcd.write_text("ABCDEFGHIJKLMNOP")  # Fill entire second line

# Keep it running
while True:
    time.sleep(1)
