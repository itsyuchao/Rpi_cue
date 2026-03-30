import time
from lcd_i2c import LCD_I2C

lcd = LCD_I2C(0x27, 16, 2)

# Force initialization
lcd.backlight.on()
lcd.clear()

# Write all over the screen
for row in range(2):
    for col in range(16):
        lcd.cursor.setPos(row, col)
        lcd.write_text("#")
        time.sleep(0.1)

time.sleep(10)
