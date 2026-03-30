import time
from lcd_i2c import LCD_I2C

lcd = LCD_I2C(0x27, 16, 2)

# Turn on the backlight
lcd.backlight.on()

# Show the blinking cursor
lcd.blink.on()

lcd.cursor.setPos(0, 5)
lcd.write_text('Hello')
lcd.cursor.setPos(1, 1)
lcd.write_text('Raspberry Pi!')
time.sleep(10)
