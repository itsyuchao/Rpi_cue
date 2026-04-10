import time
import board
import busio
import adafruit_drv2605

# Init I2C and driver
i2c = busio.I2C(board.SCL, board.SDA)
drv = adafruit_drv2605.DRV2605(i2c)
drv.use_ERM()

# Play all 123 built-in effects
effect_id=16
for i in range(10):
    print(f"Playing effect #{effect_id}")
    drv.sequence[0] = adafruit_drv2605.Effect(effect_id)
    drv.play()
    time.sleep(2)
    drv.stop()
