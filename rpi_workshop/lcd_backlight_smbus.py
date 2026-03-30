import smbus2
import time

bus = smbus2.SMBus(1)
addr = 0x27

# Send initialization sequence and backlight on
try:
    # These are raw LCD initialization commands
    commands = [0x33, 0x32, 0x28, 0x0C, 0x06, 0x01]
    for cmd in commands:
        bus.write_byte(addr, cmd)
        time.sleep(0.01)
    
    print("Init sent - check if backlight is on now")
    time.sleep(5)
    
except Exception as e:
    print(f"Error: {e}")
