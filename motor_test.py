"""
Standalone test for motor_controller.py.
Run after uploading ev3_motors.py to the brick.
"""
import time
from motor_controller import MotorController

EV3_MAC = "2C:6B:7D:7B:AE:02"

motor = MotorController(server_mac=EV3_MAC)
ok = motor.connect(timeout=30)
if not ok:
    print("Could not connect to EV3. Aborting test.")
    raise SystemExit(1)

print("\nTest 1: ping")
print("  ping ->", motor.ping())

print("\nTest 2: lower all")
print("  lower_all ->", motor.lower_all())
time.sleep(1)

for slot in ["slot_1", "slot_2", "slot_3", "slot_4"]:
    print(f"\nTest 3: raise {slot}")
    print(f"  raise_picture({slot!r}) ->", motor.raise_picture(slot))
    time.sleep(2)

print("\nTest 4: lower all")
print("  lower_all ->", motor.lower_all())

print("\nAll tests done.")