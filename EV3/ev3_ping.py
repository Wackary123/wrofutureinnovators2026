"""
EV3 ping test - listens over Bluetooth, echoes messages back.
"""
from pybricks.hubs import EV3Brick
from pybricks.messaging import BluetoothMailboxServer, TextMailbox

ev3 = EV3Brick()
ev3.speaker.beep()
ev3.screen.clear()
ev3.screen.print("Ping test")
ev3.screen.print("Waiting for BT...")

server = BluetoothMailboxServer()
mbox = TextMailbox("atlas", server)

server.wait_for_connection()
ev3.screen.print("Connected!")
ev3.speaker.beep()

while True:
    mbox.wait()
    cmd = mbox.read()
    ev3.screen.print("Got: " + cmd)
    mbox.send("pong:" + cmd)