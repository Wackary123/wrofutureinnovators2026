"""
Jetson-side motor controller for the ATLAS helmet.

Connects to the EV3 over Bluetooth at startup, then exposes simple methods
to raise/lower pictures. Designed to fail gracefully — if EV3 isn't
available, the helmet keeps working without motor control.

Usage:
    motor = MotorController(server_mac="2C:6B:7D:7B:AE:02")
    motor.connect_in_background()  # non-blocking
    motor.raise_picture("slot_1")  # safe even if not connected yet
    motor.lower_all()
"""
import threading
import time
from typing import Optional

# Pybricks PC-side messaging (in ~/wrofutureinnovators2026/pybricks/)
from pybricks.messaging import BluetoothMailboxClient, TextMailbox


class MotorController:
    """Manages the Bluetooth connection to the EV3 and sends motor commands.

    All public methods are safe to call even if the EV3 is not connected —
    they will return False and log a warning rather than crash the helmet.
    """

    def __init__(self, server_mac: str, mailbox_name: str = "atlas"):
        self.server_mac = server_mac
        self.mailbox_name = mailbox_name
        self._client: Optional[BluetoothMailboxClient] = None
        self._mbox: Optional[TextMailbox] = None
        self._connected = False
        self._connect_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._connect_thread: Optional[threading.Thread] = None

    # ----------------------------------------------------------
    # Connection management
    # ----------------------------------------------------------
    def connect(self, timeout: float = 30.0) -> bool:
        """Synchronous connect. Blocks until connected or timeout."""
        with self._connect_lock:
            if self._connected:
                return True

            print(f"[Motor] Connecting to EV3 at {self.server_mac} ...")
            try:
                self._client = BluetoothMailboxClient()
                self._mbox = TextMailbox(self.mailbox_name, self._client)
                self._client.connect(self.server_mac)

                # Pybricks PC mailbox eats the first message — send a warmup
                # so real commands aren't lost. Don't wait for reply.
                self._mbox.send("ping")
                time.sleep(0.5)

                self._connected = True
                print("[Motor] EV3 connected.")
                return True
            except Exception as e:
                print(f"[Motor] EV3 connect failed: {e}")
                self._client = None
                self._mbox = None
                self._connected = False
                return False

    def connect_in_background(self) -> None:
        """Try to connect without blocking the caller. Useful at helmet
        startup so the helmet can boot even if EV3 is offline."""
        if self._connect_thread and self._connect_thread.is_alive():
            return
        self._connect_thread = threading.Thread(target=self.connect, daemon=True)
        self._connect_thread.start()

    @property
    def connected(self) -> bool:
        return self._connected

    # ----------------------------------------------------------
    # Commands
    # ----------------------------------------------------------
    def raise_picture(self, name: str) -> bool:
        """Raise the named picture, lower all others. Returns True on success."""
        return self._send_command(f"raise:{name}")

    def lower_all(self) -> bool:
        """Lower all pictures. Returns True on success."""
        return self._send_command("lower_all")

    def ping(self) -> bool:
        """Send a ping. Returns True if pong received."""
        reply = self._send_command_get_reply("ping")
        return reply == "pong"

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------
    def _send_command(self, cmd: str) -> bool:
        reply = self._send_command_get_reply(cmd)
        if reply is None:
            return False
        if reply == "ok" or reply == "pong":
            return True
        print(f"[Motor] EV3 returned: {reply}")
        return False

    def _send_command_get_reply(self, cmd: str) -> Optional[str]:
        if not self._connected or self._mbox is None:
            print(f"[Motor] Not connected, dropping command: {cmd}")
            return None

        with self._send_lock:
            try:
                self._mbox.send(cmd)
                self._mbox.wait()
                reply = self._mbox.read()
                return reply
            except Exception as e:
                print(f"[Motor] Send failed for {cmd!r}: {e}")
                self._connected = False
                return None

    def disconnect(self) -> None:
        """Cleanly close the connection."""
        with self._connect_lock:
            self._connected = False
            self._client = None
            self._mbox = None