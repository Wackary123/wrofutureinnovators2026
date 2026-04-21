"""
ATLAS — Museum Helmet main script.

This version drops the Fusion HAT STT wrapper entirely and drives the
MillSO MQ5 USB lavalier mic directly via sounddevice + vosk.

Architecture (unchanged from previous version)
----------------------------------------------
Three background threads:
  1. Camera thread   — YOLOE every N frames, enqueues trigger events.
  2. Gemini worker   — drains request_queue, streams Gemini responses.
  3. STT thread      — sounddevice -> vosk KaldiRecognizer.

Main loop reads utterances, decides IDLE vs ACTIVE mode, and enqueues
user-turn requests.

Hardware
--------
Mic:     MillSO MQ5 USB lavalier on ALSA card 3 (see `arecord -l`).
Speaker: YARCHONN 3.5mm (pending). Audio out currently uses ALSA default.
"""

import json
import os
import re
import time
import threading
import queue
import subprocess
import tempfile

import cv2
import numpy as np
import sounddevice as sd
import vosk  # type: ignore

from dotenv import load_dotenv
from google import genai

from picamera2 import Picamera2  # type: ignore
from ultralytics import YOLOE  # type: ignore

try:
    import t