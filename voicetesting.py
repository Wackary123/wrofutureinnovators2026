
from fusion_hat.tts import Espeak

# Create Espeak TTS instance
tts = Espeak()
# Set amplitude 0-200, default 100
tts.set_amp(200)
# Set speed 80-260, default 150
tts.set_speed(150)
# Set gap 0-200, default 1
tts.set_gap(2)
# Set pitch 0-99, default 80
tts.set_pitch(99)

tts.say("Hello! I’m Espeak TTS.")
"""

from fusion_hat.tts import Pico2Wave

# Create Pico2Wave TTS instance
tts = Pico2Wave()

# Set the language
tts.set_lang('en-US')  # en-US, en-GB, de-DE, es-ES, fr-FR, it-IT

# Quick hello (sanity check)
tts.say("HELLO! THIS IS THE ENLGISH US")
print("en-us")

# Set the language
tts.set_lang('en-GB')  # en-US, en-GB, de-DE, es-ES, fr-FR, it-IT

# Quick hello (sanity check)
tts.say("HELLO! THIS IS ENLGISH GB")
print("en-gb")

# Set the language
tts.set_lang('de-DE')  # en-US, en-GB, de-DE, es-ES, fr-FR, it-IT

# Quick hello (sanity check)
tts.say("HELLO! THIS IS DELIMITED DELTA")
print("de-DE")

# Set the language
tts.set_lang('es-ES')  # en-US, en-GB, de-DE, es-ES, fr-FR, it-IT

# Quick hello (sanity check)
tts.say("HELLO! THIS IS SPANISH ESPANOL")
print("es-ES")

# Set the language
tts.set_lang('fr-FR')  # en-US, en-GB, de-DE, es-ES, fr-FR, it-IT

# Quick hello (sanity check)
tts.say("HELLO! THIS IS FRENCH LANGUAGE")
tts.say("BONJOUR JE M'APPELLE ATLAS")
print("fr-FR")

# Set the language
tts.set_lang('it-IT')  # en-US, en-GB, de-DE, es-ES, fr-FR, it-IT

# Quick hello (sanity check)
tts.say("HELLO THIS IS IT bUT noT IT")
print("it-IT")
"""