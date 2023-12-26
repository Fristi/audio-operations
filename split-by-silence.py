import auditok
import os

file_list = os.listdir('input')
# counter
j = 0

# Enumerate through the list of files
for index, filename in enumerate(sorted(file_list)):
    if filename == ".DS_Store": continue
    audio_regions = auditok.split(f"input/{filename}", min_dur=0.2, max_dur=320, max_silence=0.3, energy_threshold=55)
    for _, r in enumerate(audio_regions):
        if r.duration > 2.5:
            j += 1
            filename = r.save(f"out/{j}.wav")