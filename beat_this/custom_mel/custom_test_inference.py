import soundfile as sf
import numpy as np
import torch
from pathlib import Path

# from beat_this.inference import File2Beats, Audio2Frames

from beat_this.custom_vqt.custom_inference import File2Beats, Audio2Frames

def test_File2Beat():
    f2b = File2Beats()
    audio_path = Path("tests/bia_jazz.mpga")
    audio_path2 = '/home/juliohsu/Desktop/dev/beats/beat_this/tests/bia_jazz.mpga'
    beat, downbeat = f2b(audio_path2)
    assert isinstance(beat, np.ndarray)
    assert isinstance(downbeat, np.ndarray)
    return beat, downbeat


def test_Audio2Frames():
    a2f = Audio2Frames()
    audio_path = Path("tests/bia_jazz.mpga")
    audio_path2 = '/home/juliohsu/Desktop/dev/beats/beat_this/tests/bia_jazz.mpga'
    # load audio
    audio, sr = sf.read(audio_path2)
    beat, downbeat = a2f(audio, sr)
    assert isinstance(beat, torch.Tensor)
    assert isinstance(downbeat, torch.Tensor)
    return beat, downbeat

if __name__ == "__main__":

    print('running custom inference test')

    beat, downbeat = test_File2Beat()
    print('beat shape:', beat.shape)
    print('downbeat shape:', downbeat.shape)

    with open('/home/juliohsu/Desktop/dev/beats/beat_this/beat_this/custom_vqt/inferece_beats_from_test/beat_jazz_beat_vqt_512.txt', 'w') as f:
        f.write('\n'.join(map(str, beat.tolist())))
        
    with open('/home/juliohsu/Desktop/dev/beats/beat_this/beat_this/custom_vqt/inferece_beats_from_test/beat_jazz_downbeat_vqt_512.txt', 'w') as f:
        f.write('\n'.join(map(str, downbeat.tolist())))
