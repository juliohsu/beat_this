import soundfile as sf
import numpy as np
import torch
from pathlib import Path

from beat_this.inference import File2Beats, Audio2Frames, Audio2Beats

from scipy.io import wavfile

def test_File2Beat():
    f2b = File2Beats()
    audio_path = Path("tests/bia_jazz.mpga")
    beat, downbeat = f2b(audio_path)
    assert isinstance(beat, np.ndarray)
    assert isinstance(downbeat, np.ndarray)
    return beat, downbeat

def test_File2Beat2():
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
    # load audio
    audio, sr = sf.read(audio_path)
    beat, downbeat = a2f(audio, sr)
    assert isinstance(beat, torch.Tensor)
    assert isinstance(downbeat, torch.Tensor)

def test_Audio2Beats():
    a2b = Audio2Beats()
    audio_signals_path = Path("tests/bia_jazz.mpga")
    audio_signals_np = np.load(audio_signals_path)
    audio_signals_tensor = torch.tensor(audio_signals_np, dtype=torch.float32)
    beat, downbeat = a2b(audio_signals_tensor, 22050)
    assert isinstance(beat, np.ndarray)
    assert isinstance(downbeat, np.ndarray)
    return beat, downbeat

def test_Signal2Beat():
    audio_signals_path = Path("tests/bia_jazz.mpga")
    audio_signals_np = np.load(audio_signals_path)
    wavfile.write('rwc_popular_CD1_01.wav', 22050, audio_signals_np.astype(np.float32))
    print('finish wav convert!')

if __name__ == "__main__":
    print("running custom audio signal to beats inference test")

    # beat, downbeat = test_Audio2Beats()

    # with open('beat_pop1_beat.txt', 'w') as f:
    #     f.write('\n'.join(map(str, beat.tolist())))
        
    # with open('beat_pop1_downbeat.txt', 'w') as f:
    #     f.write('\n'.join(map(str, downbeat.tolist())))

    # print(f"test_File2Beat: OK - Beat: {beat.shape}")
    # print(f"test_File2Beat: OK - Downbeat: {downbeat.shape}")

    # test_Signal2Beat()

    beat, downbeat = test_File2Beat2()

    with open('beat_jazz_beat.txt', 'w') as f:
        f.write('\n'.join(map(str, beat.tolist())))
        
    with open('beat_jazz_downbeat.txt', 'w') as f:
        f.write('\n'.join(map(str, downbeat.tolist())))

    # print(f"test_File2Beat: OK - Beat: {beat.shape}")
    # print(f"test_File2Beat: OK - Downbeat: {downbeat.shape}")

