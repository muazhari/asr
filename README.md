# voice-transcriber

> A simple voice transcriber application using Deepgram.

## Tutorial

### 1. Install the dependencies.

```bash
uv pip install -e . -U
```

### 2. Get device index.

```bash
python audio_io.py
```

### 3. Run the app variant.

- Single Language

```bash
python app.py --input mic --device 1 --model nova-3 --language en --format srt --diarize true
```

- Multiple Language

```bash
python app.py --input mic --device 1 --model nova-3 nova-3 --language en id --format srt --diarize true
```
