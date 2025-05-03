import argparse
import asyncio
import copy
import json
import math
import os
import sys
import time
import traceback
import uuid
import wave
from datetime import datetime, timedelta
from typing import List

import aiohttp
import pyaudiowpatch as pyaudio
import websockets

session_id = uuid.uuid4()
start_time = datetime.now()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 8000

mic_data: List = []
SAMPLE_SIZE = None  # Will be set later

REALTIME_RESOLUTION = 0.250

data_dir = os.path.abspath(os.path.join(os.path.curdir, "data"))
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


async def response_consumer(lang, queue):
    while True:
        data = await queue.get()
        if data is None:
            break
        response_file_name = f"{data['session_id'].hex}_{data['start_time'].isoformat()}_{data['language']}.json"
        response_file_name = response_file_name.replace(":", "-")
        response_file_path = os.path.join(data["data_dir"], response_file_name)
        with open(response_file_path, "w") as f:
            f.write(json.dumps(data["transcripts"]))
        queue.task_done()


async def subtitle_consumer(lang, queue):
    while True:
        data = await queue.get()
        if data is None:
            break
        transcript_file_name = f"{data['session_id'].hex}_{data['start_time'].isoformat()}_{data['language']}.{data['format']}"
        transcript_file_name = transcript_file_name.replace(":", "-")
        transcript_file_path = os.path.join(data["data_dir"], transcript_file_name)
        with open(transcript_file_path, "w") as f:
            f.write(json.dumps(data["sub_transcripts"]))
        queue.task_done()


async def wav_consumer(queue):
    while True:
        data = await queue.get()
        if data is None:
            break
        wave_file_name = f"{data['session_id'].hex}_{data['start_time'].isoformat()}_{data['language']}.wav"
        wave_file_name = wave_file_name.replace(":", "-")
        wave_file_path = os.path.join(data["data_dir"], wave_file_name)
        with wave.open(wave_file_path, "wb") as wave_file:
            wave_file.setnchannels(data["channels"])
            wave_file.setsampwidth(data["sample_size"])
            wave_file.setframerate(data["rate"])
            wave_file.writeframes(b"".join(data["mic_data"]))
        queue.task_done()


def subtitle_time_formatter(seconds, separator):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def subtitle_formatter(response, format, subtitle_line_counter):
    start = response["start"]
    end = start + response["duration"]
    alternatives = response.get("channel", {}).get("alternatives", [{}])[0]
    transcript = alternatives.get("transcript", "")
    words = alternatives.get("words", [])
    language = response.get("metadata", {}).get("language", "?")

    transcript_words = []
    transcript_speakers = []
    for word in words:
        speaker = f"{word.get('speaker', '?')}"
        suffix = " " * (int(math.fabs(len(word["punctuated_word"]) - len(speaker))))
        transcript_speaker = speaker if len(speaker) >= len(word["punctuated_word"]) else speaker + suffix
        transcript_speakers.append(transcript_speaker)
        transcript_word = word["punctuated_word"] if len(word["punctuated_word"]) >= len(speaker) else word[
                                                                                                           "punctuated_word"] + suffix
        transcript_words.append(transcript_word)

    separator = "," if format == "srt" else '.'
    prefix = "- " if format == "vtt" else ""
    subtitle_string = (
        f"{subtitle_line_counter}\n"
        f"{subtitle_time_formatter(start, separator)} --> "
        f"{subtitle_time_formatter(end, separator)}\n"
        f"{prefix}{' '.join(transcript_words)}\n"
        f"{prefix}{' '.join(transcript_speakers)}\n"
        f"{prefix}{language}\n\n"
    )
    return subtitle_string


def mic_callback(languages):
    def inner(input_data, frame_count, time_info, status_flags):
        mic_data.append(input_data)
        return (input_data, pyaudio.paContinue)

    return inner


async def ws_executor(key, method, format, response_queues, subtitle_queues, **kwargs):
    transcripts = []
    language = kwargs["language"]

    deepgram_url = f'{kwargs["host"]}/v1/listen?smart_format=true&no_delay=true'
    if kwargs["diarize"]:
        deepgram_url += f"&diarize={kwargs['diarize']}"
    if kwargs["language"]:
        deepgram_url += f"&language={kwargs['language']}"
    if kwargs["model"]:
        deepgram_url += f"&model={kwargs['model']}"
    if kwargs["tier"]:
        deepgram_url += f"&tier={kwargs['tier']}"
    if method == "mic":
        deepgram_url += f"&encoding=linear16&sample_rate={RATE}"
    elif method == "wav":
        deepgram_url += f'&channels={kwargs["channels"]}&sample_rate={kwargs["sample_rate"]}&encoding=linear16'

    async with websockets.connect(deepgram_url, additional_headers={"Authorization": "Token {}".format(key)}) as ws:
        print("")
        print(f'ℹ️  Deepgram URL: {deepgram_url}')
        if kwargs["model"]:
            print(f'ℹ️  Model: {kwargs["model"]}')
        if kwargs["tier"]:
            print(f'ℹ️  Tier: {kwargs["tier"]}')
        if kwargs["language"]:
            print(f'ℹ️  Language: {kwargs["language"]}')
        if kwargs["diarize"]:
            print(f'ℹ️  Diarization: {kwargs["diarize"]}')
        print("🟢 (1/5) Successfully opened Deepgram streaming connection")

        async def ws_keepalive(ws):
            while True:
                await ws.send(json.dumps({"type": "KeepAlive"}))
                await asyncio.sleep(1)

        async def ws_sender(ws):
            print(
                f'🟢 (2/5) Ready to stream {method if method in ["mic", "url"] else kwargs["filepath"]} audio to Deepgram')
            if method == "mic":
                mic_data_index = 0
                while True:
                    while not mic_data_index < len(mic_data):
                        await asyncio.sleep(0.1)
                    mic_datum = mic_data[mic_data_index]
                    mic_data_index += 1
                    await ws.send(mic_datum)
            elif method == "url":
                async with aiohttp.ClientSession() as session:
                    async with session.get(kwargs["url"]) as audio:
                        while True:
                            remote_url_data = await audio.content.readany()
                            await ws.send(remote_url_data)
                            if not remote_url_data:
                                break
            elif method == "wav":
                data = kwargs["data"]
                byte_rate = kwargs["sample_width"] * kwargs["sample_rate"] * kwargs["channels"]
                chunk_size = int(byte_rate * REALTIME_RESOLUTION)
                while len(data):
                    chunk, data = data[:chunk_size], data[chunk_size:]
                    await asyncio.sleep(REALTIME_RESOLUTION)
                    await ws.send(chunk)
                await ws.send(json.dumps({"type": "CloseStream"}))
                print("🟢 (5/5) Successfully closed Deepgram connection, waiting for final transcripts if necessary")

        async def ws_receiver(ws):
            async for msg in ws:
                res = json.loads(msg)
                res_metadata = res.setdefault("metadata", {})
                res_metadata.setdefault("language", language)
                if res.get("start") is not None:
                    res_start_time = start_time + timedelta(seconds=res["start"])
                    res_metadata.setdefault("start_time", res_start_time.isoformat())
                transcripts.append(res)
                res_data = {
                    "session_id": session_id,
                    "language": language,
                    "start_time": start_time,
                    "transcripts": transcripts,
                    "data_dir": data_dir,
                    "format": "json",
                }
                if res.get("is_final"):
                    await response_queues[language].put(res_data)
                if res.get("created"):
                    print(f"{language}")
                    print(f'🟢 Request finished with a duration of {res["duration"]} seconds. Exiting!')

        functions = [
            asyncio.create_task(ws_keepalive(ws)),
            asyncio.create_task(ws_sender(ws)),
            asyncio.create_task(ws_receiver(ws)),
        ]
        await asyncio.gather(*functions)


async def run(key, method, format, **kwargs):
    languages = kwargs["language"]
    models = kwargs["model"]
    mix_language = "_".join(languages)
    response_queues = {lang: asyncio.Queue() for lang in languages + [mix_language]}
    subtitle_queues = {lang: asyncio.Queue() for lang in languages + [mix_language]}
    wav_queue = asyncio.Queue()

    consumer_tasks = []
    for lang in languages + [mix_language]:
        consumer_tasks.append(asyncio.create_task(response_consumer(lang, response_queues[lang])))
        consumer_tasks.append(asyncio.create_task(subtitle_consumer(lang, subtitle_queues[lang])))
    consumer_tasks.append(asyncio.create_task(wav_consumer(wav_queue)))

    ws_functions = []
    for model, language in zip(models, languages):
        copied_kwargs = copy.deepcopy(kwargs)
        copied_kwargs["language"] = language
        copied_kwargs["model"] = model
        ws_functions.append(
            asyncio.create_task(
                ws_executor(
                    key,
                    method,
                    format,
                    response_queues,
                    subtitle_queues,
                    **copied_kwargs
                )
            )
        )

    async def res_receiver():
        first_transcript = True
        subtitle_line_counter = 0
        sub_transcripts = []
        while True:
            lang_responses = await asyncio.gather(*[response_queues[lang].get() for lang in languages])
            lang_transcripts = [response["transcripts"][-1] for response in lang_responses]
            lang_transcripts.sort(
                key=lambda r: r.get("channel", {}).get("alternatives", [{}])[0].get("confidence", 0),
                reverse=True
            )
            res = lang_transcripts[0]
            if res.get("is_final"):
                alternatives = res.get("channel", {}).get("alternatives", [{}])[0]
                transcript = alternatives.get("transcript", "")
                if kwargs["timestamps"]:
                    words = alternatives.get("words", [])
                    start = words[0]["start"] if words else None
                    end = words[-1]["end"] if words else None
                    transcript += f" [{start} - {end}]" if start and end else ""
                if transcript:
                    if first_transcript:
                        print("🟢 (4/5) Began receiving transcription")
                        print("")
                        if format == "vtt":
                            print("WEBVTT\n")
                        first_transcript = False
                    if format in ["vtt", "srt"]:
                        subtitle_line_counter += 1
                        transcript = subtitle_formatter(res, format, subtitle_line_counter)
                    print(transcript)
                    sub_transcripts.append(transcript)
                    if format in ["vtt", "srt"]:
                        sub_data = {
                            "session_id": session_id,
                            "language": mix_language,
                            "start_time": start_time,
                            "sub_transcripts": sub_transcripts,
                            "format": format,
                            "data_dir": data_dir,
                        }
                        await subtitle_queues[mix_language].put(sub_data)
                        if method == "mic":
                            wav_data = {
                                "session_id": session_id,
                                "language": mix_language,
                                "start_time": start_time,
                                "mic_data": mic_data,
                                "channels": CHANNELS,
                                "sample_size": SAMPLE_SIZE,
                                "rate": RATE,
                                "format": "wav",
                                "data_dir": data_dir,
                            }
                            await wav_queue.put(wav_data)
            for lang in languages:
                response_queues[lang].task_done()

    async def microphone():
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=kwargs["device"],
            frames_per_buffer=CHUNK,
            stream_callback=mic_callback(languages),
        )
        stream.start_stream()
        global SAMPLE_SIZE
        SAMPLE_SIZE = audio.get_sample_size(FORMAT)
        while stream.is_active():
            await asyncio.sleep(0.1)
        stream.stop_stream()
        stream.close()

    functions = []
    if method == "mic":
        functions.append(asyncio.create_task(microphone()))
    functions.extend(ws_functions)
    functions.append(asyncio.create_task(res_receiver()))
    functions.extend(consumer_tasks)
    await asyncio.gather(*functions)


def validate_input(input):
    if input.lower().startswith("mic"):
        return input
    elif input.lower().endswith("wav"):
        if os.path.exists(input):
            return input
    elif input.lower().startswith("http"):
        return input
    raise argparse.ArgumentTypeError(f'{input} is an invalid input.')


def validate_format(format):
    if format.lower() in ["text", "vtt", "srt"]:
        return format
    raise argparse.ArgumentTypeError(f'{format} is invalid. Please enter "text", "vtt", or "srt".')


def validate_dg_host(dg_host):
    if dg_host.startswith("wss://") or dg_host.startswith("ws://"):
        return dg_host.rstrip('/')
    raise argparse.ArgumentTypeError(f'{dg_host} is invalid. Please provide a WebSocket URL.')


def parse_args():
    parser = argparse.ArgumentParser(description="Submits data to the real-time streaming endpoint.")
    parser.add_argument("-k", "--key", default=os.environ.get("DEEPGRAM_API_KEY"), help="YOUR_DEEPGRAM_API_KEY")
    parser.add_argument("-i", "--input", nargs="?", const=1, type=validate_input, help="Input to stream to Deepgram.")
    parser.add_argument("--device", type=int, required=True, help="Device index for microphone.")
    parser.add_argument("--model", nargs="+", type=str, default=["nova-3", "nova-3"], help="Model to use.")
    parser.add_argument("--tier", nargs="?", const="", default="", help="Model tier.")
    parser.add_argument("--timestamps", nargs="?", const=1, default=False, help="Include timestamps.")
    parser.add_argument("--format", nargs="?", const=1, default="text", type=validate_format, help="Output format.")
    parser.add_argument("--language", nargs="+", type=str, default=["en", "id"], help="Language of audio.")
    parser.add_argument("--diarize", nargs="?", const=1, default=False, help="Enable diarization.")
    parser.add_argument("--host", nargs="?", const=1, default="wss://api.deepgram.com", type=validate_dg_host,
                        help="Deepgram host.")
    return parser.parse_args()


def main():
    args = parse_args()
    input = args.input
    format = args.format.lower()
    host = args.host
    while True:
        try:
            if input.lower().startswith("mic"):
                asyncio.run(run(args.key, "mic", format, model=args.model, tier=args.tier, host=host,
                                timestamps=args.timestamps, language=args.language, diarize=args.diarize,
                                device=args.device))
            elif input.lower().endswith("wav"):
                with wave.open(input, "rb") as fh:
                    channels, sample_width, sample_rate, num_samples, _, _ = fh.getparams()
                    assert sample_width == 2, "WAV data must be 16-bit."
                    data = fh.readframes(num_samples)
                    asyncio.run(
                        run(args.key, "wav", format, model=args.model, tier=args.tier, data=data, channels=channels,
                            sample_width=sample_width, sample_rate=sample_rate, filepath=input, host=host,
                            timestamps=args.timestamps, language=args.language, diarize=args.diarize))
            elif input.lower().startswith("http"):
                asyncio.run(run(args.key, "url", format, model=args.model, tier=args.tier, url=input, host=host,
                                timestamps=args.timestamps, language=args.language, diarize=args.diarize))
            else:
                raise argparse.ArgumentTypeError(f'🔴 {input} is an invalid input.')
        except Exception as e:
            print(f"🔴 ERROR: {e}")
            traceback.print_exc()
        time.sleep(0.1)


if __name__ == "__main__":
    sys.exit(main() or 0)
