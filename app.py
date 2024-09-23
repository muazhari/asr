import argparse
import asyncio
import copy
import itertools
import json
import math
import multiprocessing
import os
import sys
import traceback
import uuid
import wave
from datetime import datetime
from multiprocessing.connection import Connection
from typing import Dict, List, Any

import aiohttp
import pyaudio
import websockets
from future.backports.datetime import timedelta

session_id = uuid.uuid4()
start_time = datetime.now()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 8000

mic_data: List = []
processes: Dict[str, multiprocessing.Process] = {}
pipes: Dict[str, tuple[Connection, Connection]] = {}
collections: Dict[str, Any] = {}

# Mimic sending a real-time stream by sending this many seconds of audio at a time.
# Used for file "streaming" only.
REALTIME_RESOLUTION = 0.250

data_dir = os.path.abspath(
    os.path.join(os.path.curdir, "data")
)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def res_executor(pipe: Connection):
    for data in iter(pipe.recv, None):
        try:
            response_file_name = f"{data['session_id'].hex}_{data['start_time'].isoformat()}_{data['language']}.{data['format']}"
            response_file_name = response_file_name.replace(":", "-")
            response_file_path = os.path.abspath(
                os.path.join(
                    data["data_dir"],
                    response_file_name,
                )
            )
            with open(response_file_path, "w") as f:
                f.write(json.dumps(data["all_responses"]))
        except Exception as e:
            traceback.print_exc()


def sub_executor(pipe: Connection):
    for data in iter(pipe.recv, None):
        try:
            transcript_file_name = f"{data['session_id'].hex}_{data['start_time'].isoformat()}_{data['language']}.{data['format']}"
            transcript_file_name = transcript_file_name.replace(":", "-")
            transcript_file_path = os.path.abspath(
                os.path.join(
                    data["data_dir"],
                    transcript_file_name,
                )
            )
            with open(transcript_file_path, "w") as f:
                f.write("".join(data["all_transcripts"]))
        except Exception as e:
            traceback.print_exc()


def wav_executor(pipe):
    for data in iter(pipe.recv, None):
        try:
            wave_file_name = f"{data['session_id'].hex}_{data['start_time'].isoformat()}_{data['language']}.{data['format']}"
            wave_file_name = wave_file_name.replace(":", "-")
            wave_file_path = os.path.abspath(
                os.path.join(
                    data["data_dir"],
                    wave_file_name,
                )
            )
            with wave.open(wave_file_path, "wb") as wave_file:
                wave_file.setnchannels(data["channels"])
                wave_file.setsampwidth(data["sample_size"])
                wave_file.setframerate(data["rate"])
                wave_file.writeframes(b"".join(data["mic_data"]))
        except Exception as e:
            traceback.print_exc()


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

    # attach speaker diarization below transcript words
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


# Used for microphone streaming only.
def mic_callback(languages):
    def inner(input_data, frame_count, time_info, status_flags):
        mic_data.append(input_data)
        return (input_data, pyaudio.paContinue)

    return inner


async def ws_executor(key, method, format, **kwargs):
    all_transcripts = []
    all_responses = []

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
        deepgram_url += "&encoding=linear16&sample_rate=16000"

    elif method == "wav":
        data = kwargs["data"]
        deepgram_url += f'&channels={kwargs["channels"]}&sample_rate={kwargs["sample_rate"]}&encoding=linear16'

    # Connect to the real-time streaming endpoint, attaching our credentials.
    async with websockets.connect(
            deepgram_url, extra_headers={"Authorization": "Token {}".format(key)}
    ) as ws:
        print("")
        print(f'ℹ️  Request ID: {ws.response_headers.get("dg-request-id")}')
        print(f'ℹ️  Deepgram URL: {deepgram_url}')
        print(f'ℹ️  Deepgram Request Headers:\n{ws.request_headers}'.strip())
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
                f'🟢 (2/5) Ready to stream {method if (method == "mic" or method == "url") else kwargs["filepath"]} audio to Deepgram{". Speak into your microphone to transcribe." if method == "mic" else ""}'
            )

            if method == "mic":
                mic_data_index = 0
                while True:
                    while not mic_data_index < len(mic_data):
                        await asyncio.sleep(0.1)
                    mic_datum = mic_data[mic_data_index]
                    mic_data_index += 1
                    await ws.send(mic_datum)

            elif method == "url":
                # Listen for the connection to open and send streaming audio from the URL to Deepgram
                async with aiohttp.ClientSession() as session:
                    async with session.get(kwargs["url"]) as audio:
                        while True:
                            remote_url_data = await audio.content.readany()
                            await ws.send(remote_url_data)

                            # If no data is being sent from the live stream, then break out of the loop.
                            if not remote_url_data:
                                break

            elif method == "wav":
                nonlocal data
                # How many bytes are contained in one second of audio?
                byte_rate = (
                        kwargs["sample_width"] * kwargs["sample_rate"] * kwargs["channels"]
                )
                # How many bytes are in `REALTIME_RESOLUTION` seconds of audio?
                chunk_size = int(byte_rate * REALTIME_RESOLUTION)

                while len(data):
                    chunk, data = data[:chunk_size], data[chunk_size:]
                    # Mimic real-time by waiting `REALTIME_RESOLUTION` seconds
                    # before the next packet.
                    await asyncio.sleep(REALTIME_RESOLUTION)
                    # Send the data
                    await ws.send(chunk)

                await ws.send(json.dumps({"type": "CloseStream"}))
                print(
                    "🟢 (5/5) Successfully closed Deepgram connection, waiting for final transcripts if necessary"
                )
            return

        async def ws_receiver(ws):
            """Print out the messages received from the server."""
            first_message = True
            first_transcript = True
            transcript = ""
            subtitle_line_counter = 0

            async for msg in ws:
                res = json.loads(msg)
                res_metadata = res.setdefault("metadata", {})
                res_metadata.setdefault("language", kwargs["language"])
                if res.get("start", None) is not None:
                    res_start_time = start_time + timedelta(seconds=res["start"])
                    res_metadata.setdefault("start_time", res_start_time.isoformat())
                all_responses.append(res)
                res_data: Dict = {
                    "session_id": session_id,
                    "language": kwargs["language"],
                    "start_time": start_time,
                    "all_responses": all_responses,
                    "data_dir": data_dir,
                    "format": "json",
                }
                res_key = f"{kwargs['language']}/res"
                pipes[res_key][0].send(res_data)

                if first_message:
                    # print(
                    #     "🟢 (3/5) Successfully receiving Deepgram messages, waiting for finalized transcription..."
                    # )
                    first_message = False
                # handle local server messages
                if res.get("msg"):
                    print(f"{kwargs['language']}")
                    print(res['msg'])
                if res.get("is_final"):
                    response_collection: List = collections[res_key]
                    response_collection.append(res)

                    alternatives = res.get("channel", {}).get("alternatives", [{}])[0]
                    transcript = alternatives.get("transcript", "")

                    if kwargs["timestamps"]:
                        words = res.get("channel", {}).get("alternatives", [{}])[0].get("words", [])
                        start = words[0]["start"] if words else None
                        end = words[-1]["end"] if words else None
                        transcript += " [{} - {}]".format(start, end) if (start and end) else ""
                    if transcript != "":
                        if first_transcript:
                            print(f"{kwargs['language']}")
                            print("🟢 (4/5) Began receiving transcription")
                            # if using webvtt, print out header
                            if format == "vtt":
                                pass
                                # print("WEBVTT\n")
                            first_transcript = False
                        if format == "vtt" or format == "srt":
                            subtitle_line_counter += 1
                            transcript = subtitle_formatter(res, format, subtitle_line_counter)
                        # print(transcript)
                        all_transcripts.append(transcript)

                        # save subtitle data if specified
                        if format == "vtt" or format == "srt":
                            sub_data: Dict = {
                                "session_id": session_id,
                                "language": kwargs["language"],
                                "start_time": start_time,
                                "all_transcripts": all_transcripts,
                                "format": format,
                                "data_dir": data_dir,
                            }
                            sub_pipe_key = f"{kwargs['language']}/sub"
                            pipes[sub_pipe_key][0].send(sub_data)

                if res.get("created"):
                    print(f"{kwargs['language']}")
                    print(
                        f'🟢 Request finished with a duration of {res["duration"]} seconds. Exiting!'
                    )
                    pass

        functions = [
            asyncio.ensure_future(ws_keepalive(ws)),
            asyncio.ensure_future(ws_sender(ws)),
            asyncio.ensure_future(ws_receiver(ws)),
        ]

        await asyncio.gather(*functions)


async def run(key, method, format, **kwargs):
    ws_functions = []
    mix_language = "_".join(kwargs["language"])
    mix_model = "_".join(kwargs["model"])
    for model, language in zip(kwargs["model"] + [mix_model], kwargs["language"] + [mix_language], strict=True):
        copied_kwargs = copy.deepcopy(kwargs)
        copied_kwargs["language"] = language
        copied_kwargs["model"] = model
        res_key = f"{language}/res"
        sub_key = f"{language}/sub"
        wav_key = f"{language}/wav"
        res_collection = []
        collections[res_key] = res_collection
        res_pipe = multiprocessing.Pipe()
        sub_pipe = multiprocessing.Pipe()
        wav_pipe = multiprocessing.Pipe()
        pipes[res_key] = res_pipe
        pipes[sub_key] = sub_pipe
        pipes[wav_key] = wav_pipe
        res_process = multiprocessing.Process(target=res_executor, args=(res_pipe[1],))
        sub_process = multiprocessing.Process(target=sub_executor, args=(sub_pipe[1],))
        wav_process = multiprocessing.Process(target=wav_executor, args=(wav_pipe[1],))
        res_process.start()
        sub_process.start()
        wav_process.start()
        processes[res_key] = res_process
        processes[sub_key] = sub_process
        processes[wav_key] = wav_process
        if language != mix_language:
            ws_functions.append(asyncio.ensure_future(ws_executor(key, method, format, **copied_kwargs)))

    async def res_receiver():
        """Print out the messages received from the server."""
        first_message = True
        first_transcript = True
        all_transcripts = []
        all_responses = []
        subtitle_line_counter = 0
        response_index = 0
        response_collections = [collections[f"{language}/res"] for language in kwargs["language"]]
        last_responses = {}

        while True:
            while not all(
                    response_index < len(response_collection)
                    for response_collection in response_collections
            ):
                await asyncio.sleep(0.1)

            responses = [response_collection[response_index] for response_collection in response_collections]
            response_index += 1

            def response_comparator(response):
                alternatives = response.get("channel", {}).get("alternatives", [])
                for alternative in alternatives:
                    words = alternative.get("words", [])
                    alternative_word_confidences = sum([word["confidence"] for word in words])
                    alternative_word_confidences /= len(words) if len(words) > 0 else 1
                    alternative["confidence"] = alternative_word_confidences

                alternatives.sort(
                    key=lambda alt: alt.get("confidence", 0),
                    reverse=True
                )

                return alternatives[0].get("confidence", 0) if len(alternatives) > 0 else 0

            responses.sort(
                key=response_comparator,
                reverse=True
            )

            try:
                assert all(
                    responses[i].get("channel").get("alternatives")[0].get("confidence") >= \
                    responses[j].get("channel").get("alternatives")[0].get("confidence")
                    for i, j in zip(range(len(responses)), range(1, len(responses)), strict=False)
                    if responses[i].get("is_final") and responses[j].get("is_final")
                )

                assert all(
                    response.get("start") >= last_response.get("start")
                    for response, last_response in itertools.product(responses, last_responses.values())
                    if (response.get("metadata").get("language") == last_response.get("metadata").get("language"))
                    and (response.get("is_final") and last_response.get("is_final"))
                )
            except Exception as e:
                print(json.dumps(responses))
                print(json.dumps(last_responses))
                raise e

            res = responses[0]
            res_language = res.get("metadata").get("language")
            last_responses[res_language] = res
            all_responses.append(res)

            res_key = f"{mix_language}/res"
            res_data: Dict = {
                "session_id": session_id,
                "language": mix_language,
                "start_time": start_time,
                "all_responses": all_responses,
                "data_dir": data_dir,
                "format": "json",
            }
            pipes[res_key][0].send(res_data)

            if first_message:
                print("")
                print(
                    "🟢 (3/5) Successfully receiving Deepgram messages, waiting for finalized transcription..."
                )
                first_message = False

            # handle local server messages
            if res.get("msg"):
                print(res["msg"])
            if res.get("is_final"):
                alternatives = res.get("channel", {}).get("alternatives", [{}])[0]
                transcript = alternatives.get("transcript", "")

                if kwargs["timestamps"]:
                    words = res.get("channel", {}).get("alternatives", [{}])[0].get("words", [])
                    start = words[0]["start"] if words else None
                    end = words[-1]["end"] if words else None
                    transcript += " [{} - {}]".format(start, end) if (start and end) else ""
                if transcript != "":
                    if first_transcript:
                        print("🟢 (4/5) Began receiving transcription")
                        print("")
                        # if using webvtt, print out header
                        if format == "vtt":
                            print("WEBVTT\n")
                        first_transcript = False
                    if format == "vtt" or format == "srt":
                        subtitle_line_counter += 1
                        transcript = subtitle_formatter(res, format, subtitle_line_counter)
                    print(transcript)
                    all_transcripts.append(transcript)

                    # save subtitle data if specified
                    if format == "vtt" or format == "srt":
                        sub_data: Dict = {
                            "session_id": session_id,
                            "language": mix_language,
                            "start_time": start_time,
                            "all_transcripts": all_transcripts,
                            "format": format,
                            "data_dir": data_dir,
                        }
                        sub_pipe_key = f"{mix_language}/sub"
                        pipes[sub_pipe_key][0].send(sub_data)

                        # also save mic data if we were live streaming audio
                        # otherwise the wav file will already be saved to disk
                        if method == "mic":
                            wav_data: Dict = {
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
                            wav_pipe_key = f"{mix_language}/wav"
                            pipes[wav_pipe_key][0].send(wav_data)

            if res.get("created"):
                print(
                    f'🟢 Request finished with a duration of {res["duration"]} seconds. Exiting!'
                )

    # Set up microphone if streaming from mic
    async def microphone():
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=7,
            frames_per_buffer=CHUNK,
            stream_callback=mic_callback(kwargs["language"]),
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
        functions.append(asyncio.ensure_future(microphone()))

    functions.extend(ws_functions)
    functions.append(asyncio.ensure_future(res_receiver()))

    await asyncio.gather(*functions)


def validate_input(input):
    if input.lower().startswith("mic"):
        return input

    elif input.lower().endswith("wav"):
        if os.path.exists(input):
            return input

    elif input.lower().startswith("http"):
        return input

    raise argparse.ArgumentTypeError(
        f'{input} is an invalid input. Please enter the path to a WAV file, a valid stream URL, or "mic" to stream from your microphone.'
    )


def validate_format(format):
    if (
            format.lower() == ("text")
            or format.lower() == ("vtt")
            or format.lower() == ("srt")
    ):
        return format

    raise argparse.ArgumentTypeError(
        f'{format} is invalid. Please enter "text", "vtt", or "srt".'
    )


def validate_dg_host(dg_host):
    if (
            # Check that the host is a websocket URL
            dg_host.startswith("wss://")
            or dg_host.startswith("ws://")
    ):
        # Trim trailing slash if necessary
        if dg_host[-1] == '/':
            return dg_host[:-1]
        return dg_host

    raise argparse.ArgumentTypeError(
        f'{dg_host} is invalid. Please provide a WebSocket URL in the format "{{wss|ws}}://hostname[:port]".'
    )


def parse_args():
    """Parses the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Submits data to the real-time streaming endpoint."
    )
    parser.add_argument(
        "-k", "--key", required=True, help="YOUR_DEEPGRAM_API_KEY (authorization)"
    )
    parser.add_argument(
        "-i",
        "--input",
        help='Input to stream to Deepgram. Can be "mic" to stream from your microphone (requires pyaudio), the path to a WAV file, or the URL to a direct audio stream. Defaults to the included file preamble.wav',
        nargs="?",
        const=1,
        default="preamble.wav",
        type=validate_input,
    )
    parser.add_argument(
        "-m",
        "--model",
        help='Which model to make your request against. Defaults to none specified. See https://developers.deepgram.com/docs/models-overview for all model options.',
        nargs="+",
        type=str,
        default=["nova-2-meeting", "nova-2"]
    )
    parser.add_argument(
        "-t",
        "--tier",
        help='Which model tier to make your request against. Defaults to none specified. See https://developers.deepgram.com/docs/tier for all tier options.',
        nargs="?",
        const="",
        default="",
    )
    parser.add_argument(
        "-ts",
        "--timestamps",
        help='Whether to include timestamps in the printed streaming transcript. Defaults to False.',
        nargs="?",
        const=1,
        default=False,
    )
    parser.add_argument(
        "-f",
        "--format",
        help='Format for output. Can be "text" to return plain text, "VTT", or "SRT". If set to VTT or SRT, the audio file and subtitle file will be saved to the data/ directory. Defaults to "text".',
        nargs="?",
        const=1,
        default="text",
        type=validate_format,
    )
    parser.add_argument(
        "-l",
        "--language",
        help='The language of the audio data. Defaults to ["en", "id"].',
        nargs="+",
        type=str,
        default=["en", "id"],
    )
    parser.add_argument(
        "-d",
        "--diarize",
        help='Whether to diarize the audio data. Defaults to False.',
        nargs="?",
        const=1,
        default=False,
    )

    # Parse the host
    parser.add_argument(
        "--host",
        help='Point the test suite at a specific Deepgram URL (useful for on-prem deployments). Takes "{{wss|ws}}://hostname[:port]" as its value. Defaults to "wss://api.deepgram.com".',
        nargs="?",
        const=1,
        default="wss://api.deepgram.com",
        type=validate_dg_host,
    )
    return parser.parse_args()


def main():
    """Entrypoint for the example."""
    # Parse the command-line arguments.
    args = parse_args()
    input = args.input
    format = args.format.lower()
    host = args.host

    try:
        if input.lower().startswith("mic"):
            asyncio.run(
                run(args.key, "mic", format,
                    model=args.model,
                    tier=args.tier,
                    host=host,
                    timestamps=args.timestamps,
                    language=args.language,
                    diarize=args.diarize)
            )

        elif input.lower().endswith("wav"):
            if os.path.exists(input):
                # Open the audio file.
                with wave.open(input, "rb") as fh:
                    (
                        channels,
                        sample_width,
                        sample_rate,
                        num_samples,
                        _,
                        _,
                    ) = fh.getparams()
                    assert sample_width == 2, "WAV data must be 16-bit."
                    data = fh.readframes(num_samples)
                    asyncio.run(
                        run(
                            args.key,
                            "wav",
                            format,
                            model=args.model,
                            tier=args.tier,
                            data=data,
                            channels=channels,
                            sample_width=sample_width,
                            sample_rate=sample_rate,
                            filepath=args.input,
                            host=host,
                            timestamps=args.timestamps,
                            language=args.language,
                            diarize=args.diarize,
                        )
                    )
            else:
                raise argparse.ArgumentTypeError(
                    f"🔴 {args.input} is not a valid WAV file."
                )

        elif input.lower().startswith("http"):
            asyncio.run(run(args.key, "url", format,
                            model=args.model,
                            tier=args.tier,
                            url=input,
                            host=host,
                            timestamps=args.timestamps,
                            language=args.language,
                            diarize=args.diarize))

        else:
            raise argparse.ArgumentTypeError(
                f'🔴 {input} is an invalid input. Please enter the path to a WAV file, a valid stream URL, or "mic" to stream from your microphone.'
            )

    except websockets.exceptions.InvalidStatusCode as e:
        print(f'🔴 ERROR: Could not connect to Deepgram! {e.headers.get("dg-error")}')
        print(
            f'🔴 Please contact Deepgram Support (developers@deepgram.com) with request ID {e.headers.get("dg-request-id")}'
        )
        traceback.print_exc()
        return
    except websockets.exceptions.ConnectionClosedError as e:
        error_description = f"Unknown websocket error."
        print(
            f"🔴 ERROR: Deepgram connection unexpectedly closed with code {e.code} and payload {e.reason}"
        )

        if e.reason == "DATA-0000":
            error_description = "The payload cannot be decoded as audio. It is either not audio data or is a codec unsupported by Deepgram."
        elif e.reason == "NET-0000":
            error_description = "The service has not transmitted a Text frame to the client within the timeout window. This may indicate an issue internally in Deepgram's systems or could be due to Deepgram not receiving enough audio data to transcribe a frame."
        elif e.reason == "NET-0001":
            error_description = "The service has not received a Binary frame from the client within the timeout window. This may indicate an internal issue in Deepgram's systems, the client's systems, or the network connecting them."

        print(f"🔴 {error_description}")
        print(
            f"🔴 Please contact Deepgram Support (developers@deepgram.com) with the request ID listed above."
        )
        traceback.print_exc()
        return

    except websockets.exceptions.ConnectionClosedOK:
        traceback.print_exc()
        return

    except Exception as e:
        print(f"🔴 ERROR: {e}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    sys.exit(main() or 0)
