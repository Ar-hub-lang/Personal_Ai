import os
import re
import subprocess
import sys
import wave
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHISPER_CLI = PROJECT_ROOT / ".whisper-runtime/build/bin/whisper-cli"
WHISPER_BASE_MODEL = PROJECT_ROOT / "models/ggml-base.bin"
WHISPER_SMALL_MODEL = PROJECT_ROOT / "models/ggml-small.bin"
SAMPLE_RATE = 16_000
MAX_SECONDS = 20
MAX_AUDIO_BYTES = SAMPLE_RATE * 2 * MAX_SECONDS + 4096


def transcribe(audio: bytes, prompt: str = "") -> str:
    try:
        with wave.open(BytesIO(audio), "rb") as recording:
            frames = recording.getnframes()
            if (
                recording.getnchannels() != 1
                or recording.getsampwidth() != 2
                or recording.getframerate() != SAMPLE_RATE
                or not 0 < frames <= SAMPLE_RATE * MAX_SECONDS
                or len(recording.readframes(frames)) != frames * 2
            ):
                raise ValueError("Send up to 20 seconds of 16 kHz mono PCM WAV audio.")
    except (wave.Error, EOFError) as exc:
        raise ValueError("The recording is not a valid WAV file.") from exc

    model = WHISPER_SMALL_MODEL if WHISPER_SMALL_MODEL.is_file() else WHISPER_BASE_MODEL
    if not WHISPER_CLI.is_file() or not model.is_file():
        raise FileNotFoundError("Whisper model or CLI is missing.")

    with TemporaryDirectory(prefix="personal-ai-voice-") as directory:
        base = Path(directory)
        recording_path = base / "recording.wav"
        recording_path.write_bytes(audio)
        command = [
            str(WHISPER_CLI), "-m", str(model), "-f", str(recording_path),
            "-l", "auto", "-nt", "-otxt", "-of", str(base / "transcript"),
        ]
        if prompt:
            command.extend(["--prompt", prompt])

        env = os.environ.copy()
        if sys.platform == "darwin":
            # A CMake-built whisper-cli may still reference the original checkout.
            library_paths = [str(WHISPER_CLI.parent)]
            if env.get("DYLD_LIBRARY_PATH"):
                library_paths.append(env["DYLD_LIBRARY_PATH"])
            env["DYLD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
        subprocess.run(
            command,
            env=env,
            capture_output=True,
            check=True,
            timeout=30,
        )
        return (base / "transcript.txt").read_text(encoding="utf-8").strip()


def synthesize(reply: str) -> bytes:
    with TemporaryDirectory(prefix="personal-ai-voice-") as directory:
        base = Path(directory)
        text_path = base / "reply.txt"
        audio_path = base / "reply.aiff"
        wav_path = base / "reply.wav"
        # Treat model output as text rather than macOS speech commands.
        text_path.write_text(re.sub(r"\[{2,}", "[", reply), encoding="utf-8")
        subprocess.run(
            ["/usr/bin/say", "-f", str(text_path), "-o", str(audio_path)],
            capture_output=True,
            check=True,
            timeout=30,
        )
        subprocess.run(
            ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16", str(audio_path), str(wav_path)],
            capture_output=True,
            check=True,
            timeout=15,
        )
        audio = wav_path.read_bytes()
        try:
            with wave.open(BytesIO(audio), "rb") as recording:
                if not recording.getnframes():
                    raise OSError("macOS returned an empty spoken reply.")
        except (wave.Error, EOFError) as exc:
            raise OSError("macOS returned an invalid spoken reply.") from exc
        return audio
