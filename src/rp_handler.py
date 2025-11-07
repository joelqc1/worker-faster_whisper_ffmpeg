"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""
import base64
import os
import tempfile
import subprocess
from typing import Optional, Union

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict


# ---------------------------
# Model setup
# ---------------------------
MODEL = predict.Predictor()
MODEL.setup()


# ---------------------------
# Helpers
# ---------------------------
def parse_time_to_seconds(value: Optional[Union[str, float, int]]) -> Optional[float]:
    """
    Accepts:
      - float/int seconds (e.g. 12.5)
      - string seconds (e.g. "12.5")
      - string HH:MM:SS(.ms) (e.g. "00:01:12.500")
    Returns seconds as float or None.
    """
    if value is None:
        return None

    # Already numeric
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return None

    if ":" in s:
        # HH:MM:SS(.ms)
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = parts
        elif len(parts) == 2:  # MM:SS
            h, m, sec = "0", parts[0], parts[1]
        else:
            raise ValueError(f"Invalid time format: {value}")

        try:
            h = float(h)
            m = float(m)
            sec = float(sec)
        except Exception:
            raise ValueError(f"Invalid time format: {value}")

        return h * 3600.0 + m * 60.0 + sec

    # plain seconds as string
    try:
        return float(s)
    except Exception:
        raise ValueError(f"Invalid time value: {value}")


def trim_audio(input_path: str, start_time=None, end_time=None) -> str:
    """
    Trim audio/video with ffmpeg and output a WAV mono 16k file for Whisper.
    start_time/end_time can be numbers (seconds) or strings (seconds or HH:MM:SS).
    Returns a path to a temporary WAV file.
    """
    st = parse_time_to_seconds(start_time)
    et = parse_time_to_seconds(end_time)

    if st is None and et is None:
        return input_path  # nothing to do

    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]

    # Use input seeking when start is provided for speed
    if st is not None:
        cmd += ["-ss", str(st)]

    cmd += ["-i", input_path]

    # Duration/end handling
    if et is not None:
        if st is not None and et > st:
            cmd += ["-t", str(et - st)]
        else:
            # If only end_time is set (or et <= st), use absolute -to from start of input
            cmd += ["-to", str(et)]

    # Re-encode to a consistent WAV for Whisper
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_path]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        # Include some stderr for debugging
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}: {proc.stderr[:4000]}")

    return out_path


def base64_to_tempfile(base64_file: str) -> str:
    """
    Convert base64 file to tempfile (WAV extension by default).
    Returns a path to a temporary file.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
        return temp_file.name


# ---------------------------
# Main handler
# ---------------------------
@rp_debugger.FunctionTimer
def run_whisper_job(job):
    """
    Run inference on the model.

    Parameters:
        job (dict): Input job containing the model parameters

    Returns:
        dict: The result of the prediction
    """
    job_input = job["input"]
    temp_paths_to_cleanup = []  # collect extra temp files we create

    # ---- Validation ----
    with rp_debugger.LineTimer("validation_step"):
        input_validation = validate(job_input, INPUT_VALIDATIONS)
        if "errors" in input_validation:
            return {"error": input_validation["errors"]}
        job_input = input_validation["validated_input"]

    # ---- Input checks ----
    if not job_input.get("audio", False) and not job_input.get("audio_base64", False):
        return {"error": "Must provide either audio or audio_base64"}

    if job_input.get("audio", False) and job_input.get("audio_base64", False):
        return {"error": "Must provide either audio or audio_base64, not both"}

    # ---- Obtain audio file ----
    if job_input.get("audio", False):
        with rp_debugger.LineTimer("download_step"):
            audio_input = download_files_from_urls(job["id"], [job_input["audio"]])[0]

    if job_input.get("audio_base64", False):
        audio_input = base64_to_tempfile(job_input["audio_base64"])
        temp_paths_to_cleanup.append(audio_input)

    # ---- Optional trimming ----
    start_time = job_input.get("start_time")
    end_time = job_input.get("end_time")
    if start_time is not None or end_time is not None:
        trimmed_path = trim_audio(audio_input, start_time, end_time)
        # track for cleanup only if it's a new file
        if trimmed_path != audio_input:
            temp_paths_to_cleanup.append(trimmed_path)
            audio_input = trimmed_path

    # ---- Prediction ----
    with rp_debugger.LineTimer("prediction_step"):
        whisper_results = MODEL.predict(
            audio=audio_input,
            model_name=job_input["model"],
            transcription=job_input["transcription"],
            translation=job_input["translation"],
            translate=job_input["translate"],
            language=job_input["language"],
            temperature=job_input["temperature"],
            best_of=job_input["best_of"],
            beam_size=job_input["beam_size"],
            patience=job_input["patience"],
            length_penalty=job_input["length_penalty"],
            suppress_tokens=job_input.get("suppress_tokens", "-1"),
            initial_prompt=job_input["initial_prompt"],
            condition_on_previous_text=job_input["condition_on_previous_text"],
            temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
            compression_ratio_threshold=job_input["compression_ratio_threshold"],
            logprob_threshold=job_input["logprob_threshold"],
            no_speech_threshold=job_input["no_speech_threshold"],
            enable_vad=job_input["enable_vad"],
            word_timestamps=job_input["word_timestamps"],
        )

    # ---- Cleanup ----
    with rp_debugger.LineTimer("cleanup_step"):
        rp_cleanup.clean(["input_objects"])
        for p in temp_paths_to_cleanup:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    return whisper_results


runpod.serverless.start({"handler": run_whisper_job})
