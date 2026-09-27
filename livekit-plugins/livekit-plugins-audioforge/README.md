# LiveKit Agents plugin for audioforge

STT, VAD and a streaming turn detector for the self-hosted
[audioforge](https://github.com/maxmelichov/nvidia-audio-models) speech server: NVIDIA's streaming
FastConformer ASR with learned VAD and end-of-turn heads plus NVIDIA Streaming Sortformer diarization on one
80 ms clock, CPU-only. One server session per `AgentSession` feeds all three, so the user turn is committed
exactly where the server cut the transcript, on the diarizer's primary-speaker track.

## Installation

```bash
pip install livekit-plugins-audioforge
```

Start the server (once; the two NVIDIA models are CC-BY-4.0, ~1 GB download; see the server repository's
`research/NEMO_IMPORT.md` and `research/SORTFORMER_IMPORT.md`):

```bash
python -m audioforge.serve --asr runs/stage1_served.afm --diar runs/nemo_sortformer_v2.afm --port 8765
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import audioforge

fe = audioforge.AudioforgeFrontend("ws://127.0.0.1:8765")  # one server session per agent session
session = AgentSession(
    stt=fe.stt(),
    vad=fe.vad(),
    llm=...,
    tts=...,
    turn_handling={
        "turn_detection": "stt",  # the server's turn_end -> STT END_OF_SPEECH commits the turn
        "endpointing": {"min_delay": 0.0},  # the server already waited timeout_ms
        "interruption": {"mode": "vad"},  # barge-in from the server's VAD, no cloud model
    },
)
fe.attach(session)  # sends agent_end after every agent utterance (server --enroll after_agent_arm)
```

Options on `AudioforgeFrontend`: `turn_policy` (`"timeout"` default; `"hybrid_dyn"` / `"hybrid_silero"` for
fewer cut-ins at the price of more dead air), `timeout_ms`, `eot_threshold`, `final_source` (`"offline"` with a
server running `--final-asr tdt_v3` to get the offline per-turn transcript), `language`. With a server started
with `--lid`, detected language changes update the `language` of the transcripts that follow.

`fe.turn_detector()` returns an object for `turn_detection=` implementing LiveKit's streaming turn-detector
protocol (private, `_StreamingTurnDetector`) for users who want LiveKit's endpointing min/max delay logic;
`turn_detection="stt"` is the recommended and measured mode.

No hosted endpoint exists; the plugin author maintains the server. Tested with livekit-agents 1.8.3.
