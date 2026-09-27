# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VAD from the audioforge server's per-frame VAD probability (no local model).

Every server ``frame`` (one per 80 ms) becomes an ``INFERENCE_DONE`` event whose ``probability`` is
the VAD head's output and whose ``inference_duration`` is the arrival lag of that frame;
``START_OF_SPEECH`` fires after ``min_speech_duration`` above ``activation_threshold`` and
``END_OF_SPEECH`` after ``min_silence_duration`` below it, with the speech audio in ``frames``. It is
an any-speaker VAD: it does not know who is talking.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from audioforge_client import DEFAULT_URL, FrameEvent, Stats

from livekit import rtc
from livekit.agents import APIConnectionError, vad
from livekit.agents.utils import aio

from .frontend import FRAME_S, SAMPLE_RATE, AudioforgeFrontend, _Link
from .log import logger

WallOf = Any


class _VadMapper:
    """``frame`` events -> VADEvents (Silero-like hysteresis on the server's 80 ms probabilities)."""

    def __init__(
        self,
        activation_threshold: float = 0.5,
        min_speech_duration: float = 0.16,
        min_silence_duration: float = 0.56,
    ) -> None:
        self.thr = activation_threshold
        self.min_speech = min_speech_duration
        self.min_silence = min_silence_duration
        self.reset()

    def reset(self) -> None:
        self.speaking = False
        self.speech_run = 0.0
        self.silence_run = 0.0
        self.speech_dur = 0.0
        self.onset_t = 0.0

    def on_frame(
        self, ev: FrameEvent, now: float, wall_of: WallOf
    ) -> list[tuple[vad.VADEventType, dict[str, Any]]]:
        p, t = ev.vad, ev.t
        lag = max(0.0, now - wall_of(t))
        out: list[tuple[vad.VADEventType, dict[str, Any]]] = []
        if p >= self.thr:
            self.speech_run += FRAME_S
            self.silence_run = 0.0
        else:
            self.silence_run += FRAME_S
            self.speech_run = 0.0
        if self.speaking:
            self.speech_dur += FRAME_S
            if self.silence_run >= self.min_silence - 1e-9:
                out.append(
                    (
                        vad.VADEventType.END_OF_SPEECH,
                        {
                            "speech_duration": self.speech_dur - self.silence_run,
                            "silence_duration": self.silence_run,
                            "span": (self.onset_t, t - self.silence_run),
                        },
                    )
                )
                self.speaking, self.speech_dur = False, 0.0
        elif self.speech_run >= self.min_speech - 1e-9:
            self.speaking, self.speech_dur, self.onset_t = (
                True,
                self.speech_run,
                t - self.speech_run,
            )
            out.append(
                (
                    vad.VADEventType.START_OF_SPEECH,
                    {
                        "speech_duration": self.speech_dur,
                        "silence_duration": 0.0,
                        "span": (self.onset_t, t),
                    },
                )
            )
        out.append(
            (
                vad.VADEventType.INFERENCE_DONE,
                {
                    "probability": p,
                    "speaking": self.speaking,
                    "speech_duration": self.speech_dur if self.speaking else 0.0,
                    "silence_duration": self.silence_run,
                },
            )
        )
        return [
            (
                typ,
                {
                    **kw,
                    "t": t,
                    "lag": lag,
                    "raw_speech": self.speech_run,
                    "raw_silence": self.silence_run,
                },
            )
            for typ, kw in out
        ]


class VAD(vad.VAD):
    """Voice activity from the audioforge server's VAD head.

    Build it from an :class:`AudioforgeFrontend` (``fe.vad()``) to share the server session with
    the STT; a standalone ``VAD(url)`` opens its own session.
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        frontend: AudioforgeFrontend | None = None,
        activation_threshold: float = 0.5,
        min_speech_duration: float = 0.16,
        min_silence_duration: float = 0.56,
        keep_audio_s: float = 60.0,
        **frontend_kwargs: Any,
    ) -> None:
        """
        Args:
            url: The audioforge server (ignored when ``frontend`` is given).
            frontend: The shared session factory.
            activation_threshold: VAD probability above which a frame counts as speech.
            min_speech_duration: Seconds of speech before ``START_OF_SPEECH``.
            min_silence_duration: Seconds of silence before ``END_OF_SPEECH``.
            keep_audio_s: How much of the stream's own audio to keep for the events' ``frames``.
            **frontend_kwargs: :class:`AudioforgeOptions` fields for a standalone frontend.
        """
        super().__init__(capabilities=vad.VADCapabilities(update_interval=FRAME_S))
        self._fe = frontend or AudioforgeFrontend(url, **frontend_kwargs)
        self.activation_threshold = activation_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.keep_audio_s = keep_audio_s

    @property
    def frontend(self) -> AudioforgeFrontend:
        return self._fe

    @property
    def model(self) -> str:
        return "audioforge-vad-head"

    @property
    def provider(self) -> str:
        return "audioforge"

    def stream(self) -> VADStream:
        return VADStream(self)


class VADStream(vad.VADStream):
    def __init__(self, v: VAD) -> None:
        self._af = v
        self._mapper = _VadMapper(
            v.activation_threshold, v.min_speech_duration, v.min_silence_duration
        )
        self._audio: deque[tuple[float, rtc.AudioFrame]] = deque()
        self._pushed_s = 0.0
        super().__init__(v)

    async def _main_task(self) -> None:
        fe = self._af._fe
        attempts = 0
        while True:
            lk, q = await fe._acquire(self)
            fwd = asyncio.create_task(self._forward(lk), name="audioforge.vad.forward")
            try:
                while (item := await q.get()) is not None:
                    ev, now = item
                    if isinstance(ev, FrameEvent):
                        for typ, kw in self._mapper.on_frame(ev, now, lk.wall_of):
                            self._event_ch.send_nowait(self._event(typ, kw))
                    elif isinstance(ev, Stats):
                        return
            finally:
                await aio.cancel_and_wait(fwd)
                fe._release(self, lk, q)
            if self._input_ch.closed or lk.error is None or attempts >= 3:
                if lk.error is not None and not self._input_ch.closed:
                    raise APIConnectionError(f"audioforge session {lk.id} lost: {lk.error}")
                return
            attempts += 1
            logger.warning("audioforge VAD: session %s lost (%s); reconnecting", lk.id, lk.error)
            self._mapper.reset()
            await asyncio.sleep(0.5 * attempts)

    async def _forward(self, lk: _Link) -> None:
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                self._mapper.reset()  # hard segment boundary (VADStream.flush contract)
                continue
            lk.push(item, self)
            self._audio.append((self._pushed_s, item))
            self._pushed_s += item.duration
            while self._audio and self._audio[0][0] < self._pushed_s - self._af.keep_audio_s:
                self._audio.popleft()
        if lk.feeder is self or lk.feeder is None:
            lk.finish_audio()

    def _frames(self, t0: float, t1: float) -> list[rtc.AudioFrame]:
        return [f for s, f in self._audio if s + f.duration > t0 and s < t1]

    def _event(self, typ: vad.VADEventType, kw: dict[str, Any]) -> vad.VADEvent:
        frames = self._frames(*kw["span"]) if "span" in kw else []
        return vad.VADEvent(
            type=typ,
            samples_index=int(round(kw["t"] * SAMPLE_RATE)),
            timestamp=kw["t"],
            speech_duration=kw["speech_duration"],
            silence_duration=kw["silence_duration"],
            frames=frames,
            probability=kw.get("probability", 0.0),
            inference_duration=kw["lag"],
            speaking=kw.get("speaking", typ == vad.VADEventType.START_OF_SPEECH),
            raw_accumulated_silence=kw["raw_silence"],
            raw_accumulated_speech=kw["raw_speech"],
        )
