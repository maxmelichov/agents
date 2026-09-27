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

"""Streaming STT on the audioforge server.

Server events map to LiveKit speech events as follows: ``partial`` -> ``INTERIM_TRANSCRIPT``;
``final`` -> ``FINAL_TRANSCRIPT`` with ``speaker_id="S<k>"`` (the diarizer column of the primary
speaker at the turn end) and ``metadata["audioforge"]``; the ``turn_end`` of the cutting policy ->
``END_OF_SPEECH`` right after its final, so ``turn_detection="stt"`` commits the user turn exactly
where the server cut the transcript. ``START_OF_SPEECH`` fires on the first of ``min_speech_ms`` of
VAD >= 0.5 or a non-empty partial. A ``language`` event (server ``--lid``) changes the language of
the transcripts that follow. ``stats`` -> ``RECOGNITION_USAGE``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from audioforge_client import (
    DEFAULT_URL,
    Event,
    Final,
    FrameEvent,
    LanguageEvent,
    Partial,
    Stats,
    TurnEnd,
)

from livekit.agents import APIConnectionError, LanguageCode, stt
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, aio, merge_frames

from .frontend import FRAME_S, SAMPLE_RATE, AudioforgeFrontend, AudioforgeOptions, _Link

WallOf = Any
"""``Callable[[float], float]``: server audio time -> wall-clock capture time."""


class _SpeechMapper:
    """Server events -> LiveKit SpeechEvents. A pure state machine, tested without a socket.

    ``offset`` is ``RecognizeStream.start_time_offset``; SpeechData start/end times are
    ``offset + server audio time``.
    """

    def __init__(
        self, opts: AudioforgeOptions, min_speech_ms: int = 160, vad_threshold: float = 0.5
    ) -> None:
        self.opts = opts
        self.min_frames = max(1, int(round(min_speech_ms / (FRAME_S * 1000))))
        self.vad_thr = vad_threshold
        self.in_speech = False
        self.vad_run = 0
        self.seg_start_t = 0.0
        self.primary: int | None = None
        self.eot: float | None = None
        self.pending_turn_end: TurnEnd | None = None
        self.other_turn_end: TurnEnd | None = None
        self.turn_end_by_t: dict[float, TurnEnd] = {}
        self.last_interim = ""
        self.request_id = ""
        self.language = opts.language
        self.diar_lag_s = opts.diar_lag_s

    def _ev(self, typ: stt.SpeechEventType, **kw: Any) -> stt.SpeechEvent:
        return stt.SpeechEvent(type=typ, request_id=self.request_id, **kw)

    def _data(
        self,
        text: str,
        t_end: float,
        offset: float,
        meta: dict[str, Any] | None = None,
        speaker: int | None = None,
    ) -> stt.SpeechData:
        return stt.SpeechData(
            language=LanguageCode(self.language),
            text=text,
            start_time=offset + self.seg_start_t,
            end_time=offset + t_end,
            confidence=0.0,
            speaker_id=f"S{speaker}" if speaker is not None else None,
            is_primary_speaker=(speaker == self.primary) if speaker is not None else None,
            metadata={"audioforge": meta} if meta is not None else None,
        )

    def _start(self, onset_t: float, wall_of: WallOf) -> stt.SpeechEvent:
        self.in_speech = True
        self.seg_start_t = max(0.0, onset_t)
        return self._ev(
            stt.SpeechEventType.START_OF_SPEECH, speech_start_time=wall_of(self.seg_start_t)
        )

    def on_event(
        self, ev: Event, wall_of: WallOf, offset: float = 0.0, audio_s: float = 0.0
    ) -> list[stt.SpeechEvent]:
        out: list[stt.SpeechEvent] = []
        if isinstance(ev, FrameEvent):
            self.primary, self.eot = ev.primary, ev.eot
            self.vad_run = self.vad_run + 1 if ev.vad >= self.vad_thr else 0
            if not self.in_speech and self.vad_run >= self.min_frames:
                out.append(self._start(ev.t - self.vad_run * FRAME_S, wall_of))
        elif isinstance(ev, Partial):
            text = ev.text.strip()
            if text and not self.in_speech:
                out.append(self._start(ev.t - FRAME_S * 2, wall_of))
            if text and text != self.last_interim:
                self.last_interim = text
                out.append(
                    self._ev(
                        stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[self._data(text, ev.t, offset, speaker=self.primary)],
                    )
                )
        elif isinstance(ev, TurnEnd):
            if ev.policy == self.opts.cut_policy:
                self.pending_turn_end = ev
                self.turn_end_by_t[ev.t] = ev
            else:  # "both": the non-cutting policy's decision, reported in the next final's metadata
                self.other_turn_end = ev
        elif isinstance(ev, LanguageEvent):
            self.language = ev.language
        elif isinstance(ev, Final):
            if not self.opts.use_final(ev.source):
                return out
            te = self.pending_turn_end
            if ev.is_offline:  # an offline final: its own turn's decision
                te = self.turn_end_by_t.pop(ev.t, te)
            text, spk = ev.text.strip(), ev.speaker
            meta: dict[str, Any] = {
                "speaker": spk,
                "primary": self.primary,
                "t": ev.t,
                "turn_end": te.raw if te is not None else None,
                "eot": self.eot,
                "source": ev.source,
            }
            if self.other_turn_end is not None:
                meta["other_turn_end"], self.other_turn_end = self.other_turn_end.raw, None
            if text:
                if not self.in_speech:
                    out.append(self._start(ev.t - FRAME_S, wall_of))
                out.append(
                    self._ev(
                        stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[self._data(text, ev.t, offset, meta, spk)],
                    )
                )
            if self.in_speech:
                if te is not None:
                    frame_t = te.raw.get("frame_t")
                    last_frame = (
                        float(frame_t)
                        if isinstance(frame_t, (int, float))
                        else te.t - self.diar_lag_s
                    )
                    last_speech_t = last_frame - te.silence_ms / 1000.0
                else:
                    last_speech_t = ev.t
                out.append(
                    self._ev(
                        stt.SpeechEventType.END_OF_SPEECH,
                        speech_end_time=wall_of(max(self.seg_start_t, last_speech_t)),
                    )
                )
            self.in_speech, self.pending_turn_end, self.vad_run = False, None, 0
            self.last_interim = ""
        elif isinstance(ev, Stats):
            out.append(
                self._ev(
                    stt.SpeechEventType.RECOGNITION_USAGE,
                    recognition_usage=stt.RecognitionUsage(audio_duration=float(audio_s)),
                )
            )
        return out


class STT(stt.STT):
    """Streaming speech-to-text on a self-hosted audioforge server.

    Build it from an :class:`AudioforgeFrontend` (``fe.stt()``) to share the server session with
    the VAD and turn detector. A standalone ``STT(url)`` owns its own session.
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        frontend: AudioforgeFrontend | None = None,
        min_speech_ms: int = 160,
        **frontend_kwargs: Any,
    ) -> None:
        """
        Args:
            url: The audioforge server (ignored when ``frontend`` is given).
            frontend: The shared session factory.
            min_speech_ms: Consecutive server VAD >= 0.5 that opens ``START_OF_SPEECH`` (a partial
                opens it too).
            **frontend_kwargs: :class:`AudioforgeOptions` fields for a standalone frontend.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=True,
                aligned_transcript=False,
                offline_recognize=True,
            )
        )
        self._fe = frontend or AudioforgeFrontend(url, **frontend_kwargs)
        self._min_speech_ms = min_speech_ms

    @property
    def frontend(self) -> AudioforgeFrontend:
        return self._fe

    @property
    def model(self) -> str:
        lk = self._fe.link
        return lk.ready.model if lk is not None and lk.ready is not None else "audioforge"

    @property
    def provider(self) -> str:
        return "audioforge"

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(stt=self, conn_options=conn_options)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Batch recognition: a private server session, all audio at once, the finals joined."""
        frame = merge_frames(buffer) if isinstance(buffer, list) else buffer
        lk = _Link(self._fe.opts)
        await lk.open()
        q = lk.subscribe()
        lk.push(frame, self)
        lk.finish_audio()
        finals: list[str] = []
        spk: int | None = None
        while (item := await q.get()) is not None:
            ev, _ = item
            if isinstance(ev, Final) and ev.text.strip() and self._fe.opts.use_final(ev.source):
                finals.append(ev.text.strip())
                spk = ev.speaker if spk is None else spk
            if isinstance(ev, Stats):
                break
        await lk.aclose()
        if lk.error is not None and lk.stats is None:
            raise APIConnectionError(f"audioforge recognize failed: {lk.error}")
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=lk.id,
            alternatives=[
                stt.SpeechData(
                    language=LanguageCode(self._fe.opts.language),
                    text=" ".join(finals),
                    start_time=0.0,
                    end_time=lk.audio_s,
                    speaker_id=f"S{spk}" if spk is not None else None,
                )
            ],
        )


class SpeechStream(stt.RecognizeStream):
    def __init__(self, *, stt: STT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._af = stt

    async def _run(self) -> None:
        fe = self._af._fe
        lk, q = await fe._acquire(self)
        mapper = _SpeechMapper(fe.opts, self._af._min_speech_ms)
        mapper.request_id = lk.id
        mapper.diar_lag_s = lk.diar_lag_s
        fwd = asyncio.create_task(self._forward(lk), name="audioforge.stt.forward")
        try:
            while (item := await q.get()) is not None:
                ev, _ = item
                for out in mapper.on_event(ev, lk.wall_of, self.start_time_offset, lk.audio_s):
                    self._event_ch.send_nowait(out)
                if isinstance(ev, Stats):
                    break
            if lk.error is not None and lk.stats is None:
                raise APIConnectionError(
                    f"audioforge session {lk.id} lost: {lk.error}", retryable=True
                )
        finally:
            await aio.cancel_and_wait(fwd)
            fe._release(self, lk, q)

    async def _forward(self, lk: _Link) -> None:
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                continue  # no segment flush: turn ends are decided server-side
            lk.push(item, self)
        if lk.feeder is self or lk.feeder is None:
            lk.finish_audio()  # input ended -> server flushes, sends the last final + stats
