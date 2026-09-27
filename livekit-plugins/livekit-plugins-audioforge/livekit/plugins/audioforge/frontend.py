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

"""One audioforge server session shared by the STT, VAD and turn detector of an AgentSession.

The audioforge server (https://github.com/maxmelichov/nvidia-audio-models, ``audioforge/serve.py``)
takes int16 16 kHz PCM over a WebSocket and emits, on one 80 ms clock, ``frame`` events (VAD and
turn-head probabilities, diarizer speaker columns), ``partial`` transcripts, ``turn_end`` decisions
on the diarizer's primary-speaker track, and the turn's ``final`` transcript at each decision.
Protocol framing comes from the ``audioforge-client`` package.

:class:`AudioforgeFrontend` opens **one** server session (one ASR + one diarizer pass) when the
first of its streams starts and ends it when the last one closes. LiveKit pushes the same audio into
the STT, the VAD and the turn detector; the first stream that pushes audio feeds the server and the
others only read events. Speaker identities (diarizer columns) persist for the life of that session.
"""

from __future__ import annotations

import asyncio
import bisect
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from audioforge_client import (
    DEFAULT_URL,
    FINAL_SOURCES,
    POLICIES,
    Event,
    Ready,
    ServerError,
    Stats,
    config_message,
    cut_policy,
    parse_message,
    use_final,
)

from livekit import rtc
from livekit.agents import APIConnectionError
from livekit.agents.utils import aio

from .log import logger

if TYPE_CHECKING:
    from .stt import STT
    from .turn_detector import TurnDetector
    from .vad import VAD

SAMPLE_RATE = 16000
FRAME_S = 0.08
DIAR_LAG_MAX_S = 0.96
"""A diarizer column is final at most this long (plus compute) after its frame ends."""

_Item = tuple[Event, float]
"""What subscribers receive: the parsed server event and its arrival wall time."""


@dataclass
class AudioforgeOptions:
    """Session options: what the server is asked for and how its finals are read.

    Args:
        url: The audioforge server.
        turn_policy: Server turn policy. ``"timeout"`` (default, the measured product default):
            silence of the diarizer's primary speaker >= ``timeout_ms``; other speakers do not
            delay it. ``"hybrid_dyn"`` / ``"hybrid_silero"``: the turn head OR an any-speaker
            Silero silence (fewer cut-ins, more dead air). ``"hybrid"``, ``"head"``, ``"both"``,
            ``"timeout_quiet"`` as in the server.
        timeout_ms: Silence for the timeout policy.
        eot_threshold: Turn-head threshold for the head-based policies; None leaves the
            server's default for the policy.
        language: Language reported on transcripts until the server's ``--lid`` says otherwise.
        connect_timeout: Seconds to wait for the socket and the ``ready`` message.
        end_timeout: Seconds a closing session waits for the server's last final and stats.
        diar_lag_s: Fallback for the diarizer's column lag when the server's ``ready`` message
            does not report ``column_lag_ms``; used to place ``speech_end_time``.
        final_source: With a server running ``--final-asr``: ``"stream"`` (default) uses the
            streaming finals; ``"offline"`` holds each turn until the offline model's final of
            that turn arrives and uses it instead. Servers without the flag are unaffected.
    """

    url: str = DEFAULT_URL
    turn_policy: str = "timeout"
    timeout_ms: int = 1000
    eot_threshold: float | None = None
    language: str = "en"
    connect_timeout: float = 10.0
    end_timeout: float = 10.0
    diar_lag_s: float = 0.76
    final_source: str = "stream"

    def __post_init__(self) -> None:
        if self.turn_policy not in POLICIES:
            raise ValueError(f"turn_policy must be one of {POLICIES}, got {self.turn_policy!r}")
        if self.final_source not in FINAL_SOURCES:
            raise ValueError(
                f"final_source must be one of {FINAL_SOURCES}, got {self.final_source!r}"
            )
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

    def use_final(self, source: str | None) -> bool:
        """Whether a final with this ``source`` is part of the transcript."""
        return use_final({"source": source}, self.final_source)

    @property
    def cut_policy(self) -> str:
        """The policy whose ``turn_end`` cuts the finals (the server's rule)."""
        return cut_policy(self.turn_policy)


class _PcmConverter:
    """rtc.AudioFrame (any rate / channels, int16) -> int16 mono 16 kHz samples."""

    def __init__(self) -> None:
        self.rate: int | None = None
        self.resampler: rtc.AudioResampler | None = None

    def __call__(self, frame: rtc.AudioFrame) -> npt.NDArray[np.int16]:
        x = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            x = x.reshape(-1, frame.num_channels).mean(1).astype(np.int16)
        if self.rate is None:
            self.rate = frame.sample_rate
            if self.rate != SAMPLE_RATE:
                self.resampler = rtc.AudioResampler(
                    self.rate, SAMPLE_RATE, num_channels=1, quality=rtc.AudioResamplerQuality.HIGH
                )
        elif frame.sample_rate != self.rate:
            raise ValueError(f"sample rate changed mid-stream ({self.rate} -> {frame.sample_rate})")
        if self.resampler is None:
            return x
        out = self.resampler.push(rtc.AudioFrame(x.tobytes(), self.rate, 1, len(x)))
        return _cat(out)

    def flush(self) -> npt.NDArray[np.int16]:
        if self.resampler is None:
            return np.zeros(0, np.int16)
        return _cat(self.resampler.flush())


def _cat(frames: list[rtc.AudioFrame]) -> npt.NDArray[np.int16]:
    parts = [np.frombuffer(f.data, dtype=np.int16) for f in frames]
    return np.concatenate(parts) if parts else np.zeros(0, np.int16)


class _Link:
    """One WebSocket session: audio from a single feeder, events fanned out to subscribers.

    ``wall_of(t)`` maps the server's audio time ``t`` (seconds of audio received) to the wall-clock
    time at which that sample was pushed (its capture time, for live audio).
    """

    def __init__(self, opts: AudioforgeOptions) -> None:
        self.opts = opts
        self.id = uuid.uuid4().hex[:12]
        self.ws: Any = None
        self.ready: Ready | None = None
        self.stats: Stats | None = None
        self.subs: list[asyncio.Queue[_Item | None]] = []
        self.feeder: object | None = None
        self.samples = 0
        self.diar_lag_s = opts.diar_lag_s
        self.error: BaseException | None = None
        self.errors: list[ServerError] = []
        self.end_sent = False
        self.pending_controls: list[str] = []
        self.controls_sent: list[tuple[str, float]] = []
        self.closed = asyncio.Event()
        self._cum: list[int] = []
        self._wall: list[float] = []
        self._conv = _PcmConverter()
        self._outq: asyncio.Queue[bytes | str] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []

    async def open(self) -> None:
        from websockets.asyncio.client import connect

        timeout = self.opts.connect_timeout
        try:
            self.ws = await asyncio.wait_for(
                connect(self.opts.url, max_size=2**22, ping_interval=None, open_timeout=timeout),
                timeout,
            )
            first = await asyncio.wait_for(self.ws.recv(), timeout)
        except Exception as e:
            if self.ws is not None:
                await self.ws.close()
            raise APIConnectionError(
                f"audioforge server {self.opts.url} unreachable: {type(e).__name__}: {e}"
            ) from e
        events = parse_message(first)
        if not events or not isinstance(events[0], Ready):
            await self.ws.close()
            raise APIConnectionError(
                f"audioforge server {self.opts.url}: expected a ready message, got {first!r}"
            )
        self.ready = events[0]
        if self.ready.column_lag_ms is not None:
            self.diar_lag_s = self.ready.column_lag_ms / 1000.0
        cfg = config_message(
            turn_policy=self.opts.turn_policy,
            timeout_ms=self.opts.timeout_ms,
            eot_threshold=self.opts.eot_threshold,
            sample_rate=SAMPLE_RATE,
        )
        await self.ws.send(json.dumps(cfg))
        for kind in self.pending_controls:
            await self.ws.send(json.dumps({"type": kind}))
            self.controls_sent.append((kind, 0.0))
        self.pending_controls = []
        self._tasks = [
            asyncio.create_task(self._reader(), name="audioforge.reader"),
            asyncio.create_task(self._writer(), name="audioforge.writer"),
        ]

    # ------------------------------------------------------------ subscribers
    def subscribe(self) -> asyncio.Queue[_Item | None]:
        q: asyncio.Queue[_Item | None] = asyncio.Queue()
        if self.closed.is_set():
            q.put_nowait(None)
        self.subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[_Item | None]) -> None:
        if q in self.subs:
            self.subs.remove(q)

    # ------------------------------------------------------------ audio in
    def push(self, frame: rtc.AudioFrame, src: object) -> bool:
        """Send ``frame`` if ``src`` is (or becomes) the feeder. Returns whether it was sent."""
        if self.end_sent or self.closed.is_set():
            return False
        if self.feeder is None:
            self.feeder = src
        if src is not self.feeder:
            return False
        self._send_pcm(self._conv(frame))
        return True

    def _send_pcm(self, pcm: npt.NDArray[np.int16]) -> None:
        if not len(pcm):
            return
        self.samples += len(pcm)
        self._cum.append(self.samples)
        self._wall.append(time.time())
        if len(self._cum) > 60000:
            del self._cum[:10000], self._wall[:10000]
        self._outq.put_nowait(pcm.astype("<i2").tobytes())

    def send_control(self, kind: str) -> bool:
        """Queue ``agent_end`` / ``enroll`` behind the audio already pushed, so the server applies
        it at exactly this audio position. Returns whether it was queued."""
        if self.end_sent or self.closed.is_set():
            return False
        self._outq.put_nowait(json.dumps({"type": kind}))
        self.controls_sent.append((kind, self.audio_s))
        return True

    def finish_audio(self) -> None:
        """Send ``{"type": "end"}``: the server flushes, emits the last final + stats and closes."""
        if self.end_sent or self.closed.is_set():
            return
        self._send_pcm(self._conv.flush())
        self.end_sent = True
        self._outq.put_nowait(json.dumps({"type": "end"}))

    def wall_of(self, t: float) -> float:
        k = t * SAMPLE_RATE
        if not self._cum:
            return time.time()
        i = bisect.bisect_left(self._cum, k)
        if i >= len(self._cum):
            return self._wall[-1] + (k - self._cum[-1]) / SAMPLE_RATE
        return self._wall[i] - (self._cum[i] - k) / SAMPLE_RATE

    @property
    def audio_s(self) -> float:
        return self.samples / SAMPLE_RATE

    # ------------------------------------------------------------ tasks
    async def _reader(self) -> None:
        try:
            async for raw in self.ws:
                if not isinstance(raw, str):
                    continue
                now = time.time()
                for ev in parse_message(raw):
                    if isinstance(ev, Stats):
                        self.stats = ev
                    elif isinstance(ev, ServerError):
                        self.errors.append(ev)
                        if ev.fatal:
                            logger.error(
                                "audioforge session %s: fatal server error %s", self.id, ev.message
                            )
                            if self.error is None:
                                self.error = APIConnectionError(
                                    f"audioforge server error {ev.message}", retryable=False
                                )
                        else:
                            logger.warning("audioforge session %s: server %s", self.id, ev.message)
                    for q in list(self.subs):
                        q.put_nowait((ev, now))
        except Exception as e:  # ConnectionClosedError etc.
            if not (self.end_sent and self.stats is not None):
                self.error = e
        finally:
            if self.stats is None and self.error is None and not self.end_sent:
                self.error = APIConnectionError("audioforge server closed the session")
            self.closed.set()
            for q in list(self.subs):
                q.put_nowait(None)

    async def _writer(self) -> None:
        try:
            while True:
                item = await self._outq.get()
                await self.ws.send(item)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self.error is None and not self.closed.is_set():
                self.error = e

    async def end(self, wait: float | None = None) -> Stats | None:
        """Finish the audio, wait for the server's stats (or ``wait`` s), close."""
        self.finish_audio()
        try:
            await asyncio.wait_for(
                self.closed.wait(), self.opts.end_timeout if wait is None else wait
            )
        except asyncio.TimeoutError:
            logger.warning("audioforge session %s: no stats after end; closing", self.id)
        await self.aclose()
        return self.stats

    async def aclose(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        await aio.cancel_and_wait(*self._tasks)
        self.closed.set()


class AudioforgeFrontend:
    """Factory for the STT, VAD and turn detector that share one audioforge server session.

    The session opens when the first stream starts and ends (``{"type": "end"}``) when the last one
    closes. Standalone ``STT(url)`` / ``VAD(url)`` objects create their own frontend, and so their
    own server session each: build them from one frontend to share it.
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        turn_policy: str = "timeout",
        timeout_ms: int = 1000,
        eot_threshold: float | None = None,
        language: str = "en",
        connect_timeout: float = 10.0,
        end_timeout: float = 10.0,
        final_source: str = "stream",
    ) -> None:
        """See :class:`AudioforgeOptions` for the arguments."""
        self.opts = AudioforgeOptions(
            url=url,
            turn_policy=turn_policy,
            timeout_ms=timeout_ms,
            eot_threshold=eot_threshold,
            language=language,
            connect_timeout=connect_timeout,
            end_timeout=end_timeout,
            final_source=final_source,
        )
        self._link: _Link | None = None
        self._owners: set[object] = set()
        self._lock: asyncio.Lock | None = None
        self._bg: set[asyncio.Task[None]] = set()
        self._pending_controls: list[str] = []
        self.sessions_opened = 0
        self.last_stats: Stats | None = None
        self.controls: list[tuple[str, float]] = []
        """Every ``agent_end`` / ``enroll`` sent: (kind, seconds of audio sent before it)."""

    # factories
    def stt(self, **kwargs: Any) -> STT:
        """An :class:`~livekit.plugins.audioforge.STT` on this session."""
        from .stt import STT

        return STT(frontend=self, **kwargs)

    def vad(self, **kwargs: Any) -> VAD:
        """A :class:`~livekit.plugins.audioforge.VAD` on this session."""
        from .vad import VAD

        return VAD(frontend=self, **kwargs)

    def turn_detector(self, **kwargs: Any) -> TurnDetector:
        """A :class:`~livekit.plugins.audioforge.TurnDetector` on this session."""
        from .turn_detector import TurnDetector

        return TurnDetector(frontend=self, **kwargs)

    @property
    def link(self) -> _Link | None:
        return self._link

    # server --enroll triggers
    def agent_end(self) -> bool:
        """The agent's speech ended: send ``agent_end`` (server ``--enroll after_agent_arm`` or
        ``after_agent``), queued behind the audio already sent so the server arms at this audio
        position. Before the session exists it is sent right after the session's config.
        Synchronous, safe from event callbacks."""
        return self._control("agent_end")

    def enroll(self) -> bool:
        """Server ``--enroll explicit``: enroll the primary speaker on the next utterance."""
        return self._control("enroll")

    def _control(self, kind: str) -> bool:
        lk = self._link
        if lk is not None and not lk.closed.is_set() and not lk.end_sent and lk.ws is not None:
            ok = lk.send_control(kind)
            if ok:
                self.controls.append((kind, lk.audio_s))
            return ok
        self._pending_controls.append(kind)
        self.controls.append((kind, 0.0))
        return True

    def attach(self, session: Any) -> Callable[[], None]:
        """Send ``agent_end`` whenever the AgentSession's agent stops speaking.

        Hooks ``agent_state_changed`` (``speaking`` -> anything else): the product trigger for
        the server's ``--enroll after_agent_arm``. Returns a function that removes the hook.
        """

        def _on_agent_state(ev: Any) -> None:
            if (
                getattr(ev, "old_state", None) == "speaking"
                and getattr(ev, "new_state", None) != "speaking"
            ):
                self.agent_end()

        session.on("agent_state_changed", _on_agent_state)

        def _detach() -> None:
            session.off("agent_state_changed", _on_agent_state)

        return _detach

    async def _acquire(self, owner: object) -> tuple[_Link, asyncio.Queue[_Item | None]]:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            lk = self._link
            if lk is None or lk.closed.is_set() or lk.end_sent:
                lk = _Link(self.opts)
                lk.pending_controls, self._pending_controls = self._pending_controls, []
                await lk.open()
                self._link, self._owners = lk, set()
                self.sessions_opened += 1
                logger.info("audioforge session %s opened (%s)", lk.id, lk.ready)
            self._owners.add(owner)
            return lk, lk.subscribe()

    def _release(self, owner: object, lk: _Link, q: asyncio.Queue[_Item | None] | None) -> None:
        """Detach ``owner``; when it was the last one, end the session in the background."""
        if q is not None:
            lk.unsubscribe(q)
        if self._link is lk:
            self._owners.discard(owner)
            if self._owners:
                return
            self._link = None
        t = asyncio.get_running_loop().create_task(self._end(lk))
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def _end(self, lk: _Link) -> None:
        st = await lk.end()
        if st is not None:
            self.last_stats = st

    async def aclose(self) -> None:
        if self._link is not None:
            lk, self._link = self._link, None
            await self._end(lk)
        if self._bg:
            await asyncio.gather(*self._bg, return_exceptions=True)


__all__ = ["AudioforgeFrontend", "AudioforgeOptions", "SAMPLE_RATE", "FRAME_S"]
