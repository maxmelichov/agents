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

"""Streaming turn detector backed by the audioforge server's turn decisions.

``turn_detection="stt"`` with :class:`~livekit.plugins.audioforge.STT` is the recommended mode: the
server's ``turn_end`` becomes ``END_OF_SPEECH`` and LiveKit commits the turn there. This detector is
for users who want LiveKit's endpointing ``min_delay`` / ``max_delay`` logic instead: LiveKit ends
turns on the VAD's ``END_OF_SPEECH`` and asks the detector for an end-of-turn probability at each
pause. Policy ``"timeout"`` (default) resolves p = 1.0 when the server's timeout ``turn_end``
arrives for the current pause and 0.0 on resumed speech or after ``prediction_timeout``
(-> LiveKit waits ``max_delay``); policy ``"head"`` resolves the turn head's probability.

It implements ``livekit.agents.voice.turn._StreamingTurnDetector``, a private protocol that may
change between LiveKit releases.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from audioforge_client import DEFAULT_EOT_THRESHOLD, DEFAULT_URL, FrameEvent, Stats, TurnEnd

from livekit import rtc
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import aio
from livekit.agents.voice.turn import TurnDetectionEvent

from .frontend import DIAR_LAG_MAX_S, AudioforgeFrontend, _Item, _Link
from .log import logger


class TurnDetector:
    """``turn_detection=`` object for ``AgentSession`` (see the module docstring)."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        frontend: AudioforgeFrontend | None = None,
        policy: str = "timeout",
        unlikely_threshold: float | None = None,
        prediction_timeout: float | None = None,
        **frontend_kwargs: Any,
    ) -> None:
        """
        Args:
            url: The audioforge server (ignored when ``frontend`` is given).
            frontend: The shared session factory.
            policy: ``"timeout"`` (the server's timeout ``turn_end`` decides) or ``"head"`` (the
                turn head's probability at the pause).
            unlikely_threshold: Probability below which LiveKit waits ``max_delay``; defaults to
                0.5 for ``timeout`` and the server's head threshold for ``head``.
            prediction_timeout: Seconds after which an unanswered request resolves to 0.0; defaults
                to ``timeout_ms`` plus the diarizer's maximum lag plus compute headroom.
            **frontend_kwargs: :class:`AudioforgeOptions` fields for a standalone frontend.
        """
        if policy not in ("timeout", "head"):
            raise ValueError("policy must be 'timeout' or 'head'")
        self._fe = frontend or AudioforgeFrontend(url, **frontend_kwargs)
        self.policy = policy
        if policy == "timeout" and self._fe.opts.turn_policy == "head":
            raise ValueError(
                "policy 'timeout' needs the server's timeout turn_end: frontend turn_policy "
                "'timeout' or 'both'"
            )
        if unlikely_threshold is not None:
            self._unlikely = unlikely_threshold
        elif policy == "timeout":
            self._unlikely = 0.5
        else:
            self._unlikely = self._fe.opts.eot_threshold or DEFAULT_EOT_THRESHOLD
        self._pred_timeout = (
            prediction_timeout
            if prediction_timeout is not None
            else self._fe.opts.timeout_ms / 1000 + DIAR_LAG_MAX_S + 0.8
        )
        self.predictions: list[dict[str, Any]] = []
        """Every settled request of every stream: ``{"p", "why", "wall"}``."""

    @property
    def frontend(self) -> AudioforgeFrontend:
        return self._fe

    @property
    def model(self) -> str:
        return f"audioforge-{self.policy}"

    @property
    def provider(self) -> str:
        return "audioforge"

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> TurnDetectorStream:
        return TurnDetectorStream(self)


class TurnDetectorStream:
    """``_StreamingTurnDetectorStream`` over the shared server session."""

    def __init__(self, det: TurnDetector) -> None:
        self._det = det
        self._fut: asyncio.Future[TurnDetectionEvent] | None = None
        self._deadline: asyncio.TimerHandle | None = None
        self._armed_t: float | None = None
        self._last_voiced_t = -1.0
        self._pred_t = 0.0
        self._link: _Link | None = None
        self._q: asyncio.Queue[_Item | None] | None = None
        self._early: list[rtc.AudioFrame] = []
        self._ended = False
        self.predictions: list[dict[str, Any]] = []
        self._task = asyncio.create_task(self._run(), name="audioforge.turn")

    # protocol properties
    @property
    def model(self) -> str:
        return self._det.model

    @property
    def provider(self) -> str:
        return "audioforge"

    @property
    def is_fallback(self) -> bool:
        return False

    @property
    def prediction_timeout(self) -> float:
        return self._det._pred_timeout

    @property
    def policy(self) -> str:
        return self._det.policy

    async def unlikely_threshold(self, language: str | None) -> float | None:
        return self._det._unlikely

    async def backchannel_threshold(self, language: str | None) -> float | None:
        return None

    async def supports_language(self, language: str | None) -> bool:
        return True  # an acoustic decision (the ASR itself is English-only)

    # requests
    def predict(self) -> asyncio.Future[TurnDetectionEvent]:
        loop = asyncio.get_running_loop()
        self._settle(0.0, "superseded")
        fut: asyncio.Future[TurnDetectionEvent] = loop.create_future()
        self._fut = fut
        self._pred_t = self._link.audio_s if self._link is not None else 0.0
        if self._ended:
            self._settle(1.0, "closed")
        elif (
            self.policy == "timeout"
            and self._armed_t is not None
            and self._last_voiced_t <= self._armed_t
        ):
            self._settle(1.0, "turn_end_before_request", detection_delay=0.0)
        else:
            self._deadline = loop.call_later(
                max(0.05, self.prediction_timeout - 0.05), self._settle, 0.0, "deadline"
            )
        return fut

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        self._armed_t = None
        self._settle(0.0, "cancelled")

    def flush(self, reason: str | None = None) -> None:
        self._armed_t = None
        self._settle(0.0, f"flush:{reason}")

    def _settle(self, p: float, why: str, detection_delay: float | None = None) -> None:
        if self._deadline is not None:
            self._deadline.cancel()
            self._deadline = None
        fut, self._fut = self._fut, None
        if fut is not None and not fut.done():
            fut.set_result(
                TurnDetectionEvent(
                    type="eot_prediction",
                    end_of_turn_probability=float(p),
                    last_speaking_time=time.time(),
                    detection_delay=detection_delay,
                )
            )
            rec = {"p": float(p), "why": why, "wall": time.time()}
            self.predictions.append(rec)
            self._det.predictions.append(rec)
            logger.debug("audioforge turn prediction p=%.3f (%s)", p, why)

    # audio
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._link is not None:
            self._link.push(frame, self)
        elif not self._ended:
            self._early.append(frame)
            if len(self._early) > 500:
                self._early.pop(0)

    def end_input(self) -> None:
        if self._ended:
            return
        self._ended = True
        if self._link is not None and self._link.feeder is self:
            self._link.finish_audio()
        self._settle(1.0, "closed")

    async def aclose(self) -> None:
        self.end_input()
        await aio.cancel_and_wait(self._task)

    async def _run(self) -> None:
        fe = self._det._fe
        try:
            lk, q = await fe._acquire(self)
        except Exception as e:
            logger.error(
                "audioforge turn detector: no server session (%s); predictions fall back to the "
                "deadline (p=0 -> max_delay)",
                e,
            )
            return
        self._link, self._q = lk, q
        for f in self._early:
            lk.push(f, self)
        self._early.clear()
        try:
            while (item := await q.get()) is not None:
                ev, now = item
                if isinstance(ev, FrameEvent):
                    if ev.vad >= 0.5:
                        self._last_voiced_t = max(self._last_voiced_t, ev.t)
                    if self.policy == "head" and self._fut is not None and ev.t >= self._pred_t:
                        self._settle(ev.eot or 0.0, "head", detection_delay=now - lk.wall_of(ev.t))
                elif isinstance(ev, TurnEnd) and ev.policy == self.policy == "timeout":
                    if self._last_voiced_t > ev.t:
                        continue  # speech after the decision: it is about an earlier pause
                    if self._fut is not None:
                        self._settle(1.0, "turn_end", detection_delay=now - lk.wall_of(ev.t))
                    else:
                        self._armed_t = ev.t
                elif isinstance(ev, Stats):
                    break
        finally:
            fe._release(self, lk, q)
