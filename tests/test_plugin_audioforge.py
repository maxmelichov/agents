"""Tests for the audioforge plugin, hermetic against ``audioforge_client.testing.FakeAudioforgeServer``.

Covers:
- option validation and the shared-session frontend (one server session for STT + VAD + turn detector,
  ``agent_end`` ordering, ``attach`` on an AgentSession-like emitter);
- the STT event mapping (START_OF_SPEECH, INTERIM / FINAL_TRANSCRIPT with speaker ids, END_OF_SPEECH
  at the server's turn end, RECOGNITION_USAGE, ``final_source="offline"``, ``--lid`` language events);
- the VAD event mapping (INFERENCE_DONE per frame, START / END_OF_SPEECH hysteresis);
- the streaming turn detector (p = 1.0 on the server's turn_end for the current pause, deadline, stale
  decisions after resumed speech, head policy).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from audioforge_client import FrameEvent, TurnEnd
from audioforge_client.testing import FakeAudioforgeServer, speech_pcm

from livekit import rtc
from livekit.agents import vad as agents_vad
from livekit.agents.stt import SpeechEventType
from livekit.plugins import audioforge
from livekit.plugins.audioforge.stt import _SpeechMapper
from livekit.plugins.audioforge.vad import _VadMapper

pytestmark = pytest.mark.plugin("audioforge")

SR = 16000


def _frames(pcm: bytes, chunk_ms: int = 20) -> list[rtc.AudioFrame]:
    n = SR * chunk_ms // 1000
    out = []
    for i in range(0, len(pcm), n * 2):
        piece = pcm[i : i + n * 2]
        if len(piece) < n * 2:
            piece = piece + bytes(n * 2 - len(piece))
        out.append(rtc.AudioFrame(piece, SR, 1, n))
    return out


async def _push_realtime(streams: list, frames: list[rtc.AudioFrame], speed: float = 0.0) -> None:
    for f in frames:
        for s in streams:
            s.push_frame(f) if hasattr(s, "push_frame") else s.push_audio(f)
        await asyncio.sleep(f.duration * speed)


# ---------------------------------------------------------------------------
# options / frontend
# ---------------------------------------------------------------------------


def test_options_validation():
    with pytest.raises(ValueError):
        audioforge.AudioforgeFrontend(turn_policy="soon")
    with pytest.raises(ValueError):
        audioforge.AudioforgeFrontend(final_source="both")
    with pytest.raises(ValueError):
        audioforge.AudioforgeFrontend(timeout_ms=0)
    fe = audioforge.AudioforgeFrontend(turn_policy="hybrid_dyn")
    assert fe.opts.cut_policy == "hybrid_dyn"
    assert audioforge.AudioforgeFrontend(turn_policy="both").opts.cut_policy == "timeout"
    with pytest.raises(ValueError):
        audioforge.TurnDetector(frontend=audioforge.AudioforgeFrontend(turn_policy="head"))
    with pytest.raises(ValueError):
        audioforge.TurnDetector(policy="maybe")


def test_standalone_objects_have_their_own_frontend():
    s = audioforge.STT("ws://127.0.0.1:1", turn_policy="hybrid_dyn")
    assert s.frontend.opts.turn_policy == "hybrid_dyn"
    assert s.provider == "audioforge" and s.model == "audioforge"
    assert s.capabilities.streaming and s.capabilities.diarization
    v = audioforge.VAD("ws://127.0.0.1:1")
    assert v.capabilities.update_interval == pytest.approx(0.08)
    assert v.frontend is not s.frontend


def test_attach_sends_agent_end_when_the_agent_stops_speaking():
    class _Session:
        def __init__(self):
            self.handlers = {}

        def on(self, name, fn):
            self.handlers[name] = fn

        def off(self, name, fn):
            self.handlers.pop(name, None)

    class _Ev:
        def __init__(self, old, new):
            self.old_state, self.new_state = old, new

    fe = audioforge.AudioforgeFrontend()
    session = _Session()
    detach = fe.attach(session)
    session.handlers["agent_state_changed"](_Ev("listening", "speaking"))
    assert fe.controls == []
    session.handlers["agent_state_changed"](_Ev("speaking", "listening"))
    assert fe.controls == [("agent_end", 0.0)]
    detach()
    assert "agent_state_changed" not in session.handlers


async def test_one_session_for_stt_vad_and_turn_detector():
    async with FakeAudioforgeServer(timeout_ms=480, enroll="after_agent_arm") as srv:
        fe = audioforge.AudioforgeFrontend(srv.url, timeout_ms=480)
        assert fe.agent_end()  # before the session exists: sent right after the config
        stt_stream = fe.stt().stream()
        vad_stream = fe.vad().stream()
        td_stream = fe.turn_detector().stream()
        frames = _frames(speech_pcm(1.0, 1.0))
        await _push_realtime([stt_stream, vad_stream, td_stream], frames)
        stt_stream.end_input()
        vad_stream.end_input()
        td_stream.end_input()

        stt_events = [ev async for ev in stt_stream]
        vad_events = [ev async for ev in vad_stream]
        await stt_stream.aclose()
        await vad_stream.aclose()
        await td_stream.aclose()
        await fe.aclose()

    assert srv.connections == 1 and fe.sessions_opened == 1
    assert srv.configs[0]["turn_policy"] == "timeout" and srv.configs[0]["timeout_ms"] == 480
    assert srv.controls[0][1:] == ("agent_end", 0)
    assert srv.audio_bytes[0] == len(speech_pcm(1.0, 1.0))

    kinds = [e.type for e in stt_events]
    assert kinds[0] == SpeechEventType.START_OF_SPEECH
    assert SpeechEventType.INTERIM_TRANSCRIPT in kinds
    finals = [e for e in stt_events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
    assert len(finals) == 1
    data = finals[0].alternatives[0]
    assert data.text.startswith("w0") and data.speaker_id == "S0" and data.is_primary_speaker
    assert (
        data.metadata is not None and data.metadata["audioforge"]["turn_end"]["policy"] == "timeout"
    )
    assert (
        kinds.index(SpeechEventType.END_OF_SPEECH)
        == kinds.index(SpeechEventType.FINAL_TRANSCRIPT) + 1
    )
    assert kinds[-1] == SpeechEventType.RECOGNITION_USAGE
    assert stt_events[-1].recognition_usage is not None
    assert stt_events[-1].recognition_usage.audio_duration == pytest.approx(2.0)

    vad_kinds = [e.type for e in vad_events]
    assert vad_kinds.count(agents_vad.VADEventType.INFERENCE_DONE) == 25
    assert agents_vad.VADEventType.START_OF_SPEECH in vad_kinds
    assert agents_vad.VADEventType.END_OF_SPEECH in vad_kinds
    end = next(e for e in vad_events if e.type == agents_vad.VADEventType.END_OF_SPEECH)
    assert end.frames and end.speech_duration == pytest.approx(1.0, abs=0.2)


async def test_stt_stream_reports_a_lost_session():
    from livekit.agents import APIConnectionError
    from livekit.agents.types import APIConnectOptions

    async with FakeAudioforgeServer(timeout_ms=480, drop_after_s=0.5) as srv:
        fe = audioforge.AudioforgeFrontend(srv.url)
        # LiveKit retries a retryable stream error with a fresh session by default; disable that
        # to observe the error itself.
        stream = fe.stt().stream(conn_options=APIConnectOptions(max_retry=0))
        with pytest.raises(APIConnectionError):
            try:
                await _push_realtime([stream], _frames(speech_pcm(1.0, 1.0)), speed=0.2)
                stream.end_input()
            except RuntimeError:
                pass  # the stream closed under us once the session was lost
            async for _ in stream:
                pass
        await stream.aclose()
        await fe.aclose()


async def test_offline_final_source_and_language_events():
    async with FakeAudioforgeServer(timeout_ms=480, final_asr=True, language="he") as srv:
        fe = audioforge.AudioforgeFrontend(srv.url, timeout_ms=480, final_source="offline")
        stream = fe.stt().stream()
        await _push_realtime([stream], _frames(speech_pcm(1.0, 1.0)))
        stream.end_input()
        events = [ev async for ev in stream]
        await stream.aclose()
        await fe.aclose()
    finals = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
    assert len(finals) == 1
    d = finals[0].alternatives[0]
    assert d.metadata is not None and d.metadata["audioforge"]["source"] == "tdt_v3"
    assert d.text == d.text.upper()
    assert str(d.language) == "he"


async def test_recognize_batch():
    async with FakeAudioforgeServer(timeout_ms=480) as srv:
        s = audioforge.STT(srv.url, timeout_ms=480)
        ev = await s.recognize(_frames(speech_pcm(1.0, 0.2)))
    assert ev.type == SpeechEventType.FINAL_TRANSCRIPT
    assert ev.alternatives[0].text.startswith("w0") and ev.alternatives[0].speaker_id == "S0"


# ---------------------------------------------------------------------------
# mappers (no socket)
# ---------------------------------------------------------------------------


def _fr(t, vad_p, primary=0, eot=None):
    return FrameEvent(t=t, vad=vad_p, eot=eot, speakers=(vad_p, 0, 0, 0), primary=primary)


def test_speech_mapper_end_of_speech_time_uses_the_decision_and_lag():
    opts = audioforge.AudioforgeOptions(timeout_ms=1000)
    m = _SpeechMapper(opts)
    m.diar_lag_s = 0.16
    wall = {}
    wall_of = lambda t: 1000.0 + t  # noqa: E731
    out = []
    for i in range(1, 5):
        out += m.on_event(_fr(0.08 * i, 0.9), wall_of)
    assert out[0].type == SpeechEventType.START_OF_SPEECH and out[0].speech_start_time == 1000.0
    te = TurnEnd(t=2.16, policy="timeout", p=None, silence_ms=1000)
    out += m.on_event(te, wall_of)
    from audioforge_client import Final

    out += m.on_event(Final(t=2.16, text="hello", speaker=0), wall_of, offset=5.0)
    final, end = out[-2], out[-1]
    assert final.type == SpeechEventType.FINAL_TRANSCRIPT
    assert final.alternatives[0].start_time == pytest.approx(5.0) and final.alternatives[
        0
    ].end_time == pytest.approx(7.16)
    # last speech = decision - lag - silence = 2.16 - 0.16 - 1.0 = 1.0
    assert end.type == SpeechEventType.END_OF_SPEECH and end.speech_end_time == pytest.approx(
        1001.0
    )
    assert m.in_speech is False
    del wall


def test_speech_mapper_ignores_non_cutting_turn_ends():
    m = _SpeechMapper(audioforge.AudioforgeOptions(turn_policy="both"))
    m.on_event(TurnEnd(t=1.0, policy="head", p=0.99, silence_ms=200), lambda t: t)
    assert m.pending_turn_end is None and m.other_turn_end is not None
    m.on_event(TurnEnd(t=1.5, policy="timeout", p=None, silence_ms=1000), lambda t: t)
    assert m.pending_turn_end is not None


def test_vad_mapper_hysteresis():
    m = _VadMapper(min_speech_duration=0.16, min_silence_duration=0.24)
    kinds = []
    for i in range(1, 11):
        p = 0.9 if i <= 5 else 0.05
        kinds += [k for k, _ in m.on_frame(_fr(0.08 * i, p), now=0.08 * i, wall_of=lambda t: t)]
    assert kinds.count(agents_vad.VADEventType.INFERENCE_DONE) == 10
    # START on the 2nd speech frame, before that frame's INFERENCE_DONE.
    assert kinds.index(agents_vad.VADEventType.START_OF_SPEECH) == 1
    # END on the 3rd silent frame (frame 8), before its INFERENCE_DONE: 1 + 2 + 5 events precede it.
    assert kinds.index(agents_vad.VADEventType.END_OF_SPEECH) == 8


# ---------------------------------------------------------------------------
# turn detector
# ---------------------------------------------------------------------------


async def test_turn_detector_resolves_on_the_servers_turn_end():
    async with FakeAudioforgeServer(timeout_ms=480) as srv:
        fe = audioforge.AudioforgeFrontend(srv.url, timeout_ms=480)
        det = fe.turn_detector()
        stream = det.stream()
        assert stream.model == "audioforge-timeout" and not stream.is_fallback
        assert await stream.unlikely_threshold("en") == 0.5
        speech = _frames(speech_pcm(1.0, 0.0))
        await _push_realtime([stream], speech)
        await asyncio.sleep(0.05)
        fut = stream.predict()  # the user paused: LiveKit asks
        await _push_realtime([stream], _frames(speech_pcm(0.0, 1.0)))
        ev = await asyncio.wait_for(fut, 5)
        assert ev.end_of_turn_probability == 1.0
        assert det.predictions[-1]["why"] == "turn_end"
        # A prediction with no decision within the deadline resolves to 0.
        stream._det._pred_timeout = 0.2
        fut2 = stream.predict()
        ev2 = await asyncio.wait_for(fut2, 5)
        assert ev2.end_of_turn_probability == 0.0 and det.predictions[-1]["why"] == "deadline"
        stream.end_input()
        await stream.aclose()
        await fe.aclose()
    assert srv.connections == 1


async def test_turn_detector_arms_and_stales():
    async with FakeAudioforgeServer(timeout_ms=480) as srv:
        fe = audioforge.AudioforgeFrontend(srv.url, timeout_ms=480)
        stream = fe.turn_detector().stream()
        # turn_end arrives with no request pending -> armed; a later request resolves at once
        await _push_realtime([stream], _frames(speech_pcm(1.0, 1.0)))
        for _ in range(50):
            if stream._armed_t is not None:
                break
            await asyncio.sleep(0.05)
        assert stream._armed_t is not None
        ev = await asyncio.wait_for(stream.predict(), 1)
        assert ev.end_of_turn_probability == 1.0
        # speech after a decision makes it stale: the next request waits
        await _push_realtime([stream], _frames(speech_pcm(1.0, 1.0)))
        await asyncio.sleep(0.1)
        await _push_realtime([stream], _frames(speech_pcm(0.5, 0.0)))
        await asyncio.sleep(0.1)
        stream._det._pred_timeout = 0.2
        ev = await asyncio.wait_for(stream.predict(), 2)
        assert ev.end_of_turn_probability == 0.0
        stream.end_input()
        await stream.aclose()
        await fe.aclose()


async def test_turn_detector_head_policy():
    async with FakeAudioforgeServer(timeout_ms=480) as srv:
        fe = audioforge.AudioforgeFrontend(srv.url, timeout_ms=480, turn_policy="both")
        stream = fe.turn_detector(policy="head").stream()
        assert await stream.unlikely_threshold(None) == pytest.approx(0.98)
        await _push_realtime([stream], _frames(speech_pcm(1.0, 0.0)))
        await asyncio.sleep(0.05)
        fut = stream.predict()
        await _push_realtime([stream], _frames(speech_pcm(0.0, 0.5)))
        ev = await asyncio.wait_for(fut, 5)
        assert 0.0 <= ev.end_of_turn_probability <= 1.0
        assert stream.predictions[-1]["why"] == "head"
        stream.end_input()
        await stream.aclose()
        await fe.aclose()


def test_numpy_conversion_helpers():
    from livekit.plugins.audioforge.frontend import _PcmConverter

    conv = _PcmConverter()
    stereo = np.array([[1, 3], [5, 7]], dtype=np.int16)
    out = conv(rtc.AudioFrame(stereo.tobytes(), SR, 2, 2))
    assert out.tolist() == [2, 6]
    with pytest.raises(ValueError):
        conv(rtc.AudioFrame(bytes(4), 8000, 1, 2))
