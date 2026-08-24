"""Requesting an event by hand -- DESIGN.md §4.3.

The point of these tests is that there is no such thing as a "manual event".
The control panel chooses *when* an event arrives and *which* kind; everything
that makes an event non-punctuating -- the raised-cosine envelope, the radius
cap, the amplitude, the concurrency limit, the fact that it lands on climate --
is the scheduler's, and a requested event goes through exactly the same code
that a sampled one does. So each test below pins one of those rules to the
requested path rather than to the automatic one.
"""

from __future__ import annotations

import json

import panelstub
import pytest

from anastomosis import config, events


def _params(**kwargs) -> config.EventParams:
    return config.EventParams(**kwargs)


# ---------------------------------------------------------------------------
# The scheduler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(events.EVENT_KINDS))
def test_a_request_produces_the_kind_that_was_asked_for(kind):
    """Which kind arrives is the one thing a request gets to decide."""
    scheduler = events.EventScheduler(seed=5)
    event = scheduler.trigger(kind, _params())
    assert event is not None
    assert event.kind == kind
    assert scheduler.active == [event]

    base = events.EVENT_KINDS[kind]
    for name, value in zip(events.Channels._fields, event.channels):
        expected = getattr(base, name)
        if expected == 0.0:
            # A kind that ignores a channel must still ignore it: the jitter is
            # multiplicative, so it can never introduce one.
            assert value == 0.0, name
        else:
            assert value / expected == pytest.approx(1.0, abs=0.25), name


@pytest.mark.parametrize("kind", sorted(events.EVENT_KINDS))
def test_a_requested_event_is_shaped_like_a_sampled_one(kind):
    """Same size cap, same amplitude, same envelope timings -- and no step."""
    params = _params()
    scheduler = events.EventScheduler(seed=9)
    event = scheduler.trigger(kind, params)

    assert 0.0 < event.radius <= params.max_radius_frac
    assert 0.0 < event.peak <= params.strength
    assert 0.0 <= event.x <= 1.0 and 0.0 <= event.y <= 1.0
    for got, nominal in (
        (event.attack, params.attack_seconds),
        (event.hold, params.hold_seconds),
        (event.release, params.release_seconds),
    ):
        assert 0.75 * nominal <= got <= 1.25 * nominal

    # It starts from nothing, which is the whole reason the envelope exists.
    assert event.elapsed == 0.0
    assert event.envelope() == pytest.approx(0.0)
    rows, count = scheduler.pack(8)
    assert (rows, count) == ([], 0)


def test_the_arrival_rate_reaches_the_scheduler():
    """The panel's frequency knob is only real if the stream actually thins.

    `rate_per_hour` is the one thing that knob moves, so this pins both halves
    of what it promises: the rate is in events per *hour*, and turning it up
    produces more arrivals rather than merely a different sample.
    """
    def spawned(rate: float, hours: int, seed: int) -> int:
        scheduler = events.EventScheduler(seed=seed)
        params = _params(rate_per_hour=rate, max_concurrent=8)
        for _ in range(3600 * hours):
            scheduler.update(1.0, params)
        return scheduler.spawned

    rare = spawned(2.0, 10, seed=3)
    often = spawned(18.0, 10, seed=3)
    assert rare < often

    # And the unit is arrivals per hour, not per anything else.
    assert 0.7 * 20 <= rare <= 1.3 * 20
    assert 0.7 * 180 <= often <= 1.3 * 180


def test_the_rate_does_not_change_the_events_it_produces():
    """Frequency is not amplitude -- what arrives is the same either way.

    This is why the frequency control needs no ceiling of its own: at the top
    of its travel the field spends more of its time inside an event, and no
    event is bigger, stronger or longer than it was at the bottom.
    """
    def first(rate: float) -> events.ActiveEvent:
        scheduler = events.EventScheduler(seed=11)
        return scheduler.trigger("bloom", _params(rate_per_hour=rate))

    slow, fast = first(0.5), first(20.0)
    assert (slow.radius, slow.peak, slow.duration) == (
        fast.radius, fast.peak, fast.duration
    )
    assert slow.channels == fast.channels


def test_a_requested_event_reaches_the_gpu_and_then_expires():
    """It is packed like any other event, and retires on its own."""
    # A rate low enough that the automatic stream cannot land inside the long
    # `update` steps below and be mistaken for the event under test.
    params = _params(rate_per_hour=0.001)
    scheduler = events.EventScheduler(seed=4)
    event = scheduler.trigger("bloom", params)

    scheduler.update(event.attack, params)
    (row,), count = scheduler.pack(8)
    assert count == 1
    assert row["chan_feed"] > 0.0 and row["chan_kill"] < 0.0
    assert row["strength"] == pytest.approx(event.peak, rel=1e-6)
    assert row["radius"] == event.radius

    # Advance past the release: it goes away without anything cleaning up after
    # it, exactly as a sampled event does.
    scheduler.update(event.duration, params)
    assert scheduler.active == []
    assert scheduler.pack(8) == ([], 0)


def test_the_concurrency_cap_applies_to_requests():
    """The cap is what stops overlapping events summing into punctuation."""
    params = _params(max_concurrent=2)
    scheduler = events.EventScheduler(seed=2)
    assert scheduler.trigger("bloom", params) is not None
    assert scheduler.trigger("dieback", params) is not None
    # Refused rather than queued: an event arriving minutes after the press
    # that asked for it would be worse than none.
    assert scheduler.trigger("current", params) is None
    assert len(scheduler.active) == 2

    # And room reappears as the events in flight fade.
    scheduler.update(max(e.duration for e in scheduler.active), params)
    assert scheduler.trigger("current", params) is not None


def test_a_cap_of_zero_means_no_events_at_all():
    scheduler = events.EventScheduler(seed=2)
    assert scheduler.trigger("bloom", _params(max_concurrent=0)) is None
    assert scheduler.active == []


def test_a_request_does_not_reschedule_the_automatic_stream():
    """Poisson arrivals are memoryless; clicks must not make them predictable."""
    params = _params(rate_per_hour=0.001)
    scheduler = events.EventScheduler(seed=13)
    scheduler.update(0.1, params)
    before = scheduler._time_to_next
    assert before is not None

    scheduler.trigger("tint", params)
    assert scheduler._time_to_next == before


def test_requests_work_while_automatic_arrivals_are_off():
    """`enabled` gates arrivals the scheduler draws, not events themselves."""
    params = _params(enabled=False)
    scheduler = events.EventScheduler(seed=17)
    event = scheduler.trigger("rift", params)
    assert event is not None

    # And it still runs its envelope and retires, rather than being stranded.
    scheduler.update(event.attack, params)
    assert scheduler.pack(8)[1] == 1
    scheduler.update(event.duration, params)
    assert scheduler.active == []


def test_an_unknown_kind_is_refused_loudly():
    """A typo'd kind must not silently do nothing, or half of one."""
    scheduler = events.EventScheduler(seed=1)
    with pytest.raises(KeyError):
        scheduler.trigger("blooom", _params())
    assert scheduler.active == []


def test_a_requested_event_survives_a_checkpoint():
    """Nothing about the requested path is outside what gets saved."""
    params = _params()
    scheduler = events.EventScheduler(seed=21)
    scheduler.trigger("rift", params)
    scheduler.update(30.0, params)

    resumed = events.EventScheduler(seed=99)
    resumed.load_state(json.loads(json.dumps(scheduler.state())))
    assert resumed.active == scheduler.active
    assert resumed.pack(8) == scheduler.pack(8)


# ---------------------------------------------------------------------------
# The control panel
# ---------------------------------------------------------------------------


class _StubApp(panelstub.PanelApp):
    """The shared panel stand-in, recording what this module watches."""

    def __init__(self):
        super().__init__()
        self.requested: list[str] = []
        self.fullscreen_toggles = 0

    def toggle_fullscreen(self) -> bool:
        self.fullscreen_toggles += 1
        return False

    def trigger_event(self, kind: str) -> bool:
        self.requested.append(kind)
        return super().trigger_event(kind)


def _panel(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6", reason="the control panel needs the [ui] extra")
    from PySide6 import QtWidgets

    from anastomosis.ui.control_panel import ControlPanel

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app = _StubApp()
    return app, ControlPanel(app)


def test_every_kind_has_a_button(monkeypatch):
    """A kind the simulation can produce but the panel cannot ask for is a gap."""
    app, panel = _panel(monkeypatch)
    assert set(panel.event_buttons) == set(events.EVENT_KINDS)
    for kind, button in panel.event_buttons.items():
        assert button.text().lower() == kind
        assert button.toolTip()


def test_pressing_a_button_asks_for_that_kind(monkeypatch):
    app, panel = _panel(monkeypatch)
    panel.event_buttons["dieback"].click()
    assert app.requested == ["dieback"]
    (event,) = app.scheduler.active
    assert event.kind == "dieback"
    assert "Dieback" in panel.event_note.text()


def test_the_buttons_go_quiet_when_the_cap_is_reached(monkeypatch):
    """The refusal is visible, rather than a button that appears to do nothing."""
    app, panel = _panel(monkeypatch)
    app.params.events.max_concurrent = 1

    panel.event_buttons["bloom"].click()
    assert all(not b.isEnabled() for b in panel.event_buttons.values())
    # The reply to what the user just pressed outranks the standing text.
    assert "Bloom" in panel.event_note.text()

    panel._note_expires = 0.0
    panel._refresh_events()
    assert "as the settings allow" in panel.event_note.text()

    # Nothing arrives from a press that was refused.
    app.scheduler.trigger = lambda *a, **k: None
    panel._on_trigger("tint")
    assert "Not now" in panel.event_note.text()


# ---------------------------------------------------------------------------
# Audio-shaped requests -- DESIGN.md §16.4
# ---------------------------------------------------------------------------


def test_vigor_chooses_within_the_sampled_peak_range():
    """The shaped peak lands on exactly the interval the RNG draws from
    (0.6-1.0 x strength): the strongest shaped event is precisely the
    strongest random one, and the gentlest shaped one is the floor of the
    draw, not zero -- an onset asked for an event, however soft."""
    params = _params()
    scheduler = events.EventScheduler(seed=7)
    soft = scheduler.trigger("bloom", params, vigor=0.0, pace=0.0)
    hard = scheduler.trigger("bloom", params, vigor=1.0, pace=0.0)
    assert soft.peak == pytest.approx(0.6 * params.strength)
    assert hard.peak == pytest.approx(1.0 * params.strength)
    # And hostile hints clamp rather than escape the interval.
    scheduler.active.clear()
    wild = scheduler.trigger("bloom", params, vigor=7.0, pace=-3.0)
    assert wild.peak == pytest.approx(1.0 * params.strength)
    assert 0.75 * params.attack_seconds <= wild.attack <= 1.25 * params.attack_seconds


def test_pace_shortens_toward_the_certified_floor_and_no_further():
    """Full pace takes the default minute-scale envelope to the floors --
    within the same 0.75-1.25 jitter every drawn event gets -- and never
    below them, whatever the resolved values were."""
    params = _params()  # regulation defaults: 45 / 60 / 90
    scheduler = events.EventScheduler(seed=9)
    brisk = scheduler.trigger("current", params, vigor=0.5, pace=1.0)
    assert 0.75 * events.ATTACK_FLOOR_SECONDS <= brisk.attack \
        <= 1.25 * events.ATTACK_FLOOR_SECONDS
    assert 0.75 * events.HOLD_FLOOR_SECONDS <= brisk.hold \
        <= 1.25 * events.HOLD_FLOOR_SECONDS
    assert 0.75 * events.RELEASE_FLOOR_SECONDS <= brisk.release \
        <= 1.25 * events.RELEASE_FLOOR_SECONDS

    scheduler.active.clear()
    unhurried = scheduler.trigger("current", params, vigor=0.5, pace=0.0)
    assert unhurried.attack >= 0.75 * params.attack_seconds


def test_a_resolved_envelope_under_the_floor_is_left_alone():
    """Shaping shortens; it never lengthens. A hand-overridden envelope
    already under the floor keeps exactly what its owner asked for."""
    params = _params(attack_seconds=3.0, hold_seconds=2.0, release_seconds=4.0)
    scheduler = events.EventScheduler(seed=11)
    event = scheduler.trigger("tint", params, vigor=1.0, pace=1.0)
    assert 0.75 * 3.0 <= event.attack <= 1.25 * 3.0
    assert 0.75 * 2.0 <= event.hold <= 1.25 * 2.0
    assert 0.75 * 4.0 <= event.release <= 1.25 * 4.0


def test_an_unshaped_request_is_exactly_what_it_always_was():
    """The panel's buttons pass no hints; their draws must be untouched --
    same RNG stream, same distributions, bit for bit."""
    before = events.EventScheduler(seed=13)._spawn(_params(), kind="bloom")
    after = events.EventScheduler(seed=13)._spawn(
        _params(), kind="bloom", vigor=None, pace=None)
    assert before == after


def test_the_floors_cannot_step_the_envelope():
    """The certified worst case, walked directly: every floor at its 0.75
    jitter, ticked at the fastest rate any mode reaches, moves the raised
    cosine well under the bound the per-mode soak test asserts."""
    scheduler = events.EventScheduler(seed=15)
    event = scheduler._spawn(_params(), kind="bloom", vigor=1.0, pace=1.0)
    event.attack = events.ATTACK_FLOOR_SECONDS * 0.75
    event.hold = events.HOLD_FLOOR_SECONDS * 0.75
    event.release = events.RELEASE_FLOOR_SECONDS * 0.75
    event.elapsed = 0.0

    dt = 1.0 / 30.0
    previous = event.envelope()
    largest = 0.0
    while not event.finished:
        event.elapsed += dt
        current = event.envelope()
        largest = max(largest, abs(current - previous))
        previous = current
    assert largest < 0.02, f"floored envelope stepped by {largest:.4f}"
