"""A stand-in for the desktop compositor: measure how long this GPU makes
small jobs wait.

Run this in its own process while anastomosis (or anything else) is using the
GPU. Every interval it submits a trivial one-workgroup compute dispatch on its
own device and measures the wall time from submit to completion -- the same
shape of work a desktop compositor submits every frame, and the thing that
misses its deadline when another process holds the hardware queue with
back-to-back submissions (the WDDM queue-occupancy problem gpu_nice exists
to solve; see DESIGN.md and issue #40).

The number to watch is p99. An idle desktop gives well under a millisecond;
low single digits means small jobs are slotting into gaps promptly; hundreds
of milliseconds means the compositor is starving and windows visibly stop
compositing.

Usage:
    python tools/gpu_probe.py --seconds 60 --interval 0.1

Prints a percentile summary at the end, and a rolling line every few seconds
so a live run can be watched. Exits cleanly on Ctrl+C, printing the summary
for whatever it measured.
"""

from __future__ import annotations

import argparse
import statistics
import time

import wgpu

SHADER = """
@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out[gid.x] = out[gid.x] + 1u;
}
"""


def drain(device: wgpu.GPUDevice) -> None:
    """Block until every submission on this device has completed."""
    poll_wait = getattr(device, "_poll_wait", None)
    if poll_wait is not None:
        poll_wait()
    else:  # pragma: no cover - future wgpu-py without the private poll
        device.queue.on_submitted_work_done_sync()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def summarise(latencies: list[float]) -> str:
    ms = [v * 1000 for v in latencies]
    return (
        f"n={len(ms)}  p50={percentile(ms, 0.50):.2f}ms  "
        f"p90={percentile(ms, 0.90):.2f}ms  p99={percentile(ms, 0.99):.2f}ms  "
        f"max={max(ms):.2f}ms  mean={statistics.mean(ms):.2f}ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seconds", type=float, default=60.0,
        help="how long to probe (default: 60)",
    )
    parser.add_argument(
        "--interval", type=float, default=0.1,
        help="seconds between probes (default: 0.1, the pace of a 10 Hz "
             "compositor with plenty of slack)",
    )
    parser.add_argument(
        "--report-every", type=float, default=5.0,
        help="seconds between rolling report lines (default: 5)",
    )
    args = parser.parse_args()

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(label="gpu_probe")
    print(f"probing on: {adapter.info.get('device', '?')}", flush=True)

    module = device.create_shader_module(code=SHADER)
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": module, "entry_point": "main"},
    )
    buffer = device.create_buffer(
        size=64, usage=wgpu.BufferUsage.STORAGE,
    )
    bind = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {
            "buffer": buffer, "offset": 0, "size": 64}}],
    )

    def one_probe() -> float:
        t0 = time.perf_counter()
        encoder = device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(pipeline)
        cpass.set_bind_group(0, bind)
        cpass.dispatch_workgroups(1, 1, 1)
        cpass.end()
        device.queue.submit([encoder.finish()])
        drain(device)
        return time.perf_counter() - t0

    # Warm-up dispatches, so pipeline compilation and the driver's first-use
    # costs are not samples.
    for _ in range(3):
        one_probe()

    latencies: list[float] = []
    started = time.perf_counter()
    last_report = started
    try:
        while time.perf_counter() - started < args.seconds:
            t0 = time.perf_counter()
            latencies.append(one_probe())

            now = time.perf_counter()
            if now - last_report >= args.report_every and latencies:
                print(summarise(latencies), flush=True)
                last_report = now
            time.sleep(max(args.interval - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass

    if latencies:
        print("final: " + summarise(latencies), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
