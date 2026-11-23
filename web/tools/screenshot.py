"""Screenshot the demo page in headless Edge once the splat has loaded (window.splatReady).

    python -I web/tools/screenshot.py http://127.0.0.1:8765/?hole=7 out.png [--orbit DEGREES]

--orbit turns the camera about its target before the shot, through the viewer's own orbit controls;
--zoom-out N scrolls the wheel N notches out first (negative N zooms in).
--wait S waits S seconds before the shot (default 4) for the viewer's sort and its spatial index to finish.
"""
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

# Microsoft Edge: EDGE_PATH, else its usual install folders, else msedge on PATH
EDGE = os.environ.get("EDGE_PATH") or next(
    (p for p in (os.path.join(os.environ.get(v, ""), "Microsoft", "Edge", "Application", "msedge.exe")
                 for v in ("ProgramFiles(x86)", "ProgramFiles")) if os.path.isfile(p)), "msedge")
url, out = sys.argv[1], sys.argv[2]
orbit = float(sys.argv[sys.argv.index("--orbit") + 1]) if "--orbit" in sys.argv else 0.0
zoom_out = int(sys.argv[sys.argv.index("--zoom-out") + 1]) if "--zoom-out" in sys.argv else 0
wait = float(sys.argv[sys.argv.index("--wait") + 1]) if "--wait" in sys.argv else 4.0
port = 9333
profile = tempfile.mkdtemp(prefix="edge_shot_")
edge = subprocess.Popen([EDGE, "--headless=new", "--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist",
                         f"--remote-debugging-port={port}", f"--user-data-dir={profile}", "--window-size=1600,900",
                         "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def main() -> None:
    for _ in range(50):
        try:
            targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
            break
        except OSError:
            time.sleep(0.2)
    page = next(t for t in targets if t["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
        ids = iter(range(1, 10 ** 6))

        async def call(method, **params):
            i = next(ids)
            await ws.send(json.dumps({"id": i, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == i:
                    return msg.get("result", msg.get("error"))

        await call("Page.enable")
        await call("Page.navigate", url=url)
        t0 = time.time()
        while time.time() - t0 < 300:
            # Until the new page has parsed, #error is missing (the old about:blank may still be current).
            r = await call("Runtime.evaluate", returnByValue=True,
                           expression="[!!window.splatReady, (document.getElementById('error') || {}).textContent || '']")
            ready, err = r["result"]["value"]
            if ready or err:
                break
            await asyncio.sleep(1)
        print(f"ready={ready} after {time.time() - t0:.0f} s {err}")
        if orbit:
            # Drag across the canvas with the left button: OrbitControls turns the camera about its target.
            steps, dx = 30, orbit / 360 * 1600 * 0.9
            await call("Input.dispatchMouseEvent", type="mousePressed", x=800, y=450, button="left", clickCount=1)
            for k in range(1, steps + 1):
                await call("Input.dispatchMouseEvent", type="mouseMoved", x=800 + dx * k / steps, y=450, button="left", buttons=1)
                await asyncio.sleep(0.02)
            await call("Input.dispatchMouseEvent", type="mouseReleased", x=800 + dx, y=450, button="left", clickCount=1)
        for _ in range(abs(zoom_out)):  # negative zooms in
            await call("Input.dispatchMouseEvent", type="mouseWheel", x=800, y=450, deltaX=0,
                       deltaY=120 if zoom_out > 0 else -120)
            await asyncio.sleep(0.05)
        await asyncio.sleep(wait)  # let the sort settle
        shot = await call("Page.captureScreenshot", format="png")
        with open(out, "wb") as f:
            f.write(base64.b64decode(shot["data"]))


try:
    asyncio.run(main())
finally:
    edge.terminate()
