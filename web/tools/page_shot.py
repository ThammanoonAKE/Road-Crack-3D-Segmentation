"""Screenshot a page of the site in headless Edge, for design review.

    python -I web/tools/page_shot.py URL OUT.png [--width 1440] [--height 900] [--full] [--wait 3] [--ready]
                                    [--scale 1] [--eval JS]

--full captures the whole page height; --ready waits for window.splatReady (the viewer sets it once the
reconstruction is drawn) before the --wait delay; --eval runs a JavaScript expression first (e.g. a click).
"""
import argparse
import asyncio
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request

import websockets

# Microsoft Edge: EDGE_PATH, else its usual install folders, else msedge on PATH
EDGE = os.environ.get("EDGE_PATH") or next(
    (p for p in (os.path.join(os.environ.get(v, ""), "Microsoft", "Edge", "Application", "msedge.exe")
                 for v in ("ProgramFiles(x86)", "ProgramFiles")) if os.path.isfile(p)), "msedge")
ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("out")
ap.add_argument("--width", type=int, default=1440)
ap.add_argument("--height", type=int, default=900)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--full", action="store_true")
ap.add_argument("--ready", action="store_true")
ap.add_argument("--wait", type=float, default=3.0)
ap.add_argument("--eval", default="")
ap.add_argument("--port", type=int, default=9334)
ap.add_argument("--console", action="store_true", help="print the page's console messages and exceptions")
args = ap.parse_args()

profile = tempfile.mkdtemp(prefix="edge_page_")
edge = subprocess.Popen([EDGE, "--headless=new", "--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist",
                         "--hide-scrollbars", f"--remote-debugging-port={args.port}", f"--user-data-dir={profile}",
                         f"--window-size={args.width},{args.height}", "about:blank"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def main() -> None:
    for _ in range(50):
        try:
            targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/json"))
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
                if args.console and msg.get("method") == "Runtime.consoleAPICalled":
                    p = msg["params"]
                    text = " ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", []))
                    print(f"[console.{p.get('type')}] {text[:400]}", flush=True)
                if args.console and msg.get("method") == "Runtime.exceptionThrown":
                    d = msg["params"]["exceptionDetails"]
                    print(f"[exception] {d.get('exception', {}).get('description', d.get('text'))[:600]}", flush=True)

        if args.console:
            await call("Runtime.enable")

        await call("Emulation.setDeviceMetricsOverride", width=args.width, height=args.height,
                   deviceScaleFactor=args.scale, mobile=args.width < 700)
        await call("Page.enable")
        await call("Page.navigate", url=args.url)
        t0 = time.time()
        while time.time() - t0 < 60:
            r = await call("Runtime.evaluate", returnByValue=True, expression="document.readyState")
            if r["result"]["value"] == "complete":
                break
            await asyncio.sleep(0.25)
        if args.ready:
            while time.time() - t0 < 300:
                r = await call("Runtime.evaluate", returnByValue=True, expression="!!window.splatReady")
                if r["result"]["value"]:
                    break
                await asyncio.sleep(1)
            print(f"ready after {time.time() - t0:.0f} s")
        if args.eval:
            await call("Runtime.evaluate", expression=args.eval, awaitPromise=True)
        await asyncio.sleep(args.wait)
        params = {"format": "png"}
        if args.full:
            m = await call("Page.getLayoutMetrics")
            h = int(m["cssContentSize"]["height"])
            await call("Emulation.setDeviceMetricsOverride", width=args.width, height=h,
                       deviceScaleFactor=args.scale, mobile=args.width < 700)
            await asyncio.sleep(1.0)
            params.update(captureBeyondViewport=True)
        shot = await call("Page.captureScreenshot", **params)
        with open(args.out, "wb") as f:
            f.write(base64.b64decode(shot["data"]))


try:
    asyncio.run(main())
finally:
    edge.terminate()
