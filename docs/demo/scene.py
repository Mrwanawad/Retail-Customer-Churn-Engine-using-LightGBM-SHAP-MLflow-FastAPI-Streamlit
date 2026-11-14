"""Demo scene. Regenerate:
  streamlit run app/streamlit_app.py --server.port 8760
  python docs/demo/scene.py
  python ~/.claude/skills/demo-gif/scripts/compose.py docs/demo/build --out docs/demo.gif --fps 12 --width 1000
"""
import os, sys; sys.path.insert(0, os.path.expanduser("~/.claude/skills/demo-gif/scripts"))
from pathlib import Path
from director import Director, run

URL = os.environ.get("DEMO_URL", "http://localhost:8760")


async def union(*locs):
    boxes = [await l.bounding_box() for l in locs]
    x0 = min(b["x"] for b in boxes); y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["width"] for b in boxes); y1 = max(b["y"] + b["height"] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


async def scene(d: Director):
    p = d.page
    nav = lambda name: p.get_by_role("link", name=name)
    chart = p.locator("[data-testid='stPlotlyChart']")
    # load off-camera, then mark the start so the first frame is the fully rendered app
    await p.goto(URL, wait_until="networkidle")
    await p.get_by_text("Churn overview").first.wait_for(timeout=60_000)
    await chart.first.wait_for(timeout=60_000)
    await d.hold(2.0)
    d._log("nav", url=URL)
    await d.hold(0.5)

    await d.caption("Scan this week's churn risk", step=1)
    await d.hold(1.3)

    await d.caption("Open the riskiest customer", step=2)
    await d.click(nav("Customer scorer"))
    await d.fast(p.get_by_text("Top reasons").wait_for(timeout=60_000), label="Scoring")
    await chart.first.wait_for(timeout=60_000)
    await d.hold(0.7)
    await d.focus(await union(chart.first, p.get_by_text("Top reasons"), p.get_by_text("Profile at snapshot")), zoom=1.25)
    bb = await p.get_by_text("No visit in 67 days").bounding_box()
    await d.arrow((bb["x"], bb["y"], 170, bb["height"]), "Plain-language reasons", side="right")
    await d.hold(2.0)
    await d.clear_arrow(); await d.unfocus()

    await d.caption("Target by expected profit", step=3)
    await d.click(nav("Campaign planner"))
    await d.fast(p.get_by_text("Customers targeted", exact=True).wait_for(timeout=60_000), label="Loading")
    await chart.first.wait_for(timeout=60_000)
    await d.hold(0.6)
    await d.focus(await union(p.get_by_text("Customers targeted", exact=True), chart.first), zoom=1.3)
    await d.hold(2.0)
    await d.unfocus()

    await d.caption("Check drift and retrain status", step=4)
    await d.click(nav("Model card"))
    await d.fast(p.get_by_text("Out-of-time validation").wait_for(timeout=60_000), label="Loading")
    await d.move(p.get_by_text("Retrain triggered", exact=True))
    await d.hold(2.0)

run(scene, out=str(Path(__file__).with_name("build")), viewport=(1280, 820), title="Retail Churn Engine",
    subtitle="Weekly churn scores, SHAP reasons and profit-based targeting", url_label="retail-churn-engine")
