#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Gradio dashboard
# 
# You're using a notebook for this stage. If you'd rather use a script,
# write `app.py` instead and delete this file. `main.py` refuses to run if
# it finds both `app.py` and `app.ipynb`, so exactly one of them must exist.
# 
# Replace this cell and everything below it with your own code. When run
# via `uv run python main.py app`, this notebook must build and launch a Gradio app
# (a `demo` that calls `.launch()`) using only what your training stage already
# produced. It must never train anything itself. Load the committed weights in
# `weights/`, never whatever the training run left in `runs/`.
# 
# At minimum, the dashboard must let you:
# 
# - Upload an image and see the model's predictions at the current threshold.
# - Move the two edges of your escalation band and watch the confusion matrix,
#   the expected cost per 1,000 units, and the share of cases escalated to a
#   human all change together.
# - See the cost-against-threshold curve with your chosen operating point marked
#   on it.
# - See per-class performance, not just the overall number.
# 
# How you structure the code beyond that is your call to make and be able to
# defend: file layout, function boundaries, naming.

# In[2]:


import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from ultralytics import YOLO

R = Path("results")
model = YOLO("weights/best.pt")
scores = pd.read_csv(R / "test_scores.csv")
per_class = json.load(open(R / "per_class.json"))["per_class"]
policy = json.load(open(R / "operating_policy.json"))
single = json.load(open(R / "operating_point.json"))
A = json.load(open(R / "review_policy.json"))["assumptions"]
cost_sweep = pd.read_csv(R / "cost_sweep.csv")
tradeoff = pd.read_csv(R / "review_tradeoff.csv")

FN, REV, FPA, BASE = A["fn_cost"], A["review_cost"], A["fp_auto_cost"], A["base_rate"]
DAY = policy["returns_per_day"]
MINS = policy["assumptions"]["minutes_per_review"]
HRS = policy["assumptions"]["productive_hours_per_person"]
y = scores["has_damage"].to_numpy()
s = scores["score"].to_numpy()

NAVY, TEAL, AMBER, RED, GREY = "#1f3a5f", "#2a9d8f", "#f4a261", "#e63946", "#8d99ae"

CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
.hero {background: linear-gradient(120deg, #1f3a5f 0%, #2b6a8f 60%, #2a9d8f 100%); color: white;
       padding: 26px 30px; border-radius: 18px; margin-bottom: 6px;}
.hero h1 {color: white !important; margin: 0 0 6px 0; font-size: 30px;}
.hero p {color: #e6f0f5; margin: 0; font-size: 15px;}
.chips {margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap;}
.chip {padding: 5px 14px; border-radius: 999px; font-size: 13px; font-weight: 600; color: white;}
.cards {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 6px 0 4px 0;}
.card {background: white; border-radius: 14px; padding: 14px 16px; border: 1px solid #e3e8ef; border-top-width: 5px;
       box-shadow: 0 2px 8px rgba(31,58,95,0.07);}
.card .label {font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #667085; font-weight: 600;}
.card .value {font-size: 30px; font-weight: 800; color: #1f3a5f; line-height: 1.15; margin-top: 4px;}
.card .sub {font-size: 12.5px; color: #667085; margin-top: 3px;}
.verdict {border-radius: 16px; padding: 18px 22px; color: white; margin-top: 8px;}
.verdict .big {font-size: 26px; font-weight: 800;} .verdict .small {font-size: 14px; opacity: .95; margin-top: 4px;}
.bar {height: 12px; border-radius: 6px; background: rgba(255,255,255,.35); margin-top: 12px; position: relative;}
.bar > div {height: 12px; border-radius: 6px; background: white;}
.note {color: #667085; font-size: 13px;}
"""


def card(label, value, sub, color):
    return (f'<div class="card" style="border-top-color:{color}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def band_numbers(lo, hi):
    hi = max(hi, lo)
    zone = np.where(s < lo, 0, np.where(s >= hi, 2, 1))          # 0 pass, 1 review, 2 fail
    counts = np.array([[(zone[y == v] == k).sum() for k in range(3)] for v in (1, 0)])
    shares = counts / counts.sum(axis=1, keepdims=True)
    (pp, pr, pf), (cp, cr, cf) = shares
    cost = 1000 * (BASE * (pp * FN + pr * REV) + (1 - BASE) * (cr * REV + cf * FPA))
    review = BASE * pr + (1 - BASE) * cr
    return hi, counts, cost, review, 1000 * BASE * pp, 1000 * (1 - BASE) * cf


def band_view(lo, hi):
    hi, counts, cost, review, missed, wrong = band_numbers(lo, hi)
    per_day = DAY * review
    hours = per_day * MINS / 60
    everything = 1000 * REV
    cards = "".join([
        card("Cost per 1,000 returns", f"EUR {cost:,.0f}",
             f"a person checking every car: EUR {everything:,.0f}", TEAL if cost < everything else RED),
        card("Sent to a person", f"{review:.1%}", f"{per_day:.0f} reviews a day at {DAY} returns", AMBER),
        card("Reviewer time", f"{hours:.1f} h/day", f"about {hours / HRS:.2f} people at {MINS} min each", AMBER),
        card("Damage slips through", f"{missed:.1f}", "damaged cars per 1,000 returns", RED),
        card("Wrong automatic charges", f"{wrong:.1f}", "clean cars failed per 1,000 returns", RED),
    ])
    html = f'<div class="cards">{cards}</div>'

    f1 = Figure(figsize=(7.2, 3.8)); ax = f1.subplots()
    bins = np.linspace(0, 1, 41)
    ax.hist(s[y == 0], bins=bins, color=TEAL, alpha=.85, label="Clean cars (test)")
    ax.hist(s[y == 1], bins=bins, color=RED, alpha=.75, label="Damaged cars (test)")
    ax.axvspan(0, lo, color=TEAL, alpha=.10); ax.axvspan(lo, hi, color=AMBER, alpha=.18); ax.axvspan(hi, 1, color=RED, alpha=.10)
    ax.axvline(lo, color=NAVY, lw=1.5); ax.axvline(hi, color=NAVY, lw=1.5)
    top = ax.get_ylim()[1]
    ax.text(lo / 2, top * .93, "PASS", ha="center", color=TEAL, fontweight="bold")
    ax.text((lo + hi) / 2, top * .93, "REVIEW", ha="center", color="#b5651d", fontweight="bold")
    ax.text((hi + 1) / 2, top * .93, "FAIL", ha="center", color=RED, fontweight="bold")
    ax.set_xlabel("Damage score from the model"); ax.set_ylabel("Number of test photos")
    ax.set_title("Where the test photos land", loc="left", fontweight="bold", color=NAVY)
    ax.legend(loc="center right", frameon=False); ax.spines[["top", "right"]].set_visible(False)
    f1.tight_layout()

    f2 = Figure(figsize=(5.6, 3.8)); ax = f2.subplots()
    ax.imshow(counts / counts.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1.15)
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{counts[i, j]}", ha="center", va="center", fontsize=18, fontweight="bold",
                    color="white" if counts[i, j] / counts[i].sum() > .55 else NAVY)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Pass\n(no person)", "Review\n(a person)", "Fail\n(no person)"])
    ax.set_yticks(range(2)); ax.set_yticklabels(["Damaged\n(true)", "Clean\n(true)"])
    ax.set_title("Confusion matrix on the test photos", loc="left", fontweight="bold", color=NAVY)
    for sp in ax.spines.values(): sp.set_visible(False)
    f2.tight_layout()
    return html, f1, f2


def cost_view(lo, hi):
    hi, _, cost, review, _, _ = band_numbers(lo, hi)
    f1 = Figure(figsize=(7, 4)); ax = f1.subplots()
    ax.plot(cost_sweep["threshold"], cost_sweep["cost_per_1000"], color=NAVY, lw=2)
    t = single["chosen_threshold"]
    ax.scatter([t], [single["cost_per_1000_at_chosen"]], color=RED, s=70, zorder=3, label=f"chosen threshold {t:.2f}")
    ax.axvline(0.5, color=GREY, ls="--", lw=1, label="provisional 0.5")
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Cost per 1,000 returns (EUR)")
    ax.set_title("One threshold, every flagged car reviewed", loc="left", fontweight="bold", color=NAVY)
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); f1.tight_layout()

    f2 = Figure(figsize=(7, 4)); ax = f2.subplots()
    ax.plot(tradeoff["max_review_share"] * 100, tradeoff["min_cost_per_1000"], color=NAVY, lw=2, label="best possible")
    ax.scatter([policy["review_share"] * 100], [policy["cost_per_1000"]], color=RED, s=70, zorder=3,
               label=f"chosen band {policy['t_low']:.2f} to {policy['t_high']:.2f}")
    ax.scatter([review * 100], [cost], color=AMBER, s=110, zorder=4, edgecolor=NAVY, label="current sliders")
    ax.set_xlabel("Share of returns sent to a person (%)"); ax.set_ylabel("Cost per 1,000 returns (EUR)")
    ax.set_title("What it costs to automate more", loc="left", fontweight="bold", color=NAVY)
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); f2.tight_layout()
    return f1, f2


def predict(img, lo, hi):
    if img is None:
        return None, '<div class="note">Upload a car photo and press the button.</div>'
    hi = max(hi, lo)
    bgr = img[..., ::-1]                                           # Gradio gives RGB, the model expects BGR
    allbox = model.predict(bgr, conf=0.001, imgsz=640, verbose=False)[0]
    score = max([float(c) for c, k in zip(allbox.boxes.conf, allbox.boxes.cls) if int(k) == 0], default=0.0)
    shown = model.predict(bgr, conf=max(lo, 0.01), imgsz=640, verbose=False)[0].plot()[..., ::-1]
    if score < lo:
        color, big, small = TEAL, "PASS", "No visible damage. No person needed."
    elif score >= hi:
        color, big, small = RED, "FAIL", "Damage detected. Charge and repair without a person."
    else:
        color, big, small = AMBER, "SEND TO A PERSON", "The model is unsure, so a reviewer looks at it."
    html = (f'<div class="verdict" style="background:{color}"><div class="big">{big}</div>'
            f'<div class="small">{small} Damage score {score:.2f} (low edge {lo:.2f}, high edge {hi:.2f})</div>'
            f'<div class="bar"><div style="width:{min(score, 1) * 100:.0f}%"></div></div></div>')
    return shown, html


def class_cards():
    out = []
    for name, m in per_class.items():
        pos = name == "Damage-detected"
        body = (f'<div class="sub">Precision <b>{m["precision"]:.2f}</b> &nbsp; Recall <b>{m["recall"]:.2f}</b><br>'
                f'AP50 <b>{m["AP50"]:.2f}</b> &nbsp; mAP50-95 <b>{m["mAP50-95"]:.2f}</b></div>')
        out.append(card(name + (" (positive class)" if pos else ""), f'{m["AP50"]:.2f} AP50', body, RED if pos else TEAL))
    return f'<div class="cards">{"".join(out)}</div>'


HERO = """
<div class="hero"><h1>Returned-car damage check</h1>
<p>A model scores every returned car. Clear cases are decided automatically and only the unsure ones go to a person.
Move the two edges below and watch the cost, the workload and the errors change.</p>
<div class="chips"><span class="chip" style="background:#2a9d8f">PASS: below the low edge</span>
<span class="chip" style="background:#f4a261;color:#4a2c0a">REVIEW: a person decides</span>
<span class="chip" style="background:#e63946">FAIL: from the high edge up</span></div></div>
"""

with gr.Blocks(title="Returned-car damage check") as demo:
    gr.HTML(HERO)
    with gr.Row():
        lo = gr.Slider(0, 1, value=policy["t_low"], step=0.01, label="Low edge (below it, the car passes without a person)")
        hi = gr.Slider(0, 1, value=policy["t_high"], step=0.01, label="High edge (from it up, the car fails without a person)")
    with gr.Tabs():
        with gr.Tab("Try a photo"):
            with gr.Row():
                with gr.Column():
                    img = gr.Image(type="numpy", label="Car photo", height=380)
                    btn = gr.Button("Check this car", variant="primary")
                with gr.Column():
                    shown = gr.Image(label="What the model sees", height=380, interactive=False)
            verdict = gr.HTML()
            btn.click(predict, [img, lo, hi], [shown, verdict])
        with gr.Tab("Band, confusion matrix and cost"):
            kpis = gr.HTML()
            with gr.Row():
                hist = gr.Plot(label="Score distribution")
                matrix = gr.Plot(label="Confusion matrix")
            gr.HTML(f'<p class="note">Costs per 1,000 returns assume {BASE:.0%} of cars are damaged, a missed damage costs '
                    f'EUR {FN}, a review EUR {REV} and a wrong automatic charge EUR {FPA}. The reviewer is assumed never to '
                    f'make a mistake. The counts come from the {int((y == 1).sum())} damaged and {int((y == 0).sum())} '
                    f'clean test photos.</p>')
        with gr.Tab("Cost curves"):
            with gr.Row():
                c1 = gr.Plot(label="Cost vs threshold")
                c2 = gr.Plot(label="Cost vs human review")
        with gr.Tab("Per-class results"):
            gr.HTML(class_cards())
            gr.Dataframe(value=pd.DataFrame(per_class).T.reset_index().rename(columns={"index": "Class"}),
                         label="Test split, per class")
            gr.HTML('<p class="note">Below is the model\'s own detection confusion matrix. Its background column counts boxes '
                    'drawn at a confidence of only 0.001, so it is not a count of real false alarms.</p>')
            gr.Image(value=str(R / "confusion_matrix.png"), label="Detection confusion matrix (counts)", interactive=False)

    for ctrl in (lo, hi):
        ctrl.change(band_view, [lo, hi], [kpis, hist, matrix])
        ctrl.change(cost_view, [lo, hi], [c1, c2])
    demo.load(band_view, [lo, hi], [kpis, hist, matrix])
    demo.load(cost_view, [lo, hi], [c1, c2])

demo.launch(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=CSS)

