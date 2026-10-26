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

# In[1]:


import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

R = Path("results")
model = YOLO("weights/best.pt")                      # committed weights, never runs/
scores = pd.read_csv(R / "test_scores.csv")          # one row per test photo
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
ZONES = ["Pass", "Review", "Fail"]


def band(lo, hi):
    hi = max(hi, lo)
    zone = np.where(s < lo, "Pass", np.where(s >= hi, "Fail", "Review"))
    rows, share = [], {}
    for label, val in [("Damaged car (true)", 1), ("Clean car (true)", 0)]:
        z = zone[y == val]
        rows.append([label] + [int((z == k).sum()) for k in ZONES])
        share[val] = [float((z == k).mean()) for k in ZONES]
    table = pd.DataFrame(rows, columns=["True class", "Pass (no person)", "Review (a person)", "Fail (no person)"])
    pp, pr, pf = share[1]
    cp, cr, cf = share[0]
    cost = 1000 * (BASE * (pp * FN + pr * REV) + (1 - BASE) * (cr * REV + cf * FPA))
    review = BASE * pr + (1 - BASE) * cr
    per_day = DAY * review
    hours = per_day * MINS / 60
    text = (
        f"### Expected cost per 1,000 returns: EUR {cost:,.0f}\n"
        f"- **Share sent to a person:** {review:.1%}  ({per_day:.0f} reviews a day at {DAY} returns, "
        f"{hours:.1f} hours, about {hours / HRS:.2f} people)\n"
        f"- **Damaged cars that slip through:** {1000 * BASE * pp:.1f} per 1,000 returns\n"
        f"- **Clean cars wrongly failed:** {1000 * (1 - BASE) * cf:.1f} per 1,000 returns\n\n"
        f"Costs assume {BASE:.0%} of returns are damaged, a missed damage costs EUR {FN}, "
        f"a review EUR {REV}, a wrong automatic charge EUR {FPA}. The table counts the "
        f"{int((y == 1).sum())} damaged and {int((y == 0).sum())} clean test photos."
    )
    return table, text


def predict(img, lo, hi):
    if img is None:
        return None, "Upload a photo first."
    hi = max(hi, lo)
    bgr = img[..., ::-1]                              # Gradio gives RGB, the model expects BGR
    allbox = model.predict(bgr, conf=0.001, imgsz=640, verbose=False)[0]
    damage = [float(c) for c, k in zip(allbox.boxes.conf, allbox.boxes.cls) if int(k) == 0]
    score = max(damage, default=0.0)
    shown = model.predict(bgr, conf=max(lo, 0.01), imgsz=640, verbose=False)[0].plot()[..., ::-1]
    decision = "PASS (no person needed)" if score < lo else "FAIL (damaged, no person needed)" if score >= hi else "SEND TO A PERSON"
    return shown, f"### Damage score {score:.2f}  ->  {decision}"


def curve_single():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cost_sweep["threshold"], cost_sweep["cost_per_1000"], color="#1f4e79")
    t = single["chosen_threshold"]
    ax.scatter([t], [single["cost_per_1000_at_chosen"]], color="red", zorder=3, label=f"chosen single threshold {t:.2f}")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="provisional 0.5")
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Cost per 1,000 returns (EUR)")
    ax.set_title("Cost vs threshold (one threshold, every flagged car reviewed)")
    ax.legend(); fig.tight_layout()
    return fig


def curve_review():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tradeoff["max_review_share"] * 100, tradeoff["min_cost_per_1000"], color="#1f4e79")
    ax.scatter([policy["review_share"] * 100], [policy["cost_per_1000"]], color="red", zorder=3,
               label=f"chosen band {policy['t_low']:.2f} to {policy['t_high']:.2f}")
    ax.set_xlabel("Maximum share of returns sent to a person (%)"); ax.set_ylabel("Lowest cost per 1,000 returns (EUR)")
    ax.set_title("What it costs to automate more")
    ax.legend(); fig.tight_layout()
    return fig


pc = pd.DataFrame(per_class).T.reset_index().rename(columns={"index": "Class"})
pc["Class"] = pc["Class"].replace({"Damage-detected": "Damage-detected (positive)"})

with gr.Blocks(title="Returned-car damage check") as demo:
    gr.Markdown("# Returned-car damage check\nPositive class: **Damage-detected**. "
                "Scores below the low edge pass, scores from the high edge up fail, and a person reviews the cars in between.")
    lo = gr.Slider(0, 1, value=policy["t_low"], step=0.01, label="Low edge: below this score the car passes without a person")
    hi = gr.Slider(0, 1, value=policy["t_high"], step=0.01, label="High edge: from this score up the car fails without a person")
    with gr.Tab("Try a photo"):
        img = gr.Image(type="numpy", label="Car photo")
        btn = gr.Button("Check this car")
        shown = gr.Image(label="Predictions")
        verdict = gr.Markdown()
        btn.click(predict, [img, lo, hi], [shown, verdict])
    with gr.Tab("Band, confusion matrix and cost"):
        table = gr.Dataframe(label="Test photos: true class vs decision")
        summary = gr.Markdown()
        lo.change(band, [lo, hi], [table, summary])
        hi.change(band, [lo, hi], [table, summary])
        demo.load(band, [lo, hi], [table, summary])
    with gr.Tab("Cost curves"):
        gr.Plot(value=curve_single())
        gr.Plot(value=curve_review())
    with gr.Tab("Per-class results"):
        gr.Dataframe(value=pc, label="Test split, per class")
        gr.Markdown("The model's own detection confusion matrix. Its 'background' column counts boxes drawn at a "
                    "confidence of only 0.001, so it is not a count of real false alarms.")
        gr.Image(value=str(R / "confusion_matrix.png"), label="Confusion matrix (counts)", interactive=False)

demo.launch()

