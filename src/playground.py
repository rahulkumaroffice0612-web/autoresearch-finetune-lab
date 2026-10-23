"""LazyDecider Playground — a local Gradio app that runs the real 0.8B decision
model on a user-supplied invoice scenario.

The model (Qwen3.5-0.8B + Clef joint-decision head) is the MLX build in
artifacts/best_model, loaded via Unsloth's FastDecisionModel. On Apple Silicon it
runs on the MLX backend (fast, free, local). It reads an invoice `state` and a set
of `questions` and returns calibrated probability distributions per decision.

Run it:
  .venv/bin/python src/playground.py            # http://127.0.0.1:7860
  .venv/bin/python src/playground.py --share     # also a temporary public link

Note: this is the MLX (Apple-Silicon) build, so run it on a Mac. See REPORT.md for
the portable-weights path if you need x86/Linux.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT / "artifacts" / "best_model")

# The five accounts-payable decisions this domain asks, with the dataset's own criteria.
QUESTIONS = {
    "discrepancy_severity": {
        "instructions": "How material is any discrepancy between the invoice, the order and the delivery?",
        "type": "score",
        "criteria": [
            "None: everything reconciles.",
            "Trivial: rounding or a cosmetic difference.",
            "Moderate: a real difference worth confirming.",
            "Material: a large or unexplained difference.",
        ],
    },
    "disposition": {
        "instructions": "How should this vendor invoice be dispositioned?",
        "type": "choice",
        "criteria": {
            "approve": "Matches the order and delivery; approve for payment.",
            "hold": "Something needs confirming before payment; hold pending clarification.",
            "manual_review": "A human in finance must review the discrepancy.",
            "reject": "Should not be paid: duplicate, unauthorised or materially wrong.",
        },
    },
    "duplicate": {
        "instructions": "This invoice appears to duplicate an invoice already submitted.",
        "type": "noul",
    },
    "matches_order": {
        "instructions": "The invoice reconciles with the order and delivery.",
        "type": "noul",
        "criteria": {
            "false": "There is a discrepancy against the order or the delivery.",
            "true": "Line items, quantities and amounts reconcile.",
        },
    },
    "urgency": {
        "instructions": "How urgent is acting on this invoice?",
        "type": "score",
        "criteria": [
            "None: no time pressure.",
            "Low: routine.",
            "Moderate: a deadline or discount is approaching.",
            "High: overdue or a discount expires imminently.",
        ],
    },
}

_MODEL = None
_TOK = None


def _load():
    global _MODEL, _TOK
    if _MODEL is None:
        from unsloth import FastDecisionModel

        _MODEL, _TOK = FastDecisionModel.from_pretrained(
            MODEL_PATH, max_seq_length=1280, load_in_4bit=False,
        )
    return _MODEL, _TOK


def build_state(vendor, invoice_total, po_total, received_qty, ordered_qty,
                days_until_due, discount_days, terms, disputes_12m, invoices_12m, condition):
    unit = round(invoice_total / ordered_qty, 2) if ordered_qty else invoice_total
    po_unit = round(po_total / ordered_qty, 2) if ordered_qty else po_total
    return {
        "delivery": {"condition": condition, "received_qty": received_qty},
        "invoice": {
            "currency": "USD", "id": "INV-USER-0001", "vendor": vendor,
            "total_usd": invoice_total,
            "lines": [{"sku": "SKU-001", "qty": ordered_qty, "unit_usd": unit, "total_usd": invoice_total}],
        },
        "payment": {
            "terms": terms, "status": "scheduled",
            "days_until_due": days_until_due,
            "discount_expires_in_days": discount_days,
            "early_payment_discount_pct": 2.0 if discount_days > 0 else 0.0,
        },
        "purchase_order": {
            "id": "PO-USER-0001", "total_usd": po_total,
            "lines": [{"qty": ordered_qty, "unit_usd": po_unit}],
            "freight_terms": "freight prepaid by vendor, not separately billable",
        },
        "vendor_history": {"disputes_12m": disputes_12m, "invoices_12m": invoices_12m, "prior_invoice_ids": []},
    }


def predict(vendor, invoice_total, po_total, received_qty, ordered_qty,
            days_until_due, discount_days, terms, disputes_12m, invoices_12m, condition):
    try:
        model, tok = _load()
    except Exception as e:  # pragma: no cover
        return f"### ⚠️ Model failed to load\n```\n{e}\n```", None

    state = build_state(vendor, invoice_total, po_total, received_qty, ordered_qty,
                        days_until_due, discount_days, terms, disputes_12m, invoices_12m, condition)

    from unsloth import FastDecisionModel
    answers = FastDecisionModel.predict(model, tok, state, QUESTIONS)

    # Headline: the disposition call.
    disp = answers.get("disposition", {}).get("probabilities", {})
    top = max(disp, key=disp.get) if disp else "—"
    conf = max(disp.values()) if disp else 0.0
    md = [f"## Decision: **{top.replace('_', ' ')}**  ·  {conf*100:.0f}% confident", ""]

    rows = {q: answers.get(q, {}).get("probabilities", {}) for q in QUESTIONS}
    # A tidy table of all five decisions.
    table = {"Decision": [], "Top choice": [], "Confidence": [], "Full distribution": []}
    labels = {"discrepancy_severity": "Discrepancy severity", "disposition": "Disposition",
              "duplicate": "Duplicate?", "matches_order": "Matches order?", "urgency": "Urgency"}
    for q, probs in rows.items():
        if not probs:
            continue
        t = max(probs, key=probs.get)
        table["Decision"].append(labels.get(q, q))
        table["Top choice"].append(t.replace("_", " "))
        table["Confidence"].append(f"{probs[t]*100:.0f}%")
        dist = "  ".join(f"{k}:{v:.2f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1]))
        table["Full distribution"].append(dist)
    return "\n".join(md), table


PRESETS = {
    "Clean match (should approve)": ["Acme Metals", 1136.50, 1136.50, 50, 50, 27, 3, "net 45", 1, 33, "accepted"],
    "Over-invoiced +12% (review)": ["Vantage Materials", 131836.50, 117188.00, 100, 100, 20, 0, "net 30", 2, 18, "accepted with exceptions"],
    "Under-delivered -50% (discrepancy)": ["Globex Supply", 1670.45, 3340.90, 25, 50, -9, 0, "net 30", 2, 12, "accepted with exceptions"],
}


def load_preset(name):
    return PRESETS[name]


with gr.Blocks(title="LazyDecider Playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🦥 LazyDecider Playground\n"
        "Run the **real 0.8B agent-decision model** on your own invoice. It reads the invoice against "
        "its purchase order and delivery, then makes the accounts-payable call as calibrated "
        "probabilities. Model: [`rahulkumar0612/lazydecider-0.8b`](https://huggingface.co/rahulkumar0612/lazydecider-0.8b) "
        "· [project repo](https://github.com/rahulkumaroffice0612-web/autoresearch-finetune-lab)."
    )
    with gr.Row():
        with gr.Column(scale=1):
            preset = gr.Dropdown(list(PRESETS), label="Load an example", value=None)
            vendor = gr.Textbox("Acme Metals", label="Vendor")
            with gr.Row():
                invoice_total = gr.Number(1136.50, label="Invoice total (USD)")
                po_total = gr.Number(1136.50, label="PO total (USD)")
            with gr.Row():
                ordered_qty = gr.Number(50, label="Ordered qty", precision=0)
                received_qty = gr.Number(50, label="Received qty", precision=0)
            with gr.Row():
                days_until_due = gr.Number(27, label="Days until due (neg = overdue)", precision=0)
                discount_days = gr.Number(3, label="Discount expires in (days)", precision=0)
            terms = gr.Dropdown(["net 15", "net 30", "net 45", "net 60"], value="net 45", label="Payment terms")
            with gr.Row():
                disputes_12m = gr.Number(1, label="Vendor disputes (12m)", precision=0)
                invoices_12m = gr.Number(33, label="Vendor invoices (12m)", precision=0)
            condition = gr.Dropdown(
                ["accepted", "accepted with exceptions", "partial", "rejected"],
                value="accepted", label="Delivery condition")
            btn = gr.Button("Decide", variant="primary")
        with gr.Column(scale=1):
            headline = gr.Markdown("### Enter a scenario and press **Decide**.")
            table = gr.Dataframe(headers=["Decision", "Top choice", "Confidence", "Full distribution"],
                                 label="All five decisions", wrap=True)

    inputs = [vendor, invoice_total, po_total, received_qty, ordered_qty,
              days_until_due, discount_days, terms, disputes_12m, invoices_12m, condition]
    preset.change(load_preset, preset, inputs)
    # Show a "working" note immediately, then run the model.
    btn.click(lambda: ("### ⏳ Running the model…", None), None, [headline, table]).then(
        predict, inputs, [headline, table]
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="also expose a temporary public link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-preload", action="store_true", help="skip loading the model at startup")
    args = ap.parse_args()

    # Load the model up front so the FIRST "Decide" click is instant instead of
    # silently blocking ~15-20s. (The server is already responsive; this just
    # moves the one-time load to startup, with a clear console message.)
    if not args.no_preload:
        print("[playground] preloading model (one-time, ~15-20s on Apple Silicon)...", flush=True)
        try:
            _load()
            print("[playground] model ready — opening the UI.", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"[playground] WARN: preload failed ({e}); will load on first click.", flush=True)

    demo.launch(share=args.share, server_port=args.port)
