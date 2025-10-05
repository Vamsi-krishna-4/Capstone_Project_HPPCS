# main.py
"""
Main entrypoint for the Multimodal Medical Assistant (HPPCS[04]) prototype.
- Generates synthetic demo data (N cases)
- Optionally trains small projection heads (contrastive) in FusionPipeline
- Generates: conversation_1.json ... conversation_N.json (each contains image, clinical_note,
  similarity_score, explanation, diagnosis_tag, generated_at)
- Writes a text-only Report directly to ../Report/Report.pdf (creates ../Report/ if needed).
Notes:
- Uses two LLMs: google/flan-t5-small (explanations) and sshleifer/distilbart-cnn-12-6 (single-word tags).
- This script is defensive: it handles model load errors and will continue to produce JSONs even if training or models fail.
"""
import argparse
import json
import os
from datetime import datetime
import traceback

import torch

# local modules (must be in same folder)
from demo_data import generate_demo_pairs
from fusion_pipeline import FusionPipeline

# transformers models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# Improved generation helpers (deterministic, constrained, and post-processed)
# ---------------------------------------------------------------------------

def generate_explanation(flan_model, flan_tokenizer, note, sim_score, device):
    """
    Generate a concise 1-3 sentence explanation using flan-t5-small.
    Deterministic decoding (no sampling), short and clinical.
    """
    prompt = (
        "You are a concise clinical assistant. Read the clinical note and the numeric similarity score.\n\n"
        f"Clinical note: {note}\n"
        f"Similarity score: {sim_score:.3f}\n\n"
        "In 2 short sentences (maximum) say: 1) whether the image supports the clinical note (yes/no/partial) and why, "
        "and 2) one next diagnostic step. Do NOT repeat the clinical note text. Keep it short and clinical."
    )
    try:
        inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_ids = flan_model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,    # deterministic
                num_beams=1,
                early_stopping=True
            )
        out = flan_tokenizer.decode(out_ids[0], skip_special_tokens=True)
        out = " ".join(out.strip().split())  # collapse whitespace
        return out
    except Exception as e:
        print(f"[main] Warning: flan generation failed: {e}")
        return "Explanation unavailable."

def generate_diag_tag(bart_model, bart_tokenizer, note, sim_score, device):
    """
    Produce a single short diagnostic tag from a canonical set.
    We constrain output and then normalize/match to canonical tags.
    """
    prompt = (
        "Read the clinical note and similarity score. Output exactly one short diagnostic tag "
        "chosen from: pneumonia, pneumothorax, pulmonary_embolism, COPD, CHF, trauma, normal, uncertain.\n\n"
        f"Clinical note: {note}\nSimilarity score: {sim_score:.3f}\n\n"
        "Respond with exactly one tag (one word or underscore)."
    )
    try:
        inputs = bart_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out_ids = bart_model.generate(
                **inputs,
                min_length=1,
                max_new_tokens=4,
                do_sample=False,
                num_beams=1,
                early_stopping=True
            )
        tag_raw = bart_tokenizer.decode(out_ids[0], skip_special_tokens=True).strip().lower()
        # normalize and map to canonical list
        tag_candidate = tag_raw.split()[0].strip().replace(" ", "_").replace("-", "_")
        mapping = {
            "pneumonia": "pneumonia",
            "pneumothorax": "pneumothorax",
            "pe": "pulmonary_embolism", "pulmonary_embolism": "pulmonary_embolism",
            "pulmonaryembolism": "pulmonary_embolism",
            "copd": "COPD",
            "chf": "CHF",
            "heart_failure": "CHF",
            "trauma": "trauma",
            "normal": "normal",
            "uncertain": "uncertain",
        }
        # strip punctuation
        tag_candidate = "".join(ch for ch in tag_candidate if ch.isalnum() or ch == "_")
        tag = mapping.get(tag_candidate, None)
        if tag is None:
            for k in mapping:
                if k in tag_raw:
                    tag = mapping[k]
                    break
        if tag is None:
            tag = "uncertain" if not tag_candidate else tag_candidate[:32]
        return tag
    except Exception as e:
        print(f"[main] Warning: diag tag generation failed: {e}")
        return "uncertain"

# ---------------------------------------------------------------------------
# Report writer utilities
# ---------------------------------------------------------------------------

def build_report_text(outputs):
    """
    Build a concise text-only report string summarizing methods and the outputs list.
    Keep the report compact (<= 3 pages when rendered at 12pt).
    """
    header = "Multimodal Medical Assistant — Prototype (HPPCS[04])\n\n"
    abstract = ("Abstract: Prototype fusing clinical notes and chest X-rays to generate a similarity score, "
                "concise explanation, and a short diagnostic tag for each case. Uses efficient encoders (MiniLM + CLIP), "
                "a small learned projection head trained with contrastive loss, flan-t5-small for explanations, "
                "and distilbart for tag generation.\n\n")
    methods = ("Methods: Text encoder: sentence-transformers/all-MiniLM-L6-v2. "
               "Image encoder: openai/clip-vit-base-patch32. "
               "Projection: small MLP trained with InfoNCE (only projection trained). "
               "Explanations: flan-t5-small. Tagging: sshleifer/distilbart-cnn-12-6.\n\n")
    results = "Results:\n"
    for o in outputs:
        results += (f"Case {o['case_id']}: similarity = {o['similarity_score']:.3f} | tag = {o.get('diagnosis_tag','NA')}\n"
                    f"Explanation: {o['explanation']}\n\n")
    conclusion = ("Conclusion: The system produces JSON conversation files (image, clinical note, similarity score, "
                  "explanation, diagnostic tag, timestamp). For production: fine-tune encoders on real paired clinical-image datasets "
                  "and perform quantitative evaluation with clinician-labeled ground truth.\n")
    return header + abstract + methods + results + conclusion

def write_report_pdf(report_text, filename="../Report/Report.pdf"):
    """
    Try to create a PDF at filename using reportlab. If not available, write a text fallback at same folder.
    Returns (path, is_pdf_bool).
    """
    # ensure parent folder exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        width, height = A4
        c = canvas.Canvas(filename, pagesize=A4)
        tx = c.beginText(40, height - 40)
        tx.setFont("Helvetica", 11)
        from textwrap import wrap
        for line in report_text.splitlines():
            if not line.strip():
                tx.textLine("")
                continue
            for sub in wrap(line, 100):
                tx.textLine(sub)
        c.drawText(tx)
        c.showPage()
        c.save()
        return filename, True
    except Exception as e:
        # fallback to text
        fallback = filename.replace(".pdf", ".txt")
        try:
            with open(fallback, "w", encoding="utf-8") as f:
                f.write(report_text)
            return fallback, False
        except Exception as e2:
            print("[main] Error writing fallback report:", e2)
            return None, False

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[main] device:", device)

    # 1) Generate synthetic demo pairs (images saved into Codebase root as ./demo_xray_{i}.png)
    pairs = generate_demo_pairs(n=args.cases)
    print(f"[main] Generated {len(pairs)} demo pairs.")

    # 2) Build fusion pipeline
    fusion = FusionPipeline()

    # 3) Optionally train the small projection heads (guarded)
    trained_ok = False
    if args.train:
        try:
            fusion.train_projection(pairs, epochs=args.epochs, batch_size=args.batch_size, save=True)
            # check if weights file exists & non-empty
            w = "proj_weights.pt"
            if os.path.exists(w) and os.path.getsize(w) > 0:
                trained_ok = True
        except Exception as e:
            print("[main] Warning: training projection raised exception:", e)
            traceback.print_exc()
            trained_ok = False

    if not trained_ok:
        print("[main] Proceeding to inference (no trained projection or training skipped).")
        fusion.load_projection()

    # 4) Load LLMs (guarded — continue even on failure)
    flan_model = flan_tok = bart_model = bart_tok = None
    try:
        flan_name = "google/flan-t5-small"
        print(f"[main] loading flan model: {flan_name}")
        flan_tok = AutoTokenizer.from_pretrained(flan_name)
        flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_name).to(device)
        flan_model.eval()
    except Exception as e:
        print(f"[main] Error loading flan model {flan_name}: {e}")
        flan_model = None

    try:
        bart_name = "sshleifer/distilbart-cnn-12-6"
        print(f"[main] loading bart model: {bart_name}")
        bart_tok = AutoTokenizer.from_pretrained(bart_name)
        bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_name).to(device)
        bart_model.eval()
    except Exception as e:
        print(f"[main] Error loading bart model {bart_name}: {e}")
        bart_model = None

    # 5) For each pair produce JSON output
    outputs = []
    for idx, (img_path, note) in enumerate(pairs, start=1):
        print(f"[main] Processing case {idx}/{len(pairs)} -> {img_path}")
        sim = fusion.fuse(note, img_path)
        explanation = "explanation_unavailable"
        diag_tag = "uncertain"

        if flan_model and flan_tok:
            explanation = generate_explanation(flan_model, flan_tok, note, sim, device)
        else:
            print("[main] flan model unavailable; explanation left unavailable.")

        if bart_model and bart_tok:
            diag_tag = generate_diag_tag(bart_model, bart_tok, note, sim, device)
        else:
            print("[main] bart model unavailable; tag set to 'uncertain'.")

        result = {
            "case_id": idx,
            "image": img_path,
            "clinical_note": note,
            "similarity_score": float(sim),
            "explanation": explanation,
            "diagnosis_tag": diag_tag,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
        outname = f"./conversation_{idx}.json"
        try:
            with open(outname, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
            print(f"[main] Saved {outname}")
        except Exception as e:
            print(f"[main] Error saving {outname}: {e}")

        outputs.append(result)

    # 6) Build and write report directly into ../Report/Report.pdf (or fallback ../Report/Report.txt)
    report_text = build_report_text(outputs)
    report_path, is_pdf = write_report_pdf(report_text, filename="../Report/Report.pdf")
    if is_pdf:
        print(f"[main] Report PDF saved to {report_path}")
    else:
        if report_path:
            print(f"[main] PDF generation failed; fallback text saved to {report_path}")
        else:
            print("[main] Report generation failed entirely.")

    print("[main] Done. JSON outputs and report (or fallback) are in the parent Report/ and current folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=lambda v: v.lower() in ("true", "1", "yes"), default=True,
                        help="Train small projection heads on demo pairs (recommended).")
    parser.add_argument("--cases", type=int, default=5, help="Number of synthetic cases to generate.")
    parser.add_argument("--epochs", type=int, default=6, help="Projection training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Projection training batch size.")
    args = parser.parse_args()
    main(args)
