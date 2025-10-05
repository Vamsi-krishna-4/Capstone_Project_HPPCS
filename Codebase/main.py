# main.py
"""
Multimodal Medical Assistant — Main entrypoint (HPPCS[04])

This script:
- Generates synthetic demo data (N cases) via demo_data.generate_demo_pairs()
- Optionally trains small projection heads (contrastive) in FusionPipeline
- For each case produces: conversation_{i}.json with fields:
    case_id, image, clinical_note, similarity_score, explanation, diagnosis_tag, generated_at
- Writes a concise, text-only Report to ../Report/Project_Report.pdf (creates the directory)
- Uses two LLMs:
    - google/flan-t5-small (explanations)
    - sshleifer/distilbart-cnn-12-6 (diagnostic tag)
- Includes robust fallbacks: rule-based tagging, LLM validation, and templated fallbacks
- All files live in the same Codebase directory (no nested subdirectories), and the report is saved to ../Report/
"""

import argparse
import json
import os
import re
import traceback
from datetime import datetime

import torch

# Local modules (must be in same folder)
from demo_data import generate_demo_pairs
from fusion_pipeline import FusionPipeline

# Transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------
# Rule-based tagging utility
# ---------------------------
def rule_based_tag(clinical_note: str):
    """
    Fast deterministic tagger for common diagnoses.
    Returns one of the canonical tags or None if no rule matches.
    Canonical set: pneumonia, pneumothorax, pulmonary_embolism, COPD, CHF, trauma, normal, uncertain
    """
    if not clinical_note or not isinstance(clinical_note, str):
        return None
    s = clinical_note.lower()

    # pneumonia signatures
    if re.search(r"\bpneumonia\b|\bconsolidat(e|ion)\b|\binfiltrat(e|ion)\b|\blobar\b|\bfocal opacity\b", s):
        return "pneumonia"

    # pneumothorax
    if re.search(r"\bpneumothorax\b|\bpleural (?:air|line)\b|\bcollapsed lung\b", s):
        return "pneumothorax"

    # pulmonary embolism hints
    if re.search(r"\bpulmonary emboli|pulmonary embolism|pe\b|\bdd[- ]?imer|\bpleuritic\b|\bsudden (?:dyspnea|shortness)\b", s):
        return "pulmonary_embolism"

    # COPD
    if re.search(r"\bcopd\b|\bchronic obstructive\b|\bchronic cough\b|\bsmok", s):
        return "COPD"

    # CHF / heart failure
    if re.search(r"\bcongestive heart failure\b|\bchf\b|\bcardiomegaly\b|\bpulmonary (?:edema|congestion)\b|\bledema\b", s):
        return "CHF"

    # trauma
    if re.search(r"\btrauma\b|\bblunt chest\b|\bfracture\b|\baccident\b", s):
        return "trauma"

    # normal / clear
    if re.search(r"\bnormal\b|\bclear (lungs|chest)\b|\bno acute\b|\bno evidence of\b", s):
        return "normal"

    return None

# ---------------------------
# Robust LLM-based tag generation
# ---------------------------
def generate_diag_tag(bart_model, bart_tokenizer, note, sim_score, device):
    """
    Try rule-based tag first. If None, call the LLM with a constrained, few-shot prompt,
    deterministic beam decoding, validate and map output to canonical tags, otherwise fallback to 'uncertain'.
    """
    # 1) rule-based quick path
    tag = rule_based_tag(note)
    if tag:
        return tag

    # 2) LLM fallback (few-shot)
    prompt = (
        "You are an assistant that returns exactly one tag for a clinical note. "
        "Allowed tags: pneumonia, pneumothorax, pulmonary_embolism, COPD, CHF, trauma, normal, uncertain.\n\n"
        "Examples:\n"
        "Note: 45-year-old male with productive cough and lobar consolidation on X-ray. Tag: pneumonia\n"
        "Note: 30-year-old after trauma with sudden chest pain and visible pleural line. Tag: pneumothorax\n"
        "Note: 70-year-old with leg swelling, cardiomegaly and pulmonary congestion. Tag: CHF\n\n"
        f"Tag this note from the allowed list. Clinical note: {note}\nSimilarity score: {sim_score:.3f}\nTag:"
    )
    try:
        inputs = bart_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_ids = bart_model.generate(
                **inputs,
                min_length=1,
                max_new_tokens=6,
                do_sample=False,
                num_beams=3,
                early_stopping=True
            )
        raw = bart_tokenizer.decode(out_ids[0], skip_special_tokens=True).strip().lower()
        candidate = raw.split()[0] if raw else ""
        candidate = "".join(ch for ch in candidate if ch.isalnum() or ch == "_")

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
        tag = mapping.get(candidate, None)
        if tag is None:
            for k in mapping:
                if k in raw:
                    tag = mapping[k]; break
        if not tag:
            # reject very-short / numeric / nonsense outputs
            if len(candidate) == 0 or candidate.isdigit() or len(candidate) < 2:
                tag = "uncertain"
            else:
                tag = candidate[:32]
        return tag
    except Exception as e:
        print(f"[main] Warning: diag tag LLM failed: {e}")
        return "uncertain"

# ---------------------------
# Robust explanation generation
# ---------------------------
def generate_explanation(flan_model, flan_tokenizer, note, sim_score, device):
    """
    Few-shot explanation generator:
    - Uses small deterministic beam decoding
    - Validates output and falls back to templated messages when needed
    """
    prompt = (
        "You are a concise clinical assistant. For each clinical note and numeric similarity score produce "
        "2 short sentences: 1) whether the image supports the note (yes/no/partially) with a brief reason; "
        "2) one recommended next diagnostic step. Do NOT repeat the clinical note.\n\n"
        "Example 1:\nNote: 45-year-old with productive cough and focal lobar consolidation on X-ray.\nScore: 0.35\n"
        "Answer: The image supports lobar pneumonia with focal consolidation. Recommend sputum culture and chest radiography follow-up.\n\n"
        "Example 2:\nNote: sudden pleuritic chest pain after trauma, concern for pneumothorax.\nScore: 0.48\n"
        "Answer: The image suggests a small pneumothorax at the apex which supports the concern. Recommend immediate bedside ultrasound or upright chest X-ray.\n\n"
        f"Now answer this note and score in max 2 short sentences.\nNote: {note}\nScore: {sim_score:.3f}\nAnswer:"
    )
    try:
        inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_ids = flan_model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                num_beams=3,
                early_stopping=True
            )
        raw = flan_tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        cleaned = " ".join(raw.split())
        # Basic validation: must be longer than a trivial token and not numeric-only
        if len(cleaned) < 4 or re.fullmatch(r"[\d\W_]+", cleaned):
            # fallback: rule-based templated explanation if available
            fb_tag = rule_based_tag(note)
            if fb_tag and fb_tag != "uncertain":
                fb_map = {
                    "pneumonia": "Image supports pneumonia (focal consolidation). Recommend sputum culture and chest radiography follow-up.",
                    "pneumothorax": "Image suggests pneumothorax; recommend immediate bedside ultrasound or upright chest X-ray.",
                    "CHF": "Image suggests cardiogenic congestion; recommend BNP measurement and echocardiography.",
                    "COPD": "Imaging suggests chronic obstructive changes; recommend spirometry and clinical correlation.",
                    "pulmonary_embolism": "Chest X-ray is often non-diagnostic for PE; recommend CT pulmonary angiography if clinically indicated.",
                    "trauma": "Image shows traumatic findings; recommend urgent CT and surgical consultation as indicated.",
                    "normal": "No acute radiographic abnormality seen; correlate clinically."
                }
                return fb_map.get(fb_tag, f"Similarity score: {sim_score:.3f}. Recommend clinical correlation.")
            return f"Similarity score: {sim_score:.3f}. Recommend clinical correlation and further imaging as appropriate."
        return cleaned
    except Exception as e:
        print(f"[main] Warning: explanation LLM failed: {e}")
        fb_tag = rule_based_tag(note)
        if fb_tag and fb_tag != "uncertain":
            return f"Based on note, likely {fb_tag}. Recommend relevant diagnostic step."
        return f"Similarity score: {sim_score:.3f}. Explanation unavailable."

# ---------------------------
# Report utilities
# ---------------------------
def build_report_text(outputs):
    """
    Build concise text-only report (<= ~3 pages).
    The outputs param is the list of result dicts produced for each case.
    """
    header = "Multimodal Medical Assistant — Prototype (HPPCS[04])\n\n"
    abstract = ("Abstract: Prototype fusing clinical notes and chest X-rays to generate a similarity score, "
                "concise explanation, and a short diagnostic tag for each case. Uses MiniLM+CLIP encoders, "
                "a small projection head trained contrastively, flan-t5-small for explanations, and distilbart for tagging.\n\n")
    methods = ("Methods: Text encoder: sentence-transformers/all-MiniLM-L6-v2. "
               "Image encoder: openai/clip-vit-base-patch32. Projection: small MLP trained with InfoNCE (only projection trained). "
               "Explanations: flan-t5-small. Tagging: sshleifer/distilbart-cnn-12-6.\n\n")
    results = "Results:\n"
    for o in outputs:
        results += (f"Case {o['case_id']}: similarity = {o['similarity_score']:.3f} | tag = {o.get('diagnosis_tag','NA')}\n"
                    f"Explanation: {o['explanation']}\n\n")
    conclusion = ("Conclusion: The system produces JSON conversation files (image, clinical note, similarity score, "
                  "explanation, diagnostic tag, timestamp). For production: fine-tune encoders on real paired clinical-image datasets "
                  "and perform quantitative evaluation with clinician-labeled ground truth.\n")
    return header + abstract + methods + results + conclusion

def write_report_pdf(report_text, filename="../Report/Project_Report.pdf"):
    """
    Attempt to write a PDF to filename using reportlab. If that fails, write a .txt fallback.
    Ensures the parent directory exists.
    Returns (path, is_pdf_bool).
    """
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
        # fallback
        fallback = filename.replace(".pdf", ".txt")
        try:
            with open(fallback, "w", encoding="utf-8") as f:
                f.write(report_text)
            return fallback, False
        except Exception as e2:
            print("[main] Error writing fallback report:", e2)
            return None, False

# ---------------------------
# Main orchestration
# ---------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[main] device:", device)

    # 1) Generate demo pairs (images saved in working dir as ./demo_xray_{i}.png)
    pairs = generate_demo_pairs(n=args.cases)
    print(f"[main] Generated {len(pairs)} demo pairs.")

    # 2) Initialize fusion pipeline (encoders + projection heads)
    fusion = FusionPipeline()

    # 3) Train projection heads (guarded)
    trained_ok = False
    if args.train:
        try:
            fusion.train_projection(pairs, epochs=args.epochs, batch_size=args.batch_size, save=True)
            weights_file = "proj_weights.pt"
            if os.path.exists(weights_file) and os.path.getsize(weights_file) > 0:
                trained_ok = True
        except Exception as e:
            print("[main] Warning: training projection raised exception:", e)
            traceback.print_exc()
            trained_ok = False

    if not trained_ok:
        print("[main] Proceeding to inference (no trained projection or training skipped).")
        fusion.load_projection()

    # 4) Load LLMs (guarded, continue if they fail)
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

    # 5) Process each case: compute similarity, generate explanation & tag, save json
    outputs = []
    for idx, (img_path, note) in enumerate(pairs, start=1):
        print(f"[main] Processing case {idx}/{len(pairs)} -> {img_path}")
        sim = fusion.fuse(note, img_path)

        # explanation
        explanation = "explanation_unavailable"
        if flan_model and flan_tok:
            explanation = generate_explanation(flan_model, flan_tok, note, sim, device)
        else:
            # fallback: templated explanation using rule-based tag if available
            fb = rule_based_tag(note)
            if fb and fb != "uncertain":
                explanation = f"Likely {fb} based on clinical note. Recommend appropriate diagnostic step."
            else:
                explanation = f"Similarity score: {sim:.3f}. Explanation unavailable."

        # diagnostic tag
        diagnosis_tag = "uncertain"
        if bart_model and bart_tok:
            diagnosis_tag = generate_diag_tag(bart_model, bart_tok, note, sim, device)
        else:
            fb = rule_based_tag(note)
            diagnosis_tag = fb if fb else "uncertain"

        # build result
        result = {
            "case_id": idx,
            "image": img_path,
            "clinical_note": note,
            "similarity_score": float(sim),
            "explanation": explanation,
            "diagnosis_tag": diagnosis_tag,
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

    # 6) Build and write the report directly to ../Report/Project_Report.pdf (or fallback ../Report/Report.txt)
    report_text = build_report_text(outputs)
    report_path, is_pdf = write_report_pdf(report_text, filename="../Report/Project_Report.pdf")
    if is_pdf:
        print(f"[main] Report PDF saved to {report_path} (move to submission zip as required).")
    else:
        if report_path:
            print(f"[main] PDF generation failed; fallback text saved to {report_path}. Move into ../Report/ before zipping.")
        else:
            print("[main] Report generation failed entirely.")

    print("[main] Done. Created conversation_*.json files and report (or fallback).")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=lambda v: v.lower() in ("true", "1", "yes"), default=True,
                        help="Train projection heads on demo pairs (recommended)")
    parser.add_argument("--cases", type=int, default=5, help="Number of synthetic cases to generate")
    parser.add_argument("--epochs", type=int, default=6, help="Projection training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Projection training batch size")
    args = parser.parse_args()
    main(args)
