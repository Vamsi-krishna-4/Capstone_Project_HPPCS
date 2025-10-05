# main.py
"""
Main (robust):
- Generates synthetic demo data (n cases)
- Optionally trains projection heads (robust to failures)
- Always proceeds to run inference + generate outputs even if training fails
- Uses two LLMs: flan-t5-small (explanations) and distilbart (short tag)
- Writes conversation_1.json ... conversation_n.json
- Writes Report.pdf if reportlab available; else writes Report.txt and notifies user
"""

import argparse
import json
from datetime import datetime
import torch
import os
import traceback

from demo_data import generate_demo_pairs
from fusion_pipeline import FusionPipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_explanation(flan_model, flan_tokenizer, note, sim_score, device):
    prompt = (f"You are a concise medical assistant. Clinical note:\n{note}\n"
              f"Fusion similarity score: {sim_score:.3f}\n\n"
              "In 2-4 short sentences say whether the image supports the note, likely diagnoses, and one next diagnostic step.")
    try:
        inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_ids = flan_model.generate(**inputs, max_length=120)
        return flan_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[main] Warning: flan generation failed: {e}")
        return "Explanation unavailable due to generation error."

def generate_diag_tag(bart_model, bart_tokenizer, note, sim_score, device):
    prompt = (f"Given the clinical note:\n{note}\nSimilarity score: {sim_score:.3f}\n"
              "Provide a single short diagnostic tag (one or two words): e.g., 'pneumonia', 'pneumothorax', 'PE', 'CHF', 'COPD'.")
    try:
        inputs = bart_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        # enforce safe generation params to avoid min_length > max_length warnings
        with torch.no_grad():
            out_ids = bart_model.generate(**inputs, min_length=1, max_new_tokens=10)
        tag = bart_tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return tag.strip().split("\n")[0]
    except Exception as e:
        print(f"[main] Warning: diag tag generation failed: {e}")
        return "tag_unavailable"

def write_report_text(report_text, filename="Report.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[main] Saved report text to {filename}")
        return filename
    except Exception as e:
        print(f"[main] Error writing report text: {e}")
        return None

def write_report_pdf(report_text, filename="Report.pdf"):
    """
    Try to write PDF using reportlab. If not available, write text file instead.
    Returns the path of the file created and a boolean indicating whether it's a PDF.
    """
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
        print(f"[main] PDF report saved to {filename}")
        return filename, True
    except Exception as e:
        print(f"[main] reportlab not available or failed: {e}; falling back to text.")
        txt = filename.replace(".pdf", ".txt")
        path = write_report_text(report_text, filename=txt)
        return path, False

def build_report_text(outputs):
    header = "Multimodal Medical Assistant — Prototype (HPPCS[04])\n\n"
    abstract = ("Abstract: Prototype fusing clinical notes and chest X-rays to generate a similarity score, "
                "concise explanation, and short diagnostic tag. Uses MiniLM+CLIP encoders and a small learned projection "
                "head trained contrastively. Explanations by flan-t5-small; diagnostic tag by distilbart.\n\n")
    methods = ("Methods: Text encoder: sentence-transformers/all-MiniLM-L6-v2. Image encoder: openai/clip-vit-base-patch32. "
               "Projection: small MLP trained with InfoNCE (only projection trained). Explanations: flan-t5-small; tag: distilbart.\n\n")
    results = "Results:\n"
    for o in outputs:
        results += (f"Case {o['case_id']}: similarity={o['similarity_score']:.3f} | tag={o.get('diagnosis_tag','NA')}\n"
                    f"Explanation: {o['explanation']}\n\n")
    conclusion = ("Conclusion: Produced 5 conversation JSON files containing image, clinical note, similarity score, explanation, "
                  "diagnostic tag and timestamp. For production: fine-tune encoders on paired clinical-image datasets and add metrics.\n")
    return header + abstract + methods + results + conclusion

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[main] device:", device)

    # Step 1: generate demo pairs
    pairs = generate_demo_pairs(n=args.cases)
    print(f"[main] Generated {len(pairs)} demo pairs.")

    # Step 2: initialize fusion
    fusion = FusionPipeline()

    # Step 3: train projection (guarded)
    trained_ok = False
    if args.train:
        try:
            fusion.train_projection(pairs, epochs=args.epochs, batch_size=args.batch_size, save=True)
            # check if weights file saved and non-empty
            weights_path = "proj_weights.pt"
            if os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
                trained_ok = True
        except Exception as e:
            print("[main] Warning: training projection raised exception:", e)
            traceback.print_exc()
            trained_ok = False

    if not trained_ok:
        print("[main] Proceeding to inference (no trained projection or training skipped).")
        fusion.load_projection()  # load if available; if not, proceed with random proj heads

    # Step 4: load LLMs (guarded)
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

    outputs = []
    for idx, (img_path, note) in enumerate(pairs, start=1):
        print(f"[main] Processing case {idx}/{len(pairs)} -> {img_path}")
        sim = fusion.fuse(note, img_path)
        explanation = "explanation_unavailable"
        diag_tag = "tag_unavailable"
        if flan_model and flan_tok:
            explanation = generate_explanation(flan_model, flan_tok, note, sim, device)
        else:
            print("[main] flan model unavailable; skipping explanation generation.")
        if bart_model and bart_tok:
            diag_tag = generate_diag_tag(bart_model, bart_tok, note, sim, device)
        else:
            print("[main] bart model unavailable; skipping diagnostic tag generation.")

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

    # Step 5: create report (PDF if possible)
    report_text = build_report_text(outputs)
    report_path, is_pdf = write_report_pdf(report_text, filename="./Report.pdf")
    if is_pdf:
        print(f"[main] Report PDF created at {report_path}. Move to Report/ before zipping per instructions.")
    else:
        print(f"[main] PDF not created; fallback saved to {report_path}. Move appropriate file to Report/ before zipping.")

    print("[main] Done. Conversation files and report (or fallback) are in current folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=lambda v: v.lower() in ("true", "1", "yes"), default=True,
                        help="Train projection heads on demo pairs (recommended).")
    parser.add_argument("--cases", type=int, default=5, help="Number of synthetic cases to generate.")
    parser.add_argument("--epochs", type=int, default=6, help="Projection training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Projection training batch size.")
    args = parser.parse_args()
    main(args)
