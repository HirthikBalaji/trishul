# TRISHUL — Human Action Items (TODO_HUMAN.md)

This file tracks tasks that require human authorization, external accounts, manual review, or video submission. The automated agent continues building with mocks, unit tests, and synthetic fixtures until these are supplied.

---

## 1. External Accounts & Credentials
- [ ] **Copernicus Data Space Ecosystem (CDSE):**
  - Register free account at [dataspace.copernicus.eu](https://dataspace.copernicus.eu).
  - Add `CDSE_USERNAME` and `CDSE_PASSWORD` (or OAuth `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET`) to your local `.env`.
  - *Note:* The system runs in self-contained mock/cached mode out of the box if credentials are not provided.
- [ ] **LLM API Keys (Optional):**
  - Add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to `.env` if enabling the generative situation report paraphrase copilot.
  - *Note:* The deterministic Claim Ledger template generates 100% grounded English and Nepali situation reports without an external LLM key.

---

## 2. Linguistic & Domain Review
- [ ] **Nepali Situation Report Glossary Review:**
  - Review technical terminology in `trishul/report/nepali.py` (e.g., flood water: बाढीको पानी, debris flow: गेग्रान बहाव, cut-off settlement: सडक सम्पर्क टुटेका बस्तीहरू) with a native speaker of Nepali.

---

## 3. Training & Compute (Optional)
- [ ] **Large-scale Kuro Siwo GPU Training:**
  - A ready-to-run Kaggle/Colab notebook is provided in `notebooks/train_geosar_unet.ipynb`.
  - To train a new checkpoint from scratch on the full Kuro Siwo Webdataset on high-memory GPUs (A100/H100), execute the notebook and place the resulting `geosar_unet_v1.onnx` into `models/`.
  - *Note:* A lightweight distilled ONNX checkpoint and pre-calibrated fusion weights are included in the repo for instant CPU execution.

---

## 4. Devpost Submission & Video
- [ ] **Devpost Team Verification:**
  - Ensure all team members have registered Devpost accounts and are added to the submission page before the deadline.
  - Enter team member full names and affiliations.
- [ ] **Demo Video (3:00 – 3:45):**
  - Record screencast following `docs/video_script.md`.
  - Upload to YouTube as **Unlisted** (or Public, but not Private).
  - Paste YouTube URL into Devpost submission.
