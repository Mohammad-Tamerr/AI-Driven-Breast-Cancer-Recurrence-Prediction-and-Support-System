# AI-Driven Breast Cancer Recurrence Prediction & Patient Support Ecosystem 🚀

-------------------------------------------------------------------------------

Table of Contents
-----------------

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
	- [Prerequisites](#prerequisites)
	- [Installation](#installation)
- [Privacy & Security](#privacy--security)
- [Contributors](#contributors)
- [Next Steps](#next-steps)

-------------------------------------------------------------------------------

Overview
--------

This repository is a production-minded research codebase that implements a full-stack ecosystem to predict breast cancer recurrence and provide a patient-centric support platform. The project pairs high-performing machine learning models (Random Forest / XGBoost achieving ~95.14% accuracy on internal validation and externally validated against Baheya Center and global datasets) with a retrieval-augmented generation (RAG) medical assistant — "Rafiq / رفيق" — that is context-aware, multimodal-capable (vision + text), and tailored for clinical workflows.

This ecosystem was designed for research, demonstrators, and academic portfolios while following engineering practices that facilitate a transition to production (modular microservices, secure storage, audit logging, and configurable providers for LLMs and vector stores).

-------------------------------------------------------------------------------

System Architecture
-------------------

ASCII Diagram (high-level)

```
												 +----------------------+
												 |      Clients         |
												 | (Browser / Mobile)   |
												 +----------+-----------+
																		|
																		v
												 +----------------------+
												 |     API Gateway      |
												 |  (Auth, Rate-limit)  |
												 +----------+-----------+
																		|
																		v
			+---------------------+   +----+-----+   +----------------------+
			|  Microservices      |<->|  AI/ML   |<->|   Vector DB / File   |
			|  - Auth Service     |   | Layer    |   |   Storage (Qdrant)   |
			|  - User / Profile   |   | - RAG    |   |  + Object Storage    |
			|  - Booking Service  |   | - Vision |   |  (S3-compatible)     |
			|  - Medication Svc   |   | - Models |   +----------------------+
			+---------------------+   +----+-----+
																		|
																		v
												 +----------------------+
												 |  Logging & Monitoring|
												 |    (Audit, SSO)      |
												 +----------------------+
```

Notes:
- Clients communicate through the API Gateway which routes requests to microservices.
- The AI/ML Layer hosts the recurrence prediction engine, embedding services, and multimodal vision inference (Gemini 1.5 Flash where available).
- Vector DBs (Qdrant by default) store embeddings and chunked documents for RAG retrieval.

-------------------------------------------------------------------------------

Key Features
------------

- Prediction Engine (ML):
	- Random Forest and XGBoost models trained for recurrence prediction.
	- Achieves ~95.14% accuracy on held-out validation and validated on external cohorts (Baheya Center, METABRIC, Breast MSK 2018, TCIA).
	- Supports model export, scoring, calibration, and explainability (SHAP analyses in notebooks).

- RAG Medical Chatbot — "Rafiq / رفيق":
	- Document chunking + dense retrieval with configurable vector providers (in-memory, Qdrant).
	- Embedding pipelines with semantic search and prompt templates for context-aware responses.
	- Multimodal vision analysis for lab reports and imaging summaries using vision-capable models.

- Patient Support Ecosystem:
	- Smart Medicine Plan: active-ingredient-level drug-drug interaction checks and alerts.
	- Therapy Session Booking: online scheduling and GPS-based local support group discovery.
	- Nutrition Personalization: rule- and model-based dietary plans tied to clinical status.
	- Moderated Community Forum: patient & survivor groups with moderation controls and reporting.
	- Doctor Verification & Points: practitioner verification workflow and gamified reputation system.
	- Family/Caregiver Portal & Payments: controlled access, donation/NGO integration, and transaction hooks.

- Observability & Safety:
	- Audit logging for sensitive operations.
	- Role-based access control for clinician vs patient flows.

-------------------------------------------------------------------------------

Repository Structure
--------------------

- `data/` — sample patient JSON and small anonymized datasets.
- `notebooks/` — exploratory analysis, model training, and evaluation notebooks.
- `RafeekBot/` — lightweight prototype chat app and demo UI (Flask/HTML templates).
- `rag_chatbot/` — modular RAG backend (controllers, providers, routes, vector store adapters).
- `scripts/` — utility scripts (indexing, end-to-end chat tests, quick verification).
- `evaluation/` — evaluation tools and metric calculators.
- `docs/` — design notes, model card, and evaluation reports.

Refer to implementation files for details and to extend components.

-------------------------------------------------------------------------------

Tech Stack
----------

- Languages: Python 3.8+ (core), JavaScript (UI glue)
- Frameworks: Flask / FastAPI (microservices), scikit-learn, XGBoost, PyTorch/TensorFlow (optional)
- Vector DB: Qdrant (default), in-memory provider for testing
- LLM / Vision: Configurable providers (OpenAI, Gemini, local LLMs) — `rag_chatbot` has provider factories
- Storage: S3-compatible object store for documents, relational DB for metadata (Postgres recommended)
- DevOps: Docker, Docker Compose for local orchestration
- Security: TLS for transport, AES-256 encryption for data at rest (see Privacy & Security)

-------------------------------------------------------------------------------

Getting Started
---------------

Prerequisites

- Python 3.8+ installed
- Docker & Docker Compose (recommended for Qdrant and local stacks)
- Optional: API keys for LLM/vision providers (set via environment variables)

Installation (local dev)

```bash
# Clone the repo
git clone https://github.com/Mohammad-Tamerr/AI-Driven-Breast-Cancer-Recurrence-Prediction-and-Support-System.git
cd AI-Driven-Breast-Cancer-Recurrence-Prediction-and-Support-System

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core requirements (example)
pip install -r RafeekBot/requirements.txt
pip install -r rag_chatbot/src/requirement.txt || true

# (Optional) Start dependent services
docker-compose up -d

# Index sample documents and run the demo
python scripts/index_patients_to_qdrant.py
python RafeekBot/app.py
```

Notes:
- Environment-specific configuration is managed via environment variables. See `rag_chatbot/src/helpers/config.py`.
- Use the in-memory providers for offline experimentation.

-------------------------------------------------------------------------------

Privacy & Security
------------------

This repository is built with sensitive healthcare data considerations in mind. If you plan to use or deploy this system with real patient data, follow local and international regulations (e.g., HIPAA in the U.S., GDPR in the EU) and institutional policies.

- Data Protection:
	- Encryption in transit: TLS 1.2+ for all client-server communication.
	- Encryption at rest: AES-256 for stored PHI (recommended for object storage and DBs).
	- Minimum necessary: follow data minimization and de-identification before ingesting records.

- Access Control & Audit:
	- Role-based access control (RBAC) for patient, caregiver, clinician, and admin roles.
	- Audit logs for read/write operations on protected resources.

- Model & Inference Safety:
	- RAG responses must be used with clinical oversight — include model confidence and provenance (source chunks) in responses.
	- Never surface raw PHI to unverified users. Use strict verification for clinician-level answers.

- Compliance:
	- This codebase is research-grade and not certified for clinical deployment out-of-the-box — a formal security and compliance review is required before production use.

-------------------------------------------------------------------------------

Contributors
------------

- Maintainer: Mohammad Tamerr (@Mohammad-Tamerr)

If you'd like to contribute, please open an issue or a pull request. For formal contributions, add a short description of the change, tests, and any data-update scripts.

-------------------------------------------------------------------------------

Next Steps (suggested)
----------------------

- Add CI checks and unit/integration tests.
- Add `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` for open collaboration.
- Pin dependencies and provide `requirements.txt` per component.
- Add model cards and data sheets for transparency.

-------------------------------------------------------------------------------

Thank you for reviewing this project. If you want, I can add badges, a `CONTRIBUTING.md`, or pin dependencies next — tell me which.
