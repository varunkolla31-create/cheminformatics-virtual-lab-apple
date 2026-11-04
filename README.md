# 🔬 Cheminformatics Virtual Laboratory

**SRM Institute of Science and Technology, Tiruchirappalli**  
*Faculty of Engineering and Technology • School of Sciences • Division of Chemistry*

An interactive, multipage computational chemistry and cheminformatics web application built with **Python 3.11** and **Streamlit**, featuring an editorial UI/UX modelled on the **Oura Design System (`DESIGN.md`)**.

---

## ✨ Oura Design System UI/UX Architecture

The interface is engineered around Oura's warm, human-centric editorial design tokens:

| Token Category | Specification | Implementation |
| :--- | :--- | :--- |
| **Canvas & Surface Palette** | Warm Sandstone Cream (`#F7F1E8`) canvas, Elevated Linen (`#FDFBF7`) surface cards, Tonal Stone (`#EFE9DF`) inputs & glassmorphic sidebar | Configured in `.streamlit/config.toml` and injected via custom CSS in `utils.py` |
| **Ink Hierarchy** | Display Ink (`#000000`), Sandstone Body Ink (`#4A4741`), Muted Stone (`#7A756C` / `#A8A5A0`) | High-contrast editorial readability across light mode and **Oura Night Mode** (`#121110`) |
| **Interactive Accent** | Oura Action Blue (`#2A72DE`, hover `#2056A6`) | Primary pill buttons, active navigation states, focus rings, and Plotly chart traces |
| **Typography** | **Display Serif:** `Editorial New` / `Cormorant Garamond` / `Georgia` <br/> **Workhorse Sans:** `AkkuratLL` / `Inter` <br/> **Monospace:** `JetBrains Mono` | Editorial serif headlines paired with Swiss grotesque body text and monospace SMILES strings |
| **Component Geometry** | `8px` border radius on cards, metrics, and inputs; `9999px` full-pill radius on buttons and navigation pills | Soft ambient elevation (`rgba(0, 0, 0, 0.06) 0px 0px 18px 0px`) and smooth scroll-reveal transitions |

---

## 🧪 Core Laboratory Modules

1. **Home (`app.py`)**
   - SRM IST Division of Chemistry header, interactive module navigation, pre-loaded pharmaceutical benchmark gallery, descriptor summary table, and academic attribution cards.
2. **Molecule Viewer (`pages/1_Molecule_Viewer.py`)**
   - SMILES validation, canonical 2D skeletal depiction (`RDKit`), and interactive MMFF94 force-field minimised 3D conformer visualisation (`py3Dmol` / `stmol`) with **2D PNG** and **3D SDF** export.
3. **Property Calculator (`pages/2_Property_Calculator.py`)**
   - Computes Molecular Weight (MW), Crippen LogP, Topological Polar Surface Area (TPSA), H-Bond Donors (HBD), H-Bond Acceptors (HBA), rotatable bonds, ring count, and $F_{sp^3}$ alongside an interactive **Lipinski Rule-of-Five Bioavailability Radar** and stamped **PDF / CSV export**.
4. **Similarity Search (`pages/3_Similarity_Search.py`)**
   - Generates **Morgan (ECFP4)**, **MACCS Keys (166-bit)**, and **RDKit Topological** molecular fingerprints to rank reference compounds by **Tanimoto similarity** with horizontal bar charts and CSV export.
5. **Dataset Explorer (`pages/4_Dataset_Explorer.py`)**
   - Supports custom **CSV / SDF file uploads**, batch SMILES input, and the 24-compound pharmaceutical benchmark library with interactive MW/LogP sliders, Lipinski status filtering, and chemical space scatter/distribution charts.
6. **Theory, Assessment & Portfolio (`pages/5_About.py`)**
   - Concise revision notes on SMILES grammar, 3D force fields, Lipinski's Rule of Five, and Tanimoto similarity, paired with a **10-question self-assessment quiz** and a comprehensive **Session Lab Portfolio PDF generator**.

---

## 🎓 Academic Attribution

- **Course Coordinator & Faculty Mentor:**  
  **Dr. P Pachamuthu**  
  Professor, Division of Chemistry, School of Sciences  
  Faculty of Engineering and Technology, SRM Institute of Science and Technology, Tiruchirappalli  
  Email: `pachamuthup@srmist.edu.in`

- **Developed By (Student in SRMIST):**  
  **Varun K**  
  Registration No: `RA2611026050122` • Section: `AIML B`  
  B.Tech in Artificial Intelligence and Machine Learning, SRM Institute of Science and Technology

---

## 🚀 Local Setup & Deployment

### 1. Run Locally
```bash
git clone https://github.com/varunkolla31-create/cheminformatics-virtual-lab-apple.git
cd cheminformatics-virtual-lab-apple
pip install -r requirements.txt
streamlit run app.py
```

### 2. Deploy on Streamlit Community Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Select repository `varunkolla31-create/cheminformatics-virtual-lab-apple`, branch `main`, and main file path `app.py`.
4. Click **Deploy**.

---

*Created by Varun K | RA2611026050122 | Section: AIML B*
