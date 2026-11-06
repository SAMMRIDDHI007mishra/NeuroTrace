# NeuroTrace: Clinical & Neuropathological Foundations

## 1. Parkinson's Disease (PD) & Hypokinetic Dysarthria

### 1.1 Neuropathology
Parkinson's Disease is a progressive neurodegenerative disorder primarily characterized by the progressive degeneration of dopaminergic neurons within the **substantia nigra pars compacta (SNpc)**. The ensuing dopamine deficiency disrupts the striatal motor loop, precipitating cardinal motor manifestations: resting tremor, bradykinesia, muscular rigidity, and postural instability.

### 1.2 Vocal Biomarkers & Dysphonia
Over **89%** of patients with Parkinson's Disease experience speech and voice alterations (hypokinetic dysarthria). Crucially, acoustic changes frequently precede overt locomotor symptoms by up to 5 years:
- **Phonatory Tremor & Jitter**: Depleted dopaminergic control causes subclinical oscillations of the intrinsic laryngeal muscles (cricothyroid and thyroarytenoid), introducing involuntary frequency perturbations (micro-jitter).
- **Reduced Glottic Competence (Shimmer & HNR)**: Incomplete vocal fold closure leads to glottal air leakage, resulting in turbulent breathiness, reduced Harmonics-to-Noise Ratio (HNR), and elevated amplitude perturbation (shimmer).
- **Pitch Period Entropy (PPE)**: Non-linear complexity measurements isolate impaired neuromuscular tension regulation from voluntary vocal inflections.

---

## 2. Alzheimer's Disease (AD) & Cortical Atrophy

### 2.1 Neuropathology
Alzheimer's Disease is the most prevalent etiology of progressive dementia. Pathophysiologically, it is defined by:
1. **Extracellular Amyloid-Beta ($A\beta_{42}$) Senile Plaques**: Accumulation of insoluble peptide fragments causing synaptic disruption.
2. **Intracellular Hyperphosphorylated Tau Neurofibrillary Tangles (NFTs)**: Microtubule disassembly leading to neuronal apoptosis, beginning in the entorhinal cortex and hippocampus before disseminating throughout the cerebral cortex.

### 2.2 Neuroimaging & Cognitive Biomarkers
- **Normalized Whole Brain Volume (nWBV)**: Quantitative volumetric measurement reflecting progressive cerebral atrophy and ventricular dilation.
- **Estimated Total Intracranial Volume (eTIV)**: Serves as an individualized baseline covariate representing premorbid maximum brain size.
- **Mini-Mental State Examination (MMSE)**: 30-point standardized cognitive assessment evaluating orientation, registration, attention, calculation, recall, and language. Scores $\ge 24$ typically indicate normal cognition; scores $< 24$ indicate cognitive impairment.
- **Clinical Dementia Rating (CDR)**: Semistructured clinical interview assessing 6 cognitive domains (Memory, Orientation, Judgment, Community Affairs, Home & Hobbies, Personal Care).
  - `0.0`: Cognitively Normal
  - `0.5`: Very Mild Dementia / Mild Cognitive Impairment (MCI)
  - `1.0`: Mild Dementia
  - `2.0`: Moderate Dementia
  - `3.0`: Severe Dementia

---

## 3. Dataset Attribution & Ethical Acknowledgments

### 3.1 UCI Machine Learning Repository: Parkinson's Dataset
- **Primary Investigators**: Max A. Little, Patrick E. McSharry, Stephen J. Roberts, Declan A. E. Costello, Irene M. Moroz.
- **Citation**: Little MA, McSharry PE, Roberts SJ, Costello DA, Moroz IM. *Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection*. BioMedical Engineering OnLine 2007, 6:23.
- **Cohort**: 195 sustained vowel voice recordings from 31 subjects (23 diagnosed with Parkinson's Disease, 8 age-matched healthy controls).

### 3.2 Washington University OASIS Longitudinal MRI Study
- **Primary Investigators**: Daniel S. Marcus, Tracy H. Wang, Jamie Bandler, Denise Head, Randy L. Buckner.
- **Citation**: Marcus DS, Wang TH, Bandler J, Conturo TE, Nigro EC, Clovis C, Fox PT, Buckner RL. *Open Access Series of Imaging Studies (OASIS): Longitudinal MRI Data in Nondemented and Demented Older Adults*. Journal of Cognitive Neuroscience, 2010.
- **Cohort**: 373 MRI sessions across 150 participants aged 60 to 96, scanned at 1.5T with standardized MP-RAGE sequences.
