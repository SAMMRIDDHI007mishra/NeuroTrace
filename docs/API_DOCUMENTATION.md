# NeuroTrace REST API Documentation

The NeuroTrace platform exposes a production-ready RESTful inference API for clinical integration, EHR systems, and automated pipeline execution.

**Base URL**: `http://localhost:5000` (or your deployed host)  
**Content-Type**: `application/json`

---

## 1. System Health Check

### `GET /api/v1/health`
Verifies server status and confirms active in-memory neural ensemble models.

#### Example Request
```bash
curl -X GET http://localhost:5000/api/v1/health
```

#### Example Response (200 OK)
```json
{
  "status": "online",
  "platform": "NeuroTrace",
  "version": "1.0.0",
  "models": {
    "parkinsons": true,
    "alzheimers": true
  }
}
```

---

## 2. Clinical Case Presets

### `GET /api/v1/presets`
Retrieves standardized clinical test cases for instant demonstrations, unit validation, and integration tests.

#### Example Request
```bash
curl -X GET http://localhost:5000/api/v1/presets
```

---

## 3. Parkinson's Voice Acoustic Inference

### `POST /api/v1/predict/parkinsons`
Evaluates sustained vowel acoustic biomarkers and calculates probability of Parkinson's Disease.

#### Headers
`Content-Type: application/json`

#### Example Request Body
```json
{
  "MDVP:Fo(Hz)": 119.992,
  "MDVP:Fhi(Hz)": 157.302,
  "MDVP:Flo(Hz)": 74.997,
  "MDVP:Jitter(%)": 0.00784,
  "MDVP:Jitter(Abs)": 0.00007,
  "MDVP:RAP": 0.00370,
  "MDVP:PPQ": 0.00554,
  "Jitter:DDP": 0.01109,
  "MDVP:Shimmer": 0.04374,
  "MDVP:Shimmer(dB)": 0.426,
  "Shimmer:APQ3": 0.02182,
  "Shimmer:APQ5": 0.03130,
  "MDVP:APQ": 0.02971,
  "Shimmer:DDA": 0.06545,
  "NHR": 0.02211,
  "HNR": 21.033,
  "RPDE": 0.414783,
  "DFA": 0.815285,
  "spread1": -4.813031,
  "spread2": 0.266482,
  "D2": 2.301442,
  "PPE": 0.284654
}
```

#### Example Response (200 OK)
```json
{
  "disease": "Parkinson's Disease",
  "diagnosis": "Parkinson's Detected (Positive)",
  "probability": 0.9421,
  "confidence_percentage": 94.2,
  "risk_level": "High",
  "top_contributing_biomarkers": [
    {
      "feature": "PPE",
      "patient_value": 0.2847,
      "population_mean": 0.2066,
      "impact_score": 0.1250,
      "category": "risk_escalator",
      "description": "PPE: patient has 0.285 vs normal mean 0.207"
    },
    {
      "feature": "spread1",
      "patient_value": -4.8130,
      "population_mean": -5.6844,
      "impact_score": 0.0910,
      "category": "risk_escalator",
      "description": "spread1: patient has -4.813 vs normal mean -5.684"
    }
  ],
  "warnings": [],
  "disclaimer": "Experimental AI screener. Not a standalone diagnostic substitute for neurological clinical evaluation."
}
```

---

## 4. Alzheimer's Cognitive & Volumetric MRI Inference

### `POST /api/v1/predict/alzheimers`
Evaluates demographic, psychometric (MMSE, CDR), and structural MRI volumetric features for Dementia/Alzheimer's risk.

#### Example Request Body
```json
{
  "Gender_Encoded": 1,
  "Age": 75,
  "EDUC": 12,
  "SES": 2.0,
  "MMSE": 22.0,
  "CDR": 1.0,
  "eTIV": 1550,
  "nWBV": 0.702,
  "ASF": 1.13
}
```

#### Example Response (200 OK)
```json
{
  "disease": "Alzheimer's Disease / Dementia",
  "diagnosis": "Dementia Detected (Positive)",
  "probability": 0.9985,
  "confidence_percentage": 99.9,
  "risk_level": "High",
  "top_contributing_biomarkers": [
    {
      "feature": "CDR",
      "patient_value": 1.0,
      "population_mean": 0.29,
      "impact_score": 0.4200,
      "category": "risk_escalator"
    },
    {
      "feature": "MMSE",
      "patient_value": 22.0,
      "population_mean": 27.34,
      "impact_score": 0.2150,
      "category": "risk_escalator"
    }
  ],
  "warnings": [],
  "disclaimer": "Experimental AI screener. Not a standalone diagnostic substitute for clinical neurological and PET/MRI evaluation."
}
```

---

## 5. Batch Cohort Processing

### `POST /api/v1/batch_predict`
Multipart form upload of CSV files for high-throughput screening.

#### Form Parameters
- `file`: CSV file containing expected feature columns
- `disease`: `"parkinsons"` or `"alzheimers"`

#### Example cURL
```bash
curl -X POST http://localhost:5000/api/v1/batch_predict \
  -F "file=@patients_cohort.csv" \
  -F "disease=parkinsons"
```
