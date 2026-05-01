# BiasBYE Scanner API Documentation

Welcome to the **BiasBYE Scanner API** documentation. This API provides an intersectional subgroup discovery engine designed to detect biases and disparities in binary outcome datasets (e.g., loan approvals, hiring, recidivism).

The engine uses a combinatorial beam search to identify groups defined by one or more protected attributes that experience significant statistical disparities compared to the population baseline.

---

## 🚀 Getting Started

### API Base URL
- **Local Development**: `http://localhost:8000`
- **Production**: `https://biasbye-scanner.onrender.com`

### Base Models

#### `ScanRequest`
| Field | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `dataset_path` | `string` | Absolute path to the CSV file (local or cloud). | Required |
| `protected_attributes` | `list[string]` | List of columns to check for bias (e.g., `["gender", "race"]`). | `None` (Auto-detect) |
| `outcome_column` | `string` | The target binary variable (e.g., `approved`). | `None` (Auto-detect) |
| `positive_value` | `integer` | The value representing a "favorable" outcome. | `1` |
| `min_subgroup_size` | `integer` | Minimum number of individuals required in a subgroup. | `30` |
| `max_intersectionality` | `integer` | Max number of combined attributes (e.g., 3 = Race + Gender + Age). | `3` |
| `beam_width` | `integer` | Number of top candidates to expand at each level. | `20` |

---

## 🛠 Endpoints

### 1. Health Check
`GET /health`

Returns the current status of the API.

**Response**
```json
{
  "status": "healthy",
  "timestamp": "2024-05-02T10:00:00.000Z"
}
```

---

### 2. Start Asynchronous Scan
`POST /scan`

Initiates a background job to scan a dataset. Recommended for large datasets.

**Request Body**
```json
{
  "dataset_path": "/data/datasets/recidivism.csv",
  "protected_attributes": ["race", "sex", "age_cat"],
  "outcome_column": "two_year_recid",
  "positive_value": 0,
  "min_subgroup_size": 50
}
```

**Response**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

---

### 3. Get Scan Status & Results
`GET /scan/{job_id}`

Retrieves the current status, progress, and results (if complete) for a background job.

**Response (Success)**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "complete",
  "progress": 100,
  "created_at": "2024-05-02T10:05:00.000Z",
  "results": { ... },
  "error": null
}
```

---

### 4. Direct CSV Upload (Synchronous)
`POST /scan/upload`

Uploads a CSV file and runs the scan immediately. Returns results in the response. Use for files < 10MB.

**Request Form Data**
- `file`: (Binary CSV file)
- `protected_attributes`: (Optional, comma-separated string) e.g., `"race,gender"`
- `outcome_column`: (Optional string)
- `positive_value`: (Optional integer, default: 1)
- `min_subgroup_size`: (Optional integer, default: 30)

**Response**
Returns the same [Results Object](#results-object) as the async endpoint.

---

## 📊 Results Object

The results are returned in a structured format containing metadata, a global score, and specific disparity findings.

### `scan_metadata`
Summary of the scan parameters and high-level stats.
- `total_rows`: Number of samples analyzed.
- `baseline_rate`: The population-wide favorable outcome rate.
- `significant_disparities`: Count of groups passing the FDR-corrected significance threshold.
- `global_fairness_score`: A normalized 0-100 score (higher is better).

### `disparities` (Array)
A list of discovered subgroups ordered by significance and severity.

| Field | Type | Description |
| :--- | :--- | :--- |
| `subgroup_name` | `string` | Human-readable definition (e.g., `race=African-American + sex=Female`). |
| `attributes` | `dict` | Key-value pairs of the subgroup's defining attributes. |
| `population_size`| `int` | Number of people in this subgroup. |
| `disparity_pct` | `float` | Difference in favorable rate compared to baseline (percentage points). |
| `severity` | `string` | `low`, `medium`, `high`, or `critical`. |
| `p_value` | `float` | Raw p-value from two-proportion z-test. |
| `is_significant`| `bool` | True if the finding survives Benjamini-Hochberg FDR correction (α=0.05). |

---

## 🧠 Core Concepts

### Intersectional Discovery
Unlike standard tools that only look at one attribute at a time (e.g., "Gender"), BiasBYE discovers combined groups (e.g., "Older Hispanic Females") that may be overlooked by marginal analysis.

### Severity Logic
The API classifies findings based on:
1. **Statistical Significance**: Must pass FDR correction.
2. **Effect Size (Cohen's h)**: Magnitude of the disparity.
3. **Disparity Magnitude**: Percentage point gap from baseline.

### Auto-Detection
If `protected_attributes` or `outcome_column` are omitted:
- **Protected Attributes**: Looks for keywords like `race`, `gender`, `age`, `ethnicity`, etc.
- **Outcome**: Heuristically finds the binary numeric column with the most coverage.

---

## 🧪 Example API Response (Truncated)

```json
{
  "global_fairness_score": 72,
  "scan_metadata": {
    "significant_disparities": 14,
    "baseline_rate": 0.45
  },
  "disparities": [
    {
      "subgroup_name": "race=African-American + age_cat=25-45",
      "disparity_pct": -18.5,
      "severity": "critical",
      "is_significant": true,
      "p_value": 0.000042
    }
  ]
}
```
