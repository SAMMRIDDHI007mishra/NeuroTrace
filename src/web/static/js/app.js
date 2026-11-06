/**
 * NeuroTrace Frontend Interactive Logic
 * Handles asynchronous inference, preset auto-fill, dynamic XAI rendering & print reports.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------
    // Parkinson's Screening Logic
    // -------------------------------------------------------------
    const parkinsonsForm = document.getElementById("parkinsons-form");
    if (parkinsonsForm) {
        parkinsonsForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const submitBtn = document.getElementById("p-submit-btn");
            submitBtn.innerText = "Computing Neural Assessment...";
            submitBtn.disabled = true;

            const formData = new FormData(parkinsonsForm);
            const payload = {};
            formData.forEach((value, key) => {
                payload[key] = parseFloat(value);
            });

            try {
                const response = await fetch("/api/v1/predict/parkinsons", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                renderDiagnosisResult("p", data);
            } catch (err) {
                alert("Error during assessment: " + err);
            } finally {
                submitBtn.innerText = "Run Dysphonia Biomarker Assessment";
                submitBtn.disabled = false;
            }
        });
    }

    // -------------------------------------------------------------
    // Alzheimer's Screening Logic
    // -------------------------------------------------------------
    const alzheimersForm = document.getElementById("alzheimers-form");
    if (alzheimersForm) {
        alzheimersForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const submitBtn = document.getElementById("a-submit-btn");
            submitBtn.innerText = "Synthesizing Cognitive & MRI Profile...";
            submitBtn.disabled = true;

            const formData = new FormData(alzheimersForm);
            const payload = {};
            formData.forEach((value, key) => {
                payload[key] = parseFloat(value);
            });

            try {
                const response = await fetch("/api/v1/predict/alzheimers", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                renderDiagnosisResult("a", data);
            } catch (err) {
                alert("Error during assessment: " + err);
            } finally {
                submitBtn.innerText = "Run Cognitive & Volumetric Assessment";
                submitBtn.disabled = false;
            }
        });
    }

    // -------------------------------------------------------------
    // Batch CSV Upload
    // -------------------------------------------------------------
    const batchForm = document.getElementById("batch-form");
    if (batchForm) {
        batchForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const batchFile = document.getElementById("batch-file").files[0];
            const disease = document.getElementById("batch-disease").value;
            if (!batchFile) {
                alert("Please select a valid CSV file.");
                return;
            }

            const formData = new FormData();
            formData.append("file", batchFile);
            formData.append("disease", disease);

            const statusDiv = document.getElementById("batch-status");
            statusDiv.innerHTML = "<p>Processing cohort batch with ensembled neural classifiers...</p>";

            try {
                const resp = await fetch("/api/v1/batch_predict", {
                    method: "POST",
                    body: formData
                });
                const result = await resp.json();
                if (result.error) {
                    statusDiv.innerHTML = `<p style="color:#ef4444;">Error: ${result.error}</p>`;
                } else {
                    statusDiv.innerHTML = `
                        <div class="card" style="margin-top:16px;">
                            <h4 style="color:#10b981;">✓ Batch Screening Completed</h4>
                            <p><strong>Total Screened:</strong> ${result.total_records} | <strong>Positive Screenings:</strong> ${result.positive_count} | <strong>Cohort Avg Risk:</strong> ${(result.average_risk*100).toFixed(1)}%</p>
                            <p style="margin-top:10px; font-size:0.85rem; color:#9ca3af;">Annotated results exported to data/batch_results_${disease}.csv</p>
                        </div>
                    `;
                }
            } catch (err) {
                statusDiv.innerHTML = `<p style="color:#ef4444;">Batch execution failed: ${err}</p>`;
            }
        });
    }
});

// Render Dynamic Diagnostic Result
function renderDiagnosisResult(prefix, data) {
    const resultsContainer = document.getElementById(`${prefix}-results`);
    if (!resultsContainer) return;
    resultsContainer.style.display = "block";
    resultsContainer.scrollIntoView({ behavior: "smooth" });

    // Gauge
    const gauge = document.getElementById(`${prefix}-gauge`);
    const scoreText = document.getElementById(`${prefix}-risk-score`);
    const statusPill = document.getElementById(`${prefix}-status-pill`);
    const diagText = document.getElementById(`${prefix}-diag-text`);

    gauge.className = "risk-gauge " + data.risk_level.toLowerCase();
    scoreText.innerText = `${data.confidence_percentage}%`;
    diagText.innerText = data.diagnosis;

    if (data.risk_level === "High") {
        statusPill.className = "risk-status-pill pill-high";
        statusPill.innerText = "HIGH RISK DETECTED";
    } else if (data.risk_level === "Moderate") {
        statusPill.className = "risk-status-pill pill-moderate";
        statusPill.innerText = "MODERATE RISK";
    } else {
        statusPill.className = "risk-status-pill pill-low";
        statusPill.innerText = "NORMAL / LOW RISK";
    }

    // Explainable AI Table
    const tableBody = document.getElementById(`${prefix}-xai-tbody`);
    if (tableBody && data.top_contributing_biomarkers) {
        tableBody.innerHTML = "";
        data.top_contributing_biomarkers.forEach(b => {
            const isPos = b.impact_score > 0;
            const badgeClass = isPos ? "impact-badge-pos" : "impact-badge-neg";
            const badgeSign = isPos ? "+" : "";
            const row = document.createElement("tr");
            row.innerHTML = `
                <td><strong>${b.feature}</strong></td>
                <td>${b.patient_value}</td>
                <td>${b.population_mean}</td>
                <td class="${badgeClass}">${badgeSign}${(b.impact_score * 100).toFixed(1)}%</td>
                <td><span style="font-size:0.75rem; color:#9ca3af;">${b.category.replace('_', ' ').toUpperCase()}</span></td>
            `;
            tableBody.appendChild(row);
        });
    }
}

// Preset Loader
async function loadPreset(disease, presetKey) {
    try {
        const resp = await fetch("/api/v1/presets");
        const presets = await resp.json();
        const data = presets[disease][presetKey];
        if (!data) return;

        const features = data.features;
        for (const [key, value] of Object.entries(features)) {
            const input = document.getElementById(`${disease}-${key}`) || document.querySelector(`[name="${key}"]`);
            if (input) {
                input.value = value;
                input.classList.add("highlight");
                setTimeout(() => input.classList.remove("highlight"), 600);
            }
        }
    } catch (err) {
        console.error("Failed to load preset:", err);
    }
}
