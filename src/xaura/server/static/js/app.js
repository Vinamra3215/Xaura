/**
 * XAURA Dashboard — Client-side JavaScript
 *
 * Handles:
 *  - File upload (drag & drop + click)
 *  - Profile page redirect
 *  - Plotly chart rendering
 *  - Model run form submission
 *  - Export download
 */

// ── File Upload ─────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const progress = document.getElementById("upload-progress");
    const progressFill = document.getElementById("progress-fill");
    const progressText = document.getElementById("progress-text");
    const errorDiv = document.getElementById("upload-error");
    const errorText = document.getElementById("error-text");

    if (!dropzone) return; // Not on the landing page

    // Click to browse
    dropzone.addEventListener("click", () => fileInput.click());

    // File selected
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });

    // Drag & drop
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-over");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("drag-over");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");

        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith(".csv")) {
            uploadFile(file);
        } else {
            showError("Please upload a CSV file.");
        }
    });

    async function uploadFile(file) {
        // Show progress, hide error
        progress.style.display = "block";
        errorDiv.style.display = "none";
        progressFill.style.width = "10%";
        progressText.textContent = `Uploading ${file.name}...`;

        const formData = new FormData();
        formData.append("file", file);

        try {
            progressFill.style.width = "40%";
            progressText.textContent = "Profiling dataset...";

            const response = await fetch("/api/profile", {
                method: "POST",
                body: formData,
            });

            progressFill.style.width = "80%";

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || "Upload failed");
            }

            const data = await response.json();
            progressFill.style.width = "100%";
            progressText.textContent = "Done! Redirecting...";

            // Redirect to profile page
            setTimeout(() => {
                window.location.href = `/profile/${data.session_id}`;
            }, 500);
        } catch (err) {
            progress.style.display = "none";
            showError(err.message);
        }
    }

    function showError(message) {
        errorDiv.style.display = "block";
        errorText.textContent = message;
    }
});


// ── Plotly Chart Rendering ──────────────────────────────────────────

/**
 * Render a Plotly chart from JSON data into a container element.
 *
 * @param {string} containerId - The DOM element ID.
 * @param {string} chartJson - JSON string from fig.to_json().
 */
function renderChart(containerId, chartJson) {
    const container = document.getElementById(containerId);
    if (!container || !chartJson) return;

    try {
        const figure = JSON.parse(chartJson);
        // Override layout to fit container
        figure.layout = figure.layout || {};
        figure.layout.autosize = true;
        figure.layout.margin = figure.layout.margin || {};

        Plotly.newPlot(container, figure.data, figure.layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ["lasso2d", "select2d"],
            displaylogo: false,
        });
    } catch (err) {
        container.innerHTML = `<p style="color: var(--text-muted); text-align: center; padding: 40px;">Chart could not be loaded.</p>`;
    }
}


// ── Model Run ───────────────────────────────────────────────────────

/**
 * Submit a model run request and redirect to results page.
 *
 * @param {string} sessionId - The session ID from profile upload.
 * @param {string} modelName - The model to run.
 * @param {string} targetCol - The target column name.
 */
async function runModel(sessionId, modelName, targetCol) {
    const loadingOverlay = document.getElementById("loading-overlay");
    if (loadingOverlay) loadingOverlay.style.display = "flex";

    // Collect selected feature columns
    const checkboxes = document.querySelectorAll('input[name="feature_col"]:checked');
    const selectedColumns = Array.from(checkboxes).map(cb => cb.value);

    // Filter out the target column from features
    const featureCols = selectedColumns.filter(col => col !== targetCol);

    if (featureCols.length === 0) {
        if (loadingOverlay) loadingOverlay.style.display = "none";
        alert("Please select at least one feature column.");
        return;
    }

    try {
        const response = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: sessionId,
                model_name: modelName,
                target_col: targetCol,
                selected_columns: featureCols,
            }),
        });

        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Model run failed");
        }

        const data = await response.json();
        window.location.href = `/results/${sessionId}`;
    } catch (err) {
        if (loadingOverlay) loadingOverlay.style.display = "none";
        alert("Error: " + err.message);
    }
}


// ── Feature Column Helpers ──────────────────────────────────────────

/**
 * Toggle all feature checkboxes on or off.
 */
function toggleAllFeatures(checked) {
    const checkboxes = document.querySelectorAll('input[name="feature_col"]');
    checkboxes.forEach(cb => { cb.checked = checked; });
    updateFeatureCount();
}

/**
 * Update the feature count display and select-all checkbox state.
 */
function updateFeatureCount() {
    const total = document.querySelectorAll('input[name="feature_col"]').length;
    const checked = document.querySelectorAll('input[name="feature_col"]:checked').length;

    // Update count display
    const countEl = document.getElementById("feature-count");
    if (countEl) {
        countEl.textContent = `${checked}/${total} selected`;
    }

    // Update select-all checkbox state
    const selectAllCb = document.getElementById("select-all-cb");
    if (selectAllCb) {
        selectAllCb.checked = (checked === total && total > 0);
        selectAllCb.indeterminate = (checked > 0 && checked < total);
    }
}

// Initialize count on page load
document.addEventListener("DOMContentLoaded", function() {
    if (document.getElementById("feature-count")) {
        updateFeatureCount();
    }
});


// ── Target Column Update ────────────────────────────────────────────

/**
 * Dynamically update the target column and task type on the server,
 * then update the UI task type displays.
 */
async function updateTargetCol(sessionId, targetCol) {
    try {
        const response = await fetch("/api/profile/update_target", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: sessionId,
                target_col: targetCol,
            }),
        });

        if (!response.ok) {
            console.error("Failed to update target column");
            return;
        }

        const data = await response.json();
        const newTaskType = data.task_type || "unknown";

        // Update the UI
        const typeDisplay = document.getElementById("task-type-display");
        const typeDetailDisplay = document.getElementById("task-type-detail-display");
        const targetColDisplay = document.getElementById("target-col-display");
        const targetInfoCard = document.getElementById("target-info-card");

        if (typeDisplay) typeDisplay.textContent = newTaskType;
        if (typeDetailDisplay) typeDetailDisplay.textContent = newTaskType;
        if (targetColDisplay) targetColDisplay.textContent = targetCol || "None";

        if (targetInfoCard) {
            targetInfoCard.style.display = targetCol ? "block" : "none";
        }

    } catch (err) {
        console.error("Error updating target column:", err);
    }
}


// ── Export Download ─────────────────────────────────────────────────

/**
 * Trigger a ZIP export download.
 *
 * @param {string} sessionId - The session ID.
 */
async function exportResults(sessionId) {
    try {
        const response = await fetch(`/api/export/${sessionId}`);
        if (!response.ok) throw new Error("Export failed");

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `xaura_export_${sessionId}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (err) {
        alert("Export failed: " + err.message);
    }
}
