/**
 * XAURA Experiments Page — Client-side JavaScript
 *
 * Handles:
 *  - Fetching and rendering experiment runs
 *  - Search filtering by model name
 *  - Task-type dropdown filtering
 *  - Column sorting (model, task, date, duration)
 *  - Row selection with checkboxes
 *  - Side-by-side comparison modal
 *  - CSV log download
 *  - Individual run export / delete
 */

document.addEventListener("DOMContentLoaded", () => {
    // ── DOM references ──────────────────────────────────────────
    const tbody        = document.getElementById("experiments-tbody");
    const tableLoading = document.getElementById("table-loading");
    const tableEmpty   = document.getElementById("table-empty");
    const searchInput  = document.getElementById("search-input");
    const filterTask   = document.getElementById("filter-task");
    const compareBtn   = document.getElementById("compare-btn");
    const exportCsvBtn = document.getElementById("export-csv-btn");
    const selectAllCb  = document.getElementById("select-all");
    const compareModal = document.getElementById("compare-modal");
    const compareBody  = document.getElementById("compare-body");
    const modalClose   = document.getElementById("modal-close");

    // ── State ───────────────────────────────────────────────────
    let allRuns   = [];       // Full dataset from API
    let displayed = [];       // Currently displayed (after filter/sort)
    let selected  = new Set();
    let sortKey   = "created_at";
    let sortAsc   = false;    // newest first by default

    // ── Initial fetch ───────────────────────────────────────────
    fetchExperiments();

    // ── Fetch all experiments from API ──────────────────────────
    async function fetchExperiments() {
        tableLoading.style.display = "flex";
        tableEmpty.style.display   = "none";
        tbody.innerHTML            = "";

        try {
            const resp = await fetch("/api/experiments");
            if (!resp.ok) throw new Error("Failed to load experiments");
            const data = await resp.json();
            allRuns = data.runs || [];
        } catch (err) {
            allRuns = [];
            console.error("Fetch error:", err);
        }

        tableLoading.style.display = "none";
        applyFilters();
    }

    // ── Filter + Sort + Render ──────────────────────────────────
    function applyFilters() {
        const query    = (searchInput.value || "").toLowerCase().trim();
        const taskType = filterTask.value;

        // Filter
        displayed = allRuns.filter(run => {
            const nameMatch = !query || (run.model_name || "").toLowerCase().includes(query);
            const taskMatch = !taskType || run.task_type === taskType;
            return nameMatch && taskMatch;
        });

        // Sort
        displayed.sort((a, b) => {
            let va = a[sortKey] ?? "";
            let vb = b[sortKey] ?? "";

            // Numeric sort for duration
            if (sortKey === "duration_seconds") {
                va = parseFloat(va) || 0;
                vb = parseFloat(vb) || 0;
            }

            if (va < vb) return sortAsc ? -1 : 1;
            if (va > vb) return sortAsc ? 1 : -1;
            return 0;
        });

        renderTable();
    }

    // ── Render the table body ───────────────────────────────────
    function renderTable() {
        tbody.innerHTML = "";
        selected.clear();
        updateCompareBtn();

        if (displayed.length === 0) {
            tableEmpty.style.display = "flex";
            return;
        }
        tableEmpty.style.display = "none";

        displayed.forEach(run => {
            const id    = run.id || "";
            const model = run.model_name || "—";
            const task  = run.task_type || "—";
            const date  = formatDate(run.created_at);
            const dur   = run.duration_seconds
                ? `${parseFloat(run.duration_seconds).toFixed(2)}s`
                : "—";

            // Build a compact metric summary (top 2 metrics)
            const metricsHtml = formatMetrics(run.metrics);

            // Task badge colour
            const badgeClass = `task-badge-${task}`;

            const row = document.createElement("tr");
            row.dataset.id = id;
            row.innerHTML = `
                <td class="col-check">
                    <input type="checkbox" class="row-check" data-id="${id}">
                </td>
                <td class="col-model">
                    <span class="model-name-cell">${model}</span>
                </td>
                <td>
                    <span class="task-badge ${badgeClass}">${task}</span>
                </td>
                <td class="col-date">${date}</td>
                <td class="col-metrics">${metricsHtml}</td>
                <td class="col-duration mono">${dur}</td>
                <td class="col-actions">
                    <a href="/experiments/${id}/view" class="btn-icon" title="View Results">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                             stroke="currentColor" stroke-width="2">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                            <circle cx="12" cy="12" r="3"></circle>
                        </svg>
                    </a>
                    <button class="btn-icon" title="Export ZIP" onclick="downloadZip('${id}')">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                             stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="7 10 12 15 17 10"></polyline>
                            <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                    </button>
                    <button class="btn-icon btn-icon-danger" title="Delete" onclick="deleteRun('${id}')">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                             stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path>
                            <line x1="10" y1="11" x2="10" y2="17"></line>
                            <line x1="14" y1="11" x2="14" y2="17"></line>
                        </svg>
                    </button>
                </td>
            `;
            tbody.appendChild(row);
        });

        // Attach checkbox listeners
        tbody.querySelectorAll(".row-check").forEach(cb => {
            cb.addEventListener("change", () => {
                if (cb.checked) selected.add(cb.dataset.id);
                else selected.delete(cb.dataset.id);
                updateCompareBtn();
            });
        });
    }

    // ── Helpers ──────────────────────────────────────────────────

    function formatDate(isoStr) {
        if (!isoStr) return "—";
        const d = new Date(isoStr);
        if (isNaN(d.getTime())) return isoStr.slice(0, 16);
        return d.toLocaleDateString("en-GB", {
            day: "2-digit", month: "short", year: "numeric",
        }) + " " + d.toLocaleTimeString("en-GB", {
            hour: "2-digit", minute: "2-digit",
        });
    }

    function formatMetrics(metrics) {
        if (!metrics || typeof metrics !== "object") return "—";
        const entries = Object.entries(metrics);
        if (entries.length === 0) return "—";

        return entries.slice(0, 3).map(([k, v]) => {
            const val = typeof v === "number" ? v.toFixed(4) : v;
            return `<span class="metric-pill"><span class="metric-pill-name">${k}</span> ${val}</span>`;
        }).join(" ");
    }

    function updateCompareBtn() {
        compareBtn.disabled = selected.size < 2;
    }

    // ── Search & Filter listeners ───────────────────────────────
    searchInput.addEventListener("input", applyFilters);
    filterTask.addEventListener("change", applyFilters);

    // ── Column sorting ──────────────────────────────────────────
    document.querySelectorAll(".col-sortable").forEach(th => {
        th.style.cursor = "pointer";
        th.addEventListener("click", () => {
            const key = th.dataset.sort;
            if (sortKey === key) {
                sortAsc = !sortAsc;
            } else {
                sortKey = key;
                sortAsc = true;
            }

            // Visual indicator
            document.querySelectorAll(".col-sortable").forEach(h =>
                h.classList.remove("sort-asc", "sort-desc")
            );
            th.classList.add(sortAsc ? "sort-asc" : "sort-desc");

            applyFilters();
        });
    });

    // ── Select all checkbox ─────────────────────────────────────
    selectAllCb.addEventListener("change", () => {
        const checked = selectAllCb.checked;
        tbody.querySelectorAll(".row-check").forEach(cb => {
            cb.checked = checked;
            if (checked) selected.add(cb.dataset.id);
            else selected.delete(cb.dataset.id);
        });
        updateCompareBtn();
    });

    // ── Compare selected runs ───────────────────────────────────
    compareBtn.addEventListener("click", async () => {
        if (selected.size < 2) return;
        const ids = [...selected].join(",");

        try {
            const resp = await fetch(`/api/experiments/compare?ids=${ids}`);
            if (!resp.ok) throw new Error("Comparison failed");
            const data = await resp.json();
            showCompareModal(data.runs);
        } catch (err) {
            alert("Comparison failed: " + err.message);
        }
    });

    // ── Build comparison modal ──────────────────────────────────
    function showCompareModal(runs) {
        if (!runs || runs.length === 0) return;

        // Collect all unique metric keys
        const metricKeys = new Set();
        runs.forEach(r => {
            if (r.metrics) Object.keys(r.metrics).forEach(k => metricKeys.add(k));
        });

        let html = `<table class="data-table compare-table">`;

        // Header row: run labels
        html += `<thead><tr><th>Metric</th>`;
        runs.forEach(r => {
            html += `<th>
                <span class="model-name-cell">${r.model_name || "—"}</span>
                <br><span class="compare-date">${formatDate(r.created_at)}</span>
            </th>`;
        });
        html += `</tr></thead><tbody>`;

        // Meta rows
        html += makeCompareRow("Task", runs.map(r => r.task_type || "—"));
        html += makeCompareRow("Duration", runs.map(r =>
            r.duration_seconds ? `${parseFloat(r.duration_seconds).toFixed(2)}s` : "—"
        ));

        // Metric rows — highlight the best value
        metricKeys.forEach(key => {
            const values = runs.map(r => r.metrics ? r.metrics[key] : null);
            const nums   = values.filter(v => typeof v === "number");
            const best   = nums.length > 0 ? Math.max(...nums) : null;

            const cells = values.map(v => {
                if (v === null || v === undefined) return "—";
                const val = typeof v === "number" ? v.toFixed(4) : v;
                const isBest = v === best && nums.length > 1;
                return isBest ? `<span class="compare-best">${val}</span>` : val;
            });
            html += makeCompareRow(key, cells);
        });

        html += `</tbody></table>`;
        compareBody.innerHTML = html;
        compareModal.style.display = "flex";
    }

    function makeCompareRow(label, cells) {
        let row = `<tr><td class="compare-label">${label}</td>`;
        cells.forEach(c => { row += `<td>${c}</td>`; });
        return row + `</tr>`;
    }

    // ── Close modal ─────────────────────────────────────────────
    modalClose.addEventListener("click", () => {
        compareModal.style.display = "none";
    });
    compareModal.addEventListener("click", (e) => {
        if (e.target === compareModal) compareModal.style.display = "none";
    });

    // ── CSV export ──────────────────────────────────────────────
    exportCsvBtn.addEventListener("click", async () => {
        try {
            const resp = await fetch("/api/export/log/csv");
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "CSV export failed");
            }
            const blob = await resp.blob();
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement("a");
            a.href     = url;
            a.download = "xaura_experiment_log.csv";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            alert("Export failed: " + err.message);
        }
    });
});


// ── Global action handlers (called from inline onclick) ─────────

async function downloadZip(runId) {
    try {
        const resp = await fetch(`/api/export/${runId}/zip`);
        if (!resp.ok) throw new Error("Export failed");
        const blob = await resp.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement("a");
        a.href     = url;
        a.download = `xaura_run_${runId.slice(0, 8)}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (err) {
        alert("Export failed: " + err.message);
    }
}

async function deleteRun(runId) {
    if (!confirm("Delete this experiment run? This cannot be undone.")) return;

    try {
        const resp = await fetch(`/api/experiments/${runId}`, { method: "DELETE" });
        if (!resp.ok) throw new Error("Delete failed");

        // Remove the row from the DOM with a fade-out animation
        const row = document.querySelector(`tr[data-id="${runId}"]`);
        if (row) {
            row.style.transition = "opacity 300ms ease";
            row.style.opacity = "0";
            setTimeout(() => row.remove(), 300);
        }
    } catch (err) {
        alert("Delete failed: " + err.message);
    }
}
