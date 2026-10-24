/**
 * VoltPulse AI - Frontend Application Controller
 * Handles real-time API communication, interactive Plotly & Chart.js rendering,
 * live ML inference simulations, drag-and-drop CSV uploads, and scenario calculations.
 */

// Global App State
const state = {
  activeTab: 'patterns',
  seasons: ['Winter', 'Spring', 'Summer', 'Autumn'],
  charts: {},
  rulesData: [],
  selectedFile: null
};

// =====================================================================
// DOM Elements & Event Listeners
// =====================================================================
document.addEventListener('DOMContentLoaded', () => {
  initIcons();
  setupNavigation();
  setupFilters();
  setupUploadModal();
  setupLivePredictor();
  setupSimulator();
  setupActionButtons();

  // Initial Data Load
  loadAllData();
});

function initIcons() {
  if (window.lucide) {
    lucide.createIcons();
  }
}

// Tab Navigation
function setupNavigation() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      switchTab(target);
    });
  });
}

function switchTab(tabId) {
  state.activeTab = tabId;
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === tabId);
  });
  document.querySelectorAll('.tab-pane').forEach(p => {
    p.classList.toggle('active', p.id === `pane-${tabId}`);
  });

  // Re-trigger layout for Plotly charts in tab if needed
  window.dispatchEvent(new Event('resize'));
}

// Season Toggles
function setupFilters() {
  const seasonPills = document.querySelectorAll('.season-pill');
  seasonPills.forEach(pill => {
    pill.addEventListener('click', () => {
      const season = pill.dataset.season;
      if (pill.classList.contains('active')) {
        if (state.seasons.length > 1) {
          pill.classList.remove('active');
          state.seasons = state.seasons.filter(s => s !== season);
        } else {
          showToast('At least one season must be selected', 'warning');
          return;
        }
      } else {
        pill.classList.add('active');
        state.seasons.push(season);
      }
      loadPatterns();
    });
  });
}

// Action Buttons
function setupActionButtons() {
  document.getElementById('loadSampleBtn').addEventListener('click', async () => {
    showLoader('Loading Bundled Sample...', 'Parsing sample_upload.csv and executing machine learning models...');
    try {
      const res = await fetch('/api/load-sample', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        showToast('Sample 60-day dataset loaded successfully!', 'success');
        await loadAllData();
      } else {
        showToast(data.detail || 'Failed to load sample dataset', 'error');
      }
    } catch (err) {
      showToast('Network error while loading sample', 'error');
    } finally {
      hideLoader();
    }
  });

  document.getElementById('resetDataBtn').addEventListener('click', async () => {
    showLoader('Resetting Benchmark...', 'Restoring standard synthetic dataset...');
    try {
      const res = await fetch('/api/reset', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        showToast('Reset to standard benchmark dataset', 'info');
        await loadAllData();
      }
    } catch (err) {
      showToast('Network error on reset', 'error');
    } finally {
      hideLoader();
    }
  });
}

// =====================================================================
// Pipeline Orchestration
// =====================================================================
async function loadAllData() {
  try {
    await Promise.all([
      loadOverview(),
      loadPatterns(),
      loadClusters(),
      loadClassification(),
      loadAssociationRules(),
      loadAnomalies(),
      loadInsights()
    ]);
    triggerSimulation();
  } catch (e) {
    console.error('Error loading analytics:', e);
  }
}

// =====================================================================
// 1. Overview & High-Impact KPI Cards
// =====================================================================
async function loadOverview() {
  try {
    const res = await fetch('/api/overview');
    if (!res.ok) return;
    const data = await res.json();

    // Badges & Source
    document.getElementById('engineStatusText').textContent = 'AI Engine Online (Scikit-Learn)';
    document.getElementById('datasetSourceText').textContent = data.source || 'Standard Dataset';
    document.getElementById('dateRangeText').textContent = `${data.date_range.min} to ${data.date_range.max}`;
    document.getElementById('recordCountText').textContent = `${data.records.toLocaleString()} Hours (${data.days} Days)`;

    // KPIs
    const kpi = data.kpis;
    document.getElementById('valAvgLoad').textContent = kpi.avg_load_kw.toFixed(2);
    const standbyPct = ((kpi.standby_baseline_kw / (kpi.avg_load_kw || 1)) * 100).toFixed(0);
    document.getElementById('valStandbyRatio').textContent = `${standbyPct}% of mean`;

    document.getElementById('valPeakLoad').textContent = kpi.peak_load_kw.toFixed(2);
    document.getElementById('valHighSeason').textContent = `${kpi.high_season} (${kpi.high_season_kw.toFixed(2)} kW)`;

    document.getElementById('valTotalEnergy').textContent = Math.round(kpi.total_energy_kwh).toLocaleString();
    document.getElementById('valEstCost').textContent = `$ ${Math.round(kpi.est_cost_usd).toLocaleString()}`;

    document.getElementById('valPeakShare').textContent = kpi.peak_window_share_pct.toFixed(1);

    document.getElementById('valStandby').textContent = kpi.standby_baseline_kw.toFixed(2);
    document.getElementById('valStandbyDaily').textContent = `${kpi.standby_daily_kwh.toFixed(1)} kWh/day`;

    document.getElementById('valAnomalies').textContent = kpi.anomaly_count;
    document.getElementById('valAnomalySplit').textContent = `${kpi.anomaly_spikes} Spikes / ${kpi.anomaly_drops} Drops`;

  } catch (err) {
    console.error('Overview error:', err);
  }
}

// =====================================================================
// 2. Temporal Dynamics & Heatmap
// =====================================================================
async function loadPatterns() {
  try {
    const seasonParams = state.seasons.map(s => `seasons=${encodeURIComponent(s)}`).join('&');
    const url = `/api/patterns${seasonParams ? '?' + seasonParams : ''}`;
    const res = await fetch(url);
    if (!res.ok) return;
    const data = await res.json();

    renderHeatmap(data.heatmap);
    renderWeekdayWeekendChart(data.weekday_vs_weekend);
    renderMonthlyChart(data.monthly);
    renderTempScatter(data.temp_scatter);

  } catch (err) {
    console.error('Patterns error:', err);
  }
}

function renderHeatmap(hm) {
  const zData = hm.rows.map(r => r.values);
  const yData = hm.rows.map(r => r.day);
  const xData = hm.hours.map(h => `${h}:00`);

  const plotData = [{
    z: zData,
    x: xData,
    y: yData,
    type: 'heatmap',
    colorscale: [
      [0, '#0D1527'],
      [0.2, '#06B6D4'],
      [0.45, '#10B981'],
      [0.7, '#F59E0B'],
      [1.0, '#F43F5E']
    ],
    hoverongaps: false,
    hovertemplate: '<b>%{y}</b> at %{x}<br>Average Demand: <b>%{z:.2f} kW</b><extra></extra>',
    colorbar: {
      title: { text: 'kW', font: { color: '#94A3B8', size: 12 } },
      tickfont: { color: '#94A3B8' },
      outlinecolor: 'transparent'
    }
  }];

  const layout = {
    margin: { t: 20, r: 40, b: 45, l: 90 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: {
      tickfont: { color: '#94A3B8', size: 11 },
      gridcolor: 'rgba(255,255,255,0.05)',
      tickangle: -45
    },
    yaxis: {
      tickfont: { color: '#CBD5E1', size: 12 },
      gridcolor: 'rgba(255,255,255,0.05)',
      autorange: 'reversed'
    }
  };

  Plotly.newPlot('heatmapPlot', plotData, layout, { responsive: true, displayModeBar: false });
}

function renderWeekdayWeekendChart(data) {
  const ctx = document.getElementById('chartWeekdayWeekend').getContext('2d');
  if (state.charts.weekdayWeekend) state.charts.weekdayWeekend.destroy();

  state.charts.weekdayWeekend = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.hours.map(h => `${h}:00`),
      datasets: [
        {
          label: 'Weekday',
          data: data.weekday,
          borderColor: '#00E5FF',
          backgroundColor: 'rgba(0, 229, 255, 0.08)',
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 5
        },
        {
          label: 'Weekend',
          data: data.weekend,
          borderColor: '#F59E0B',
          backgroundColor: 'rgba(245, 158, 11, 0.05)',
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 5
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#CBD5E1', boxWidth: 14, font: { family: 'Plus Jakarta Sans', size: 12 } }
        },
        tooltip: {
          backgroundColor: '#0D1527',
          borderColor: 'rgba(255,255,255,0.15)',
          borderWidth: 1,
          titleColor: '#00E5FF',
          bodyColor: '#F8FAFC',
          padding: 10,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} kW`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#94A3B8', maxTicksLimit: 12 }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#94A3B8' },
          title: { display: true, text: 'Average kW', color: '#64748B' }
        }
      }
    }
  });
}

function renderMonthlyChart(data) {
  const ctx = document.getElementById('chartMonthly').getContext('2d');
  if (state.charts.monthly) state.charts.monthly.destroy();

  const labels = data.map(d => d.month);
  const values = data.map(d => d.kw);

  state.charts.monthly = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Mean Load (kW)',
        data: values,
        backgroundColor: values.map(v => v > 1.2 ? 'rgba(244, 63, 94, 0.75)' : 'rgba(16, 185, 129, 0.75)'),
        borderColor: values.map(v => v > 1.2 ? '#F43F5E' : '#10B981'),
        borderWidth: 1,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0D1527',
          borderColor: 'rgba(255,255,255,0.15)',
          borderWidth: 1,
          padding: 10,
          callbacks: { label: (c) => `Avg Load: ${c.parsed.y.toFixed(2)} kW` }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#94A3B8' }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#94A3B8' },
          title: { display: true, text: 'kW', color: '#64748B' }
        }
      }
    }
  });
}

function renderTempScatter(points) {
  const colors = {
    'Night': '#8B5CF6',
    'Morning': '#00E5FF',
    'Afternoon': '#F59E0B',
    'Evening': '#F43F5E'
  };

  const dayparts = ['Night', 'Morning', 'Afternoon', 'Evening'];
  const traces = dayparts.map(dp => {
    const subset = points.filter(p => p.daypart === dp);
    return {
      x: subset.map(p => p.temp_c),
      y: subset.map(p => p.kw),
      mode: 'markers',
      type: 'scatter',
      name: dp,
      marker: {
        color: colors[dp] || '#00E5FF',
        size: 5,
        opacity: 0.65
      },
      hovertemplate: `<b>${dp}</b><br>Temp: %{x:.1f}°C<br>Load: %{y:.2f} kW<extra></extra>`
    };
  });

  const layout = {
    margin: { t: 20, r: 20, b: 45, l: 50 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: {
      title: { text: 'Ambient Temperature (°C)', font: { color: '#94A3B8', size: 12 } },
      tickfont: { color: '#94A3B8' },
      gridcolor: 'rgba(255,255,255,0.05)'
    },
    yaxis: {
      title: { text: 'Active Power (kW)', font: { color: '#94A3B8', size: 12 } },
      tickfont: { color: '#94A3B8' },
      gridcolor: 'rgba(255,255,255,0.05)'
    },
    legend: {
      font: { color: '#CBD5E1' },
      orientation: 'h',
      x: 0,
      y: 1.15
    }
  };

  Plotly.newPlot('tempScatterPlot', traces, layout, { responsive: true, displayModeBar: false });
}

// =====================================================================
// 3. Unsupervised Clustering & Archetypes
// =====================================================================
async function loadClusters() {
  try {
    const res = await fetch('/api/clusters');
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById('clusterKBadge').querySelector('span').textContent = `Optimal k = ${data.chosen_k} Clusters`;

    renderCentroidsChart(data.centroids);
    renderPcaPlot(data.pca_points);
    renderElbowSilhouetteCharts(data.evaluation);
    renderClusterDistTable(data.distribution);

  } catch (err) {
    console.error('Clusters error:', err);
  }
}

function renderCentroidsChart(centroids) {
  const ctx = document.getElementById('chartCentroids').getContext('2d');
  if (state.charts.centroids) state.charts.centroids.destroy();

  const colors = ['#10B981', '#00E5FF', '#F59E0B', '#F43F5E', '#8B5CF6', '#EC4899'];
  const datasets = Object.keys(centroids.curves).map((clusterName, idx) => ({
    label: clusterName,
    data: centroids.curves[clusterName],
    borderColor: colors[idx % colors.length],
    borderWidth: 2.5,
    tension: 0.35,
    pointRadius: 0,
    pointHoverRadius: 5
  }));

  state.charts.centroids = new Chart(ctx, {
    type: 'line',
    data: {
      labels: centroids.hours.map(h => `${h}:00`),
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#CBD5E1', boxWidth: 12, font: { size: 11 } }
        },
        tooltip: {
          backgroundColor: '#0D1527',
          borderColor: 'rgba(255,255,255,0.15)',
          borderWidth: 1,
          padding: 8,
          callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(2)} kW` }
        }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#94A3B8' } },
        y: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94A3B8' }, title: { display: true, text: 'kW', color: '#64748B' } }
      }
    }
  });
}

function renderPcaPlot(points) {
  const clusters = [...new Set(points.map(p => p.cluster))];
  const colors = ['#10B981', '#00E5FF', '#F59E0B', '#F43F5E', '#8B5CF6'];

  const traces = clusters.map((cName, idx) => {
    const subset = points.filter(p => p.cluster === cName);
    return {
      x: subset.map(p => p.pc1),
      y: subset.map(p => p.pc2),
      mode: 'markers',
      type: 'scatter',
      name: cName,
      marker: {
        color: colors[idx % colors.length],
        size: 6,
        opacity: 0.8
      },
      hovertemplate: `<b>${cName}</b><br>Date: %{text}<br>PC1: %{x:.2f}, PC2: %{y:.2f}<extra></extra>`,
      text: subset.map(p => p.date)
    };
  });

  const layout = {
    margin: { t: 20, r: 20, b: 40, l: 45 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: { title: { text: 'PC1 (Mean Intensity)', font: { color: '#94A3B8', size: 11 } }, tickfont: { color: '#94A3B8' }, gridcolor: 'rgba(255,255,255,0.05)' },
    yaxis: { title: { text: 'PC2 (Diurnal Skew)', font: { color: '#94A3B8', size: 11 } }, tickfont: { color: '#94A3B8' }, gridcolor: 'rgba(255,255,255,0.05)' },
    legend: { font: { color: '#CBD5E1', size: 10 }, orientation: 'h', y: 1.15 }
  };

  Plotly.newPlot('pcaPlot', traces, layout, { responsive: true, displayModeBar: false });
}

function renderElbowSilhouetteCharts(evalData) {
  const ks = evalData.map(e => `k=${e.k}`);
  const inertias = evalData.map(e => e.inertia);
  const silhouettes = evalData.map(e => e.silhouette);

  // Elbow Chart
  const ctxElbow = document.getElementById('chartElbow').getContext('2d');
  if (state.charts.elbow) state.charts.elbow.destroy();
  state.charts.elbow = new Chart(ctxElbow, {
    type: 'line',
    data: {
      labels: ks,
      datasets: [{
        label: 'Inertia (SSE)',
        data: inertias,
        borderColor: '#00E5FF',
        backgroundColor: 'rgba(0, 229, 255, 0.1)',
        fill: true,
        tension: 0.2,
        pointRadius: 5,
        pointBackgroundColor: '#00E5FF'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#94A3B8' } },
        y: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94A3B8' } }
      }
    }
  });

  // Silhouette Chart
  const ctxSil = document.getElementById('chartSilhouette').getContext('2d');
  if (state.charts.silhouette) state.charts.silhouette.destroy();
  state.charts.silhouette = new Chart(ctxSil, {
    type: 'line',
    data: {
      labels: ks,
      datasets: [{
        label: 'Silhouette Score',
        data: silhouettes,
        borderColor: '#10B981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.2,
        pointRadius: 5,
        pointBackgroundColor: '#10B981'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#94A3B8' } },
        y: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94A3B8' } }
      }
    }
  });
}

function renderClusterDistTable(dist) {
  const tbody = document.querySelector('#clusterDistTable tbody');
  tbody.innerHTML = '';

  const actions = {
    lowest: 'Standby audit; verify baseline idle drain',
    low: 'Standard efficient profile; benchmark baseline',
    mid: 'Standard day; monitor daytime heating/cooling',
    high: 'Demand response target; shift peak loads',
    highest: 'Critical demand; inspect heavy appliances & HVAC'
  };

  dist.forEach(row => {
    const tr = document.createElement('tr');
    let key = 'mid';
    const lower = row.cluster.toLowerCase();
    if (lower.includes('lowest')) key = 'lowest';
    else if (lower.includes('highest') || lower.includes('peak')) key = 'highest';
    else if (lower.includes('high')) key = 'high';
    else if (lower.includes('low')) key = 'low';

    tr.innerHTML = `
      <td><strong>${row.cluster}</strong></td>
      <td>${row.days} days</td>
      <td>${row.percentage.toFixed(1)}%</td>
      <td>${row.avg_kw.toFixed(2)} kW</td>
      <td><span class="badge-tag ${key}">${actions[key]}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// =====================================================================
// 4. Predictive Machine Learning Classification
// =====================================================================
async function loadClassification() {
  try {
    const res = await fetch('/api/classification');
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById('classificationThresholdText').textContent = data.tertiles.description;

    renderModelComparisonCards(data.models);
    renderConfusionMatrix(data.confusion_matrix);
    renderFeatureImportanceChart(data.feature_importance);
    renderClassificationReport(data.report);

  } catch (err) {
    console.error('Classification error:', err);
  }
}

function renderModelComparisonCards(models) {
  const container = document.getElementById('modelComparisonCards');
  container.innerHTML = '';

  models.forEach(m => {
    const isRf = m.model.toLowerCase().includes('random forest');
    const card = document.createElement('div');
    card.className = `benchmark-card ${isRf ? 'best' : ''}`;
    card.innerHTML = `
      <div class="bench-title-row">
        <span class="bench-name">${m.model}</span>
        ${isRf ? '<span class="bench-tag">Ensemble Best</span>' : ''}
      </div>
      <div class="bench-stats">
        <div class="stat-item">
          <span class="stat-label">Accuracy</span>
          <span class="stat-val">${m.accuracy.toFixed(1)}%</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Macro F1</span>
          <span class="stat-val">${m.macro_f1.toFixed(1)}%</span>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderConfusionMatrix(conf) {
  const zData = conf.matrix;
  const labels = conf.labels;

  const plotData = [{
    z: zData,
    x: labels,
    y: labels,
    type: 'heatmap',
    colorscale: [
      [0, '#0D1527'],
      [0.5, '#0099FF'],
      [1.0, '#00E5FF']
    ],
    hoverongaps: false,
    text: zData.map(row => row.map(v => `${v}`)),
    texttemplate: '%{text}',
    textfont: { color: '#070B14', size: 14, family: 'JetBrains Mono' },
    showscale: false
  }];

  const layout = {
    margin: { t: 30, r: 20, b: 50, l: 60 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: { title: { text: 'Predicted Tier', font: { color: '#CBD5E1', size: 12 } }, tickfont: { color: '#CBD5E1' } },
    yaxis: { title: { text: 'Actual Tier', font: { color: '#CBD5E1', size: 12 } }, tickfont: { color: '#CBD5E1' }, autorange: 'reversed' }
  };

  Plotly.newPlot('confusionMatrixPlot', plotData, layout, { responsive: true, displayModeBar: false });
}

function renderFeatureImportanceChart(feats) {
  const ctx = document.getElementById('chartFeatureImportance').getContext('2d');
  if (state.charts.featureImp) state.charts.featureImp.destroy();

  const labels = feats.map(f => f.feature);
  const values = feats.map(f => f.importance);

  state.charts.featureImp = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Importance %',
        data: values,
        backgroundColor: 'rgba(0, 229, 255, 0.65)',
        borderColor: '#00E5FF',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94A3B8' }, title: { display: true, text: 'Contribution %', color: '#64748B' } },
        y: { grid: { display: false }, ticks: { color: '#CBD5E1' } }
      }
    }
  });
}

function renderClassificationReport(report) {
  const tbody = document.querySelector('#classReportTable tbody');
  tbody.innerHTML = '';

  const tiers = ['Low', 'Medium', 'High'];
  tiers.forEach(tier => {
    if (report[tier]) {
      const r = report[tier];
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="badge-tag ${tier.toLowerCase()}">${tier}</span></td>
        <td>${r.precision.toFixed(1)}%</td>
        <td>${r.recall.toFixed(1)}%</td>
        <td><strong>${r.f1_score.toFixed(1)}%</strong></td>
        <td>${r.support}</td>
      `;
      tbody.appendChild(tr);
    }
  });
}

// Live ML Inference Simulator
function setupLivePredictor() {
  const form = document.getElementById('livePredictForm');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const hour = parseInt(document.getElementById('inputHour').value, 10);
    const temp_c = parseFloat(document.getElementById('inputTemp').value);
    const dow = parseInt(document.getElementById('inputDow').value, 10);
    const month = parseInt(document.getElementById('inputMonth').value, 10);
    const lag_24 = parseFloat(document.getElementById('inputLag24').value);
    const lag_168 = parseFloat(document.getElementById('inputLag168').value);
    const weekend = dow >= 5 ? 1 : 0;
    const peak = (hour >= 17 && hour <= 21) ? 1 : 0;

    const payload = { hour, dow, month, weekend, temp_c, peak, lag_24, lag_168 };

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      renderPredictionResult(data);
    } catch (err) {
      showToast('Live inference error', 'error');
    }
  });

  // Run once initially with default parameters
  form.dispatchEvent(new Event('submit'));
}

function renderPredictionResult(data) {
  const badge = document.getElementById('resTierBadge');
  badge.textContent = `Predicted Tier: ${data.level}`;
  badge.className = `result-tier-badge badge-tag ${data.level.toLowerCase()}`;

  const barsContainer = document.getElementById('probBarsContainer');
  barsContainer.innerHTML = '';

  const colors = { Low: '#10B981', Medium: '#F59E0B', High: '#F43F5E' };
  Object.keys(data.probabilities).forEach(tier => {
    const pct = data.probabilities[tier];
    const row = document.createElement('div');
    row.className = 'prob-bar-row';
    row.innerHTML = `
      <span class="prob-bar-label">${tier}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width: ${pct}%; background: ${colors[tier]};"></div>
      </div>
      <span class="prob-bar-val">${pct.toFixed(0)}%</span>
    `;
    barsContainer.appendChild(row);
  });

  const factorsList = document.getElementById('resFactorsList');
  factorsList.innerHTML = `<strong>Inference Driver:</strong> ${data.factors.join(' ')}`;
}

// =====================================================================
// 5. Association Rule Mining (Apriori)
// =====================================================================
async function loadAssociationRules() {
  try {
    const res = await fetch('/api/association-rules');
    if (!res.ok) return;
    const data = await res.json();
    state.rulesData = data.rules || [];

    setupRuleFiltering();
    renderRulesTable();

  } catch (err) {
    console.error('Association rules error:', err);
  }
}

function setupRuleFiltering() {
  const search = document.getElementById('ruleSearchInput');
  const sort = document.getElementById('ruleSortSelect');

  search.addEventListener('input', () => renderRulesTable());
  sort.addEventListener('change', () => renderRulesTable());
}

function renderRulesTable() {
  const query = document.getElementById('ruleSearchInput').value.toLowerCase();
  const sortBy = document.getElementById('ruleSortSelect').value;
  const tbody = document.querySelector('#rulesTable tbody');
  const emptyNotice = document.getElementById('rulesEmptyNotice');

  let filtered = state.rulesData.filter(r => r.rule.toLowerCase().includes(query));

  filtered.sort((a, b) => b[sortBy] - a[sortBy]);

  tbody.innerHTML = '';
  if (filtered.length === 0) {
    emptyNotice.classList.remove('hidden');
    return;
  }
  emptyNotice.classList.add('hidden');

  filtered.forEach(item => {
    const tr = document.createElement('tr');
    const [antecedents, consequent] = item.rule.split('->').map(s => s.trim());
    const isHigh = consequent.includes('High');

    tr.innerHTML = `
      <td>#${item.id}</td>
      <td><span class="rule-ant-text">${formatRuleBadges(antecedents)}</span></td>
      <td style="text-align: center; color: var(--accent-cyan); font-weight: bold;">&rarr;</td>
      <td><span class="badge-tag ${isHigh ? 'high' : 'low'}">${consequent}</span></td>
      <td>${item.support.toFixed(1)}%</td>
      <td>${item.confidence.toFixed(1)}%</td>
      <td><strong style="color: var(--accent-cyan);">${item.lift.toFixed(2)}x</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function formatRuleBadges(antecedentStr) {
  return antecedentStr.split('&').map(cond => {
    const trimmed = cond.trim();
    return `<span class="cond-pill">${trimmed}</span>`;
  }).join(' ');
}

// =====================================================================
// 6. Anomaly Detection & Time Series
// =====================================================================
async function loadAnomalies() {
  try {
    const res = await fetch('/api/anomalies');
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById('anomalyRateBadge').textContent = `Rate: ${data.summary.rate_pct}% (${data.summary.total} anomalous hours)`;

    renderAnomalySeriesChart(data.recent_series);
    renderAnomaliesTable(data.top_table);

  } catch (err) {
    console.error('Anomalies error:', err);
  }
}

function renderAnomalySeriesChart(series) {
  const ctx = document.getElementById('chartAnomalySeries').getContext('2d');
  if (state.charts.anomalies) state.charts.anomalies.destroy();

  const labels = series.map(s => s.t);
  const kwValues = series.map(s => s.kw);

  // Spikes and Drops datasets for point overlays
  const spikeData = series.map(s => (s.anomaly && s.type === 'Spike') ? s.kw : null);
  const dropData = series.map(s => (s.anomaly && s.type === 'Drop') ? s.kw : null);

  state.charts.anomalies = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Power (kW)',
          data: kwValues,
          borderColor: '#00E5FF',
          borderWidth: 1.5,
          tension: 0.1,
          pointRadius: 0
        },
        {
          label: 'Spike Anomalies',
          data: spikeData,
          borderColor: '#EF4444',
          backgroundColor: '#EF4444',
          pointRadius: 6,
          pointHoverRadius: 8,
          showLine: false
        },
        {
          label: 'Drop Faults',
          data: dropData,
          borderColor: '#F59E0B',
          backgroundColor: '#F59E0B',
          pointRadius: 6,
          pointHoverRadius: 8,
          showLine: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#CBD5E1' } },
        tooltip: {
          backgroundColor: '#0D1527',
          borderColor: 'rgba(255,255,255,0.15)',
          borderWidth: 1,
          padding: 8
        }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#94A3B8', maxTicksLimit: 12 } },
        y: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94A3B8' }, title: { display: true, text: 'kW', color: '#64748B' } }
      }
    }
  });
}

function renderAnomaliesTable(anomalies) {
  const tbody = document.querySelector('#anomaliesTable tbody');
  tbody.innerHTML = '';

  anomalies.forEach(row => {
    const tr = document.createElement('tr');
    const isSpike = row.type === 'Spike';
    const rec = isSpike
      ? 'Inspect left-on high-draw appliances or compressor cycling'
      : 'Sensor dropout or power disconnection event';

    tr.innerHTML = `
      <td>${row.timestamp}</td>
      <td>${row.dayname} &middot; ${row.hour}</td>
      <td><strong>${row.kw.toFixed(2)} kW</strong></td>
      <td style="color: ${isSpike ? '#EF4444' : '#F59E0B'}; font-family: var(--font-mono); font-weight: bold;">
        ${row.robust_z > 0 ? '+' : ''}${row.robust_z.toFixed(2)} &sigma;
      </td>
      <td><span class="badge-tag ${isSpike ? 'spike' : 'drop'}">${row.type}</span></td>
      <td>${row.temp_c !== null ? `${row.temp_c.toFixed(1)}°C` : 'N/A'}</td>
      <td style="font-size: 0.8rem; color: var(--text-muted);">${rec}</td>
    `;
    tbody.appendChild(tr);
  });
}

// =====================================================================
// 7. AI Insights & What-If Scenario Simulator
// =====================================================================
async function loadInsights() {
  try {
    const res = await fetch('/api/insights');
    if (!res.ok) return;
    const data = await res.json();

    const container = document.getElementById('insightsCardsContainer');
    container.innerHTML = '';

    data.insights.forEach(item => {
      const card = document.createElement('div');
      card.className = 'insight-card';
      card.innerHTML = `
        <div class="insight-header">
          <div class="insight-category-row">
            <div class="insight-icon-box" style="background: rgba(0, 229, 255, 0.12); color: var(--accent-cyan);">
              <i data-lucide="${item.icon}"></i>
            </div>
            <span class="insight-cat-name">${item.category}</span>
          </div>
          <span class="insight-tag" style="background: rgba(16, 185, 129, 0.12); color: var(--accent-emerald);">${item.tag}</span>
        </div>
        <p class="insight-body">${item.text}</p>
      `;
      container.appendChild(card);
    });

    initIcons();

  } catch (err) {
    console.error('Insights error:', err);
  }
}

// Simulator Listeners
function setupSimulator() {
  const peakSlider = document.getElementById('simPeakShift');
  const standbySlider = document.getElementById('simStandby');
  const peakTariffInput = document.getElementById('simPeakTariff');
  const offpeakTariffInput = document.getElementById('simOffpeakTariff');

  const updateSim = () => {
    document.getElementById('valSimPeakShift').textContent = `${peakSlider.value}%`;
    document.getElementById('valSimStandby').textContent = `${standbySlider.value}%`;
    triggerSimulation();
  };

  peakSlider.addEventListener('input', updateSim);
  standbySlider.addEventListener('input', updateSim);
  peakTariffInput.addEventListener('input', updateSim);
  offpeakTariffInput.addEventListener('input', updateSim);
}

let simTimeout = null;
function triggerSimulation() {
  clearTimeout(simTimeout);
  simTimeout = setTimeout(async () => {
    const payload = {
      peak_shift_pct: parseFloat(document.getElementById('simPeakShift').value),
      standby_reduction_pct: parseFloat(document.getElementById('simStandby').value),
      peak_rate: parseFloat(document.getElementById('simPeakTariff').value),
      offpeak_rate: parseFloat(document.getElementById('simOffpeakTariff').value)
    };

    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) return;
      const data = await res.json();

      document.getElementById('simDollarSaved').textContent = `$ ${data.savings.annual_dollar_saved.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
      document.getElementById('simPctSaved').textContent = `${data.savings.pct_cost_saved}% Total Bill Cut`;

      document.getElementById('simKwhSaved').textContent = `${Math.round(data.savings.annual_kwh_eliminated).toLocaleString()} kWh/yr`;
      document.getElementById('simKwhShifted').textContent = `${Math.round(data.savings.annual_kwh_shifted).toLocaleString()} kWh Shifted to Off-Peak`;

      document.getElementById('simCo2Saved').textContent = `${Math.round(data.savings.co2_avoided_kg).toLocaleString()} kg CO₂`;
      document.getElementById('simTreesEquiv').textContent = `Equivalent to ~${Math.round(data.savings.trees_planted_equiv)} mature trees planted`;

    } catch (err) {
      console.error('Simulator error:', err);
    }
  }, 100);
}

// =====================================================================
// 8. Upload Modal & Drag-and-Drop
// =====================================================================
function setupUploadModal() {
  const modal = document.getElementById('uploadModal');
  const openBtn = document.getElementById('openUploadBtn');
  const closeBtn = document.getElementById('closeUploadModalBtn');
  const cancelBtn = document.getElementById('cancelUploadBtn');
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('csvFileInput');
  const confirmBtn = document.getElementById('confirmUploadBtn');

  openBtn.addEventListener('click', () => modal.classList.remove('hidden'));
  const closeModal = () => {
    modal.classList.add('hidden');
    state.selectedFile = null;
    fileInput.value = '';
    confirmBtn.disabled = true;
    dropZone.querySelector('.drop-title').innerHTML = `Drag & drop your CSV file here, or <span class="browse-link">browse</span>`;
  };

  closeBtn.addEventListener('click', closeModal);
  cancelBtn.addEventListener('click', closeModal);

  // File Picker
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag and drop events
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
  });
  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.name.endsWith('.csv')) {
      showToast('Please select a valid .csv file', 'error');
      return;
    }
    state.selectedFile = file;
    dropZone.querySelector('.drop-title').innerHTML = `Selected: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(1)} KB)`;
    confirmBtn.disabled = false;
  }

  // Confirm Upload
  confirmBtn.addEventListener('click', async () => {
    if (!state.selectedFile) return;

    closeModal();
    showLoader('Training Energy Mining Models...', `Analyzing ${state.selectedFile.name}, extracting diurnal patterns, clustering, and evaluating algorithms...`);

    const formData = new FormData();
    formData.append('file', state.selectedFile);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (res.ok) {
        showToast(`Analyzed ${data.records} records successfully!`, 'success');
        await loadAllData();
      } else {
        showToast(data.detail || 'Failed to process dataset', 'error');
      }
    } catch (err) {
      showToast('Upload error occurred', 'error');
    } finally {
      hideLoader();
    }
  });
}

// =====================================================================
// UI Helpers: Toast & Loader
// =====================================================================
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  const iconName = type === 'success' ? 'check-circle' : type === 'error' ? 'alert-octagon' : 'info';
  const color = type === 'success' ? 'var(--accent-emerald)' : type === 'error' ? '#EF4444' : 'var(--accent-cyan)';

  toast.innerHTML = `
    <i data-lucide="${iconName}" style="color: ${color}; width: 18px; height: 18px;"></i>
    <span>${message}</span>
  `;
  container.appendChild(toast);
  initIcons();

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(40px)';
    setTimeout(() => toast.remove(), 250);
  }, 4000);
}

function showLoader(title, desc) {
  document.getElementById('loaderTitle').textContent = title;
  document.getElementById('loaderDesc').textContent = desc;
  document.getElementById('globalLoader').classList.remove('hidden');
}

function hideLoader() {
  document.getElementById('globalLoader').classList.add('hidden');
}
