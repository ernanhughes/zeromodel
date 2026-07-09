"use strict";

const metrics = ["quality", "uncertainty", "novelty", "safety", "cost", "latency"];

const sourceRows = [
  { id: "item-001", label: "Aster", values: [0.82, 0.18, 0.64, 0.91, 0.42, 0.31] },
  { id: "item-002", label: "Beryl", values: [0.61, 0.77, 0.89, 0.72, 0.28, 0.44] },
  { id: "item-003", label: "Cinder", values: [0.94, 0.32, 0.38, 0.86, 0.67, 0.21] },
  { id: "item-004", label: "Delta", values: [0.48, 0.91, 0.74, 0.63, 0.19, 0.57] },
  { id: "item-005", label: "Elm", values: [0.73, 0.41, 0.52, 0.95, 0.36, 0.26] },
  { id: "item-006", label: "Fjord", values: [0.39, 0.66, 0.93, 0.58, 0.21, 0.73] },
  { id: "item-007", label: "Glyph", values: [0.88, 0.54, 0.71, 0.79, 0.83, 0.35] },
  { id: "item-008", label: "Harbor", values: [0.57, 0.23, 0.44, 0.89, 0.31, 0.18] },
  { id: "item-009", label: "Ion", values: [0.69, 0.84, 0.81, 0.68, 0.46, 0.62] },
  { id: "item-010", label: "Juniper", values: [0.77, 0.37, 0.59, 0.83, 0.54, 0.29] },
  { id: "item-011", label: "Kite", values: [0.52, 0.59, 0.67, 0.76, 0.14, 0.48] },
  { id: "item-012", label: "Lumen", values: [0.91, 0.71, 0.96, 0.74, 0.72, 0.39] }
];

const recipes = {
  quality: {
    version: "vpm-layout/0",
    name: "quality-first",
    row_order: {
      kind: "lexicographic",
      keys: [
        { metric_id: "quality", direction: "desc" },
        { metric_id: "uncertainty", direction: "asc" }
      ],
      tie_break: "row_id"
    },
    column_order: {
      kind: "explicit",
      metric_ids: ["quality", "safety", "uncertainty", "novelty", "cost", "latency"]
    },
    normalization: { kind: "per_metric_minmax", clip: true }
  },
  investigate: {
    version: "vpm-layout/0",
    name: "investigate",
    row_order: {
      kind: "lexicographic",
      keys: [
        { metric_id: "uncertainty", direction: "desc" },
        { metric_id: "novelty", direction: "desc" }
      ],
      tie_break: "row_id"
    },
    column_order: {
      kind: "explicit",
      metric_ids: ["uncertainty", "novelty", "quality", "safety", "latency", "cost"]
    },
    normalization: { kind: "per_metric_minmax", clip: true }
  },
  efficient: {
    version: "vpm-layout/0",
    name: "efficient",
    row_order: {
      kind: "lexicographic",
      keys: [
        { metric_id: "quality", direction: "desc" },
        { metric_id: "cost", direction: "asc" },
        { metric_id: "latency", direction: "asc" }
      ],
      tie_break: "row_id"
    },
    column_order: {
      kind: "explicit",
      metric_ids: ["quality", "cost", "latency", "safety", "uncertainty", "novelty"]
    },
    normalization: { kind: "per_metric_minmax", clip: true }
  },
  source: {
    version: "vpm-layout/0",
    name: "source-order",
    row_order: { kind: "source", tie_break: "row_id" },
    column_order: { kind: "source" },
    normalization: { kind: "per_metric_minmax", clip: true }
  }
};

const canvas = document.querySelector("#vpm-canvas");
const context = canvas.getContext("2d");
const recipeButtons = [...document.querySelectorAll(".recipe-button")];
const regionInput = document.querySelector("#region-size");
const regionOutput = document.querySelector("#region-output");
const regionMean = document.querySelector("#region-mean");
const viewName = document.querySelector("#view-name");
const recipeJson = document.querySelector("#recipe-json");
const tooltip = document.querySelector("#cell-tooltip");
const emptyState = document.querySelector("#cell-empty");
const cellDetails = document.querySelector("#cell-details");
const detailView = document.querySelector("#detail-view");
const detailRow = document.querySelector("#detail-row");
const detailMetric = document.querySelector("#detail-metric");
const detailRaw = document.querySelector("#detail-raw");
const detailNormalized = document.querySelector("#detail-normalized");

const frame = {
  left: 142,
  top: 82,
  right: 28,
  bottom: 28
};

let activeRecipeKey = "quality";
let activeView = null;
let hoveredCell = null;

function metricIndex(metricId) {
  const index = metrics.indexOf(metricId);
  if (index < 0) {
    throw new Error(`Unknown metric: ${metricId}`);
  }
  return index;
}

function normalizeSource() {
  const mins = metrics.map((_, metric) => Math.min(...sourceRows.map((row) => row.values[metric])));
  const maxs = metrics.map((_, metric) => Math.max(...sourceRows.map((row) => row.values[metric])));

  return sourceRows.map((row) => ({
    ...row,
    normalized: row.values.map((value, metric) => {
      const range = maxs[metric] - mins[metric];
      return range === 0 ? 0 : (value - mins[metric]) / range;
    })
  }));
}

const normalizedRows = normalizeSource();

function compareRows(left, right, recipe) {
  if (recipe.row_order.kind === "source") {
    return sourceRows.findIndex((row) => row.id === left.id) - sourceRows.findIndex((row) => row.id === right.id);
  }

  for (const key of recipe.row_order.keys) {
    const index = metricIndex(key.metric_id);
    const delta = left.values[index] - right.values[index];
    if (Math.abs(delta) > Number.EPSILON) {
      return key.direction === "desc" ? -delta : delta;
    }
  }

  return left.id.localeCompare(right.id);
}

function compileView(recipeKey) {
  const recipe = recipes[recipeKey];
  const orderedRows = [...normalizedRows].sort((left, right) => compareRows(left, right, recipe));
  const orderedMetrics = recipe.column_order.kind === "source"
    ? [...metrics]
    : [...recipe.column_order.metric_ids];

  return {
    recipe,
    rows: orderedRows,
    metrics: orderedMetrics,
    cells: orderedRows.map((row) => orderedMetrics.map((metricId) => {
      const sourceMetricIndex = metricIndex(metricId);
      return {
        rowId: row.id,
        rowLabel: row.label,
        metricId,
        sourceRowIndex: sourceRows.findIndex((source) => source.id === row.id),
        sourceMetricIndex,
        raw: row.values[sourceMetricIndex],
        normalized: row.normalized[sourceMetricIndex]
      };
    }))
  };
}

function colorFor(value) {
  const clamped = Math.max(0, Math.min(1, value));
  const hue = 150 - (clamped * 7);
  const saturation = 32 + (clamped * 50);
  const lightness = 8 + (clamped * 66);
  return `hsl(${hue} ${saturation}% ${lightness}%)`;
}

function textColorFor(value) {
  return value > 0.66 ? "#04110b" : value > 0.36 ? "#d8f8e6" : "#90aa9e";
}

function gridGeometry() {
  const width = canvas.width - frame.left - frame.right;
  const height = canvas.height - frame.top - frame.bottom;
  return {
    width,
    height,
    cellWidth: width / activeView.metrics.length,
    cellHeight: height / activeView.rows.length
  };
}

function drawBackground() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#07100e";
  context.fillRect(0, 0, canvas.width, canvas.height);

  const gradient = context.createRadialGradient(130, 80, 0, 130, 80, 520);
  gradient.addColorStop(0, "rgba(50, 206, 129, 0.07)");
  gradient.addColorStop(1, "rgba(50, 206, 129, 0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, canvas.width, canvas.height);
}

function drawLabels(geometry) {
  context.textBaseline = "middle";
  context.font = "600 14px ui-monospace, SFMono-Regular, Menlo, monospace";

  activeView.metrics.forEach((metricId, column) => {
    const x = frame.left + (column * geometry.cellWidth) + (geometry.cellWidth / 2);
    context.save();
    context.translate(x, frame.top - 14);
    context.rotate(-0.55);
    context.fillStyle = "#82998e";
    context.textAlign = "left";
    context.fillText(metricId, 0, 0);
    context.restore();
  });

  activeView.rows.forEach((row, rowIndex) => {
    const y = frame.top + (rowIndex * geometry.cellHeight) + (geometry.cellHeight / 2);
    context.textAlign = "right";
    context.fillStyle = "#9bb0a6";
    context.font = "700 14px Inter, system-ui, sans-serif";
    context.fillText(row.label, frame.left - 16, y - 7);
    context.fillStyle = "#526a5f";
    context.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.fillText(row.id, frame.left - 16, y + 10);
  });
}

function drawCells(geometry) {
  const gap = 3;

  activeView.cells.forEach((row, rowIndex) => {
    row.forEach((cell, columnIndex) => {
      const x = frame.left + (columnIndex * geometry.cellWidth);
      const y = frame.top + (rowIndex * geometry.cellHeight);
      const isHovered = hoveredCell && hoveredCell.row === rowIndex && hoveredCell.column === columnIndex;

      context.fillStyle = colorFor(cell.normalized);
      context.fillRect(
        x + gap / 2,
        y + gap / 2,
        geometry.cellWidth - gap,
        geometry.cellHeight - gap
      );

      context.fillStyle = textColorFor(cell.normalized);
      context.font = "700 12px ui-monospace, SFMono-Regular, Menlo, monospace";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(
        cell.normalized.toFixed(2),
        x + geometry.cellWidth / 2,
        y + geometry.cellHeight / 2
      );

      if (isHovered) {
        context.strokeStyle = "#ffffff";
        context.lineWidth = 2;
        context.strokeRect(
          x + 2,
          y + 2,
          geometry.cellWidth - 4,
          geometry.cellHeight - 4
        );
      }
    });
  });
}

function drawRegion(geometry) {
  const size = Number(regionInput.value);
  const rows = Math.min(size, activeView.rows.length);
  const columns = Math.min(size, activeView.metrics.length);
  const width = columns * geometry.cellWidth;
  const height = rows * geometry.cellHeight;

  context.fillStyle = "rgba(123, 242, 173, 0.075)";
  context.fillRect(frame.left, frame.top, width, height);
  context.strokeStyle = "#7bf2ad";
  context.lineWidth = 3;
  context.strokeRect(frame.left + 1.5, frame.top + 1.5, width - 3, height - 3);

  context.fillStyle = "#7bf2ad";
  context.font = "800 11px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.textAlign = "left";
  context.textBaseline = "bottom";
  context.fillText(`TOP-LEFT ${rows}×${columns}`, frame.left + 8, frame.top - 7);
}

function updateRegionMeasurement() {
  const size = Number(regionInput.value);
  const cells = activeView.cells
    .slice(0, size)
    .flatMap((row) => row.slice(0, size));
  const mean = cells.reduce((total, cell) => total + cell.normalized, 0) / cells.length;

  regionOutput.value = `${size} × ${size}`;
  regionMean.textContent = mean.toFixed(3);
}

function draw() {
  drawBackground();
  const geometry = gridGeometry();
  drawLabels(geometry);
  drawCells(geometry);
  drawRegion(geometry);
  updateRegionMeasurement();
}

function inspectCell(row, column) {
  const cell = activeView.cells[row][column];
  hoveredCell = { row, column };

  emptyState.hidden = true;
  cellDetails.hidden = false;
  detailView.textContent = `(${row}, ${column})`;
  detailRow.textContent = `${cell.rowId} · source[${cell.sourceRowIndex}]`;
  detailMetric.textContent = `${cell.metricId} · source[${cell.sourceMetricIndex}]`;
  detailRaw.textContent = cell.raw.toFixed(3);
  detailNormalized.textContent = cell.normalized.toFixed(3);

  tooltip.textContent = `${cell.rowId} × ${cell.metricId} = ${cell.normalized.toFixed(3)}`;
  tooltip.hidden = false;
  draw();
}

function clearInspection() {
  hoveredCell = null;
  tooltip.hidden = true;
  draw();
}

function pointerToCell(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;
  const geometry = gridGeometry();

  if (
    x < frame.left ||
    y < frame.top ||
    x >= frame.left + geometry.width ||
    y >= frame.top + geometry.height
  ) {
    return null;
  }

  return {
    row: Math.floor((y - frame.top) / geometry.cellHeight),
    column: Math.floor((x - frame.left) / geometry.cellWidth)
  };
}

function positionTooltip(event) {
  const wrap = canvas.parentElement.getBoundingClientRect();
  const left = Math.min(event.clientX - wrap.left + 14, wrap.width - 230);
  const top = Math.min(event.clientY - wrap.top + 14, wrap.height - 62);
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, top)}px`;
}

function activateRecipe(recipeKey) {
  activeRecipeKey = recipeKey;
  activeView = compileView(recipeKey);
  hoveredCell = null;
  tooltip.hidden = true;

  recipeButtons.forEach((button) => {
    const isActive = button.dataset.recipe === recipeKey;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });

  viewName.textContent = activeView.recipe.name;
  recipeJson.textContent = JSON.stringify(activeView.recipe, null, 2);
  emptyState.hidden = false;
  cellDetails.hidden = true;
  draw();
}

recipeButtons.forEach((button) => {
  button.addEventListener("click", () => activateRecipe(button.dataset.recipe));
});

regionInput.addEventListener("input", draw);

canvas.addEventListener("mousemove", (event) => {
  const cell = pointerToCell(event);
  if (!cell) {
    clearInspection();
    return;
  }

  inspectCell(cell.row, cell.column);
  positionTooltip(event);
});

canvas.addEventListener("mouseleave", clearInspection);

canvas.addEventListener("click", (event) => {
  const cell = pointerToCell(event);
  if (cell) {
    inspectCell(cell.row, cell.column);
    positionTooltip(event);
  }
});

activateRecipe(activeRecipeKey);
