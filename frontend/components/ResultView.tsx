"use client";

import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";
import { useRef, useState } from "react";
import { Bar, getElementAtEvent, Line, Pie, Scatter } from "react-chartjs-2";
import type { DriverFactor, QueryResult, Visualization } from "@/lib/types";

ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Tooltip, Legend);

const PALETTE = ["#4338ca", "#0e9384", "#dc6803", "#c11574", "#1570ef", "#6938ef", "#099250", "#b42318"];
export const columnLabel = (c: string, labels?: Record<string, string>) => labels?.[c] ?? c.replace(/_/g, " ");

export function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return String(v);
}

export function downloadCsv(columns: string[], rows: unknown[][], name: string) {
  const escape = (v: unknown) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const csv = [columns.map(escape).join(","), ...rows.map((r) => r.map(escape).join(","))].join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `${name.replace(/[^a-z0-9]+/gi, "_").slice(0, 60) || "result"}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export function DataTable({
  columns,
  rows,
  labels,
  onRowClick,
  maxHeight,
}: {
  columns: string[];
  rows: (string | number | null)[][];
  labels?: Record<string, string>;
  onRowClick?: (rowIndex: number) => void;
  maxHeight?: number | string;
}) {
  const numeric = columns.map((_, i) => rows.length > 0 && rows.every((r) => r[i] === null || typeof r[i] === "number"));
  return (
    <div className="table-wrap" style={maxHeight !== undefined ? { maxHeight } : undefined}>
      <table className="data">
        <thead>
          <tr>
            {columns.map((c, i) => (
              <th key={c + i} style={numeric[i] ? { textAlign: "right" } : undefined}>{columnLabel(c, labels)}</th>
            ))}
            {onRowClick && <th aria-label="Details" />}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className={onRowClick ? "clickable" : undefined} onClick={onRowClick ? () => onRowClick(ri) : undefined}>
              {row.map((cell, ci) => (
                <td key={ci} className={numeric[ci] ? "num" : undefined}>{fmt(cell)}</td>
              ))}
              {onRowClick && <td className="row-hint">View rows →</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ResultChart({
  viz,
  result,
  onSelect,
}: {
  viz: Visualization;
  result: QueryResult;
  onSelect?: (index: number) => void;
}) {
  const ref = useRef<ChartJS<any>>(null);
  const clickable = Boolean(onSelect) && viz.key_column !== null;
  const handleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!clickable || !ref.current) return;
    const el = getElementAtEvent(ref.current, event);
    if (el.length) onSelect!(el[0].index);
  };

  if (viz.chart_type === "drivers") {
    return <DriversChart viz={viz} />;
  }

  if (viz.chart_type === "kpi") {
    const idx = viz.value_column ? result.columns.indexOf(viz.value_column) : -1;
    return (
      <div className="kpi">
        <div className="kpi-value">{idx >= 0 ? fmt(result.rows[0][idx]) : "—"}</div>
        <div className="kpi-label">{viz.value_label}</div>
      </div>
    );
  }

  if (viz.chart_type === "scatter") {
    const r = viz.stats?.pearson_r;
    return (
      <>
        <div className="chart-box" style={{ height: 340 }}>
          <Scatter
            data={{
              datasets: [{
                label: "Records",
                data: viz.points.map((p) => ({ x: p.x, y: p.y })),
                backgroundColor: "rgba(67, 56, 202, 0.55)",
                pointRadius: 4,
                pointHoverRadius: 6,
              }],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { display: false },
                tooltip: {
                  callbacks: {
                    label: (ctx) => `${viz.points[ctx.dataIndex]?.label}: ${fmt(ctx.parsed.x)}, ${fmt(ctx.parsed.y)}`,
                  },
                },
              },
              scales: {
                x: { title: { display: true, text: viz.x_label ?? "" } },
                y: { title: { display: true, text: viz.y_label ?? "" } },
              },
            }}
          />
        </div>
        {r !== null && r !== undefined && (
          <div className="stat-strip">
            <div><span className="subtle">Correlation (r)</span><strong>{r.toFixed(2)}</strong></div>
            <div><span className="subtle">Records</span><strong>{viz.stats?.n.toLocaleString()}</strong></div>
            <div className="corr-scale" aria-hidden>
              <span style={{ left: `${((r + 1) / 2) * 100}%` }} />
            </div>
          </div>
        )}
      </>
    );
  }

  if (!["bar", "line", "pie", "histogram"].includes(viz.chart_type)) return null;
  if (viz.chart_type === "bar" && viz.labels.length === 1) {
    return (
      <div className="kpi">
        <div className="kpi-value">{fmt(viz.values[0])}</div>
        <div className="kpi-label">{viz.value_label} · {viz.labels[0]}</div>
      </div>
    );
  }

  const isPie = viz.chart_type === "pie";
  const dataset = {
    label: viz.value_label ?? "",
    data: viz.values,
    backgroundColor: isPie ? viz.labels.map((_, i) => PALETTE[i % PALETTE.length]) : "rgba(67, 56, 202, 0.85)",
    hoverBackgroundColor: isPie ? undefined : "#312e81",
    borderColor: viz.chart_type === "line" ? "#4338ca" : undefined,
    borderRadius: viz.chart_type === "bar" ? 4 : 0,
    barPercentage: viz.chart_type === "histogram" ? 0.98 : 0.8,
    categoryPercentage: viz.chart_type === "histogram" ? 1 : 0.8,
    tension: 0.3,
    pointRadius: 5,
  };
  const data = { labels: viz.labels, datasets: [dataset] };
  const common = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    onHover: undefined,
  } as const;
  const height = viz.horizontal ? Math.max(220, viz.labels.length * 26 + 40) : 300;
  const cursor = clickable ? "pointer" : "default";

  return (
    <div className="chart-box" style={{ height, cursor }}>
      {isPie && (
        <Pie ref={ref as never} onClick={handleClick} data={data}
          options={{ ...common, plugins: { legend: { position: "right" } } }} />
      )}
      {viz.chart_type === "line" && (
        <Line ref={ref as never} onClick={handleClick} data={data}
          options={{ ...common, plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: viz.value_label ?? "" } } } }} />
      )}
      {(viz.chart_type === "bar" || viz.chart_type === "histogram") && (
        <Bar
          ref={ref as never}
          onClick={handleClick}
          data={data}
          options={{
            ...common,
            indexAxis: viz.horizontal ? "y" : "x",
            plugins: { legend: { display: false } },
            scales: {
              [viz.horizontal ? "x" : "y"]: { title: { display: true, text: viz.value_label ?? "" }, beginAtZero: true },
              ...(viz.x_label ? { [viz.horizontal ? "y" : "x"]: { title: { display: true, text: viz.x_label } } } : {}),
            },
          }}
        />
      )}
    </div>
  );
}

export function DriversChart({ viz }: { viz: Visualization }) {
  const d = viz.drivers;
  const [open, setOpen] = useState<string | null>(d?.factors[0]?.factor ?? null);
  if (!d || d.factors.length === 0) return null;
  const top = d.factors.slice(0, 10);
  const max = Math.max(...top.map((f) => f.spread), 0.0001);
  const unit = d.unit === "%" ? " pts" : d.unit;
  return (
    <div className="drivers">
      <div className="drivers-head subtle">
        Gap in {d.outcome} between each factor&apos;s best and worst level · click a factor for detail
      </div>
      {top.map((f: DriverFactor) => (
        <div key={f.factor} className={`driver ${open === f.factor ? "open" : ""}`}>
          <button className="driver-row" onClick={() => setOpen(open === f.factor ? null : f.factor)} aria-expanded={open === f.factor}>
            <span className="driver-label">{f.label.charAt(0).toUpperCase() + f.label.slice(1)}</span>
            <span className="driver-bar"><span style={{ width: `${(f.spread / max) * 100}%` }} /></span>
            <span className="driver-value">{fmt(f.spread)}{unit}</span>
          </button>
          {open === f.factor && (
            <div className="driver-detail">
              <table className="data">
                <thead>
                  <tr><th>Level</th><th style={{ textAlign: "right" }}>{d.outcome}</th><th style={{ textAlign: "right" }}>Records</th></tr>
                </thead>
                <tbody>
                  {f.levels.map((lv) => (
                    <tr key={lv.level} className={lv.supported ? undefined : "muted-row"}>
                      <td>
                        {lv.level}
                        {lv.level === f.best.level && <span className="badge badge-success" style={{ marginLeft: 6 }}>best</span>}
                        {lv.level === f.worst.level && <span className="badge badge-danger" style={{ marginLeft: 6 }}>worst</span>}
                        {!lv.supported && <span className="subtle" style={{ marginLeft: 6, fontSize: 12 }}>too few to rank</span>}
                      </td>
                      <td className="num">{lv.value === null ? "—" : `${fmt(lv.value)}${d.unit === "%" ? "%" : ""}`}</td>
                      <td className="num">{lv.n.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
