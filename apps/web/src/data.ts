import type { RunData, RunIndexEntry } from "./types";

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${url}`);
  }
  return response.json() as Promise<T>;
}

async function fetchJsonl<T>(url: string): Promise<T[]> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${url}`);
  }
  const text = await response.text();
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

export async function listRuns(): Promise<RunIndexEntry[]> {
  return fetchJson<RunIndexEntry[]>("/runs/index.json");
}

export async function loadRun(entry: RunIndexEntry): Promise<RunData> {
  const base = `/runs/${entry.path}`;
  const [manifest, readouts, patches, layerScan, projections, metrics] = await Promise.all([
    fetchJson(`${base}/manifest.json`),
    fetchJsonl(`${base}/readouts.jsonl`),
    fetchJsonl(`${base}/patches.jsonl`),
    fetchJson(`${base}/layer_scan.json`),
    fetchJson(`${base}/projections.json`),
    fetchJson(`${base}/metrics.json`),
  ]);

  return {
    manifest,
    readouts,
    patches,
    layerScan,
    projections,
    metrics,
  } as RunData;
}
