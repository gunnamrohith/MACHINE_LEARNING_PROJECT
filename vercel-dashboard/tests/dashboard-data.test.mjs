import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const data = JSON.parse(await readFile(new URL("../src/data/dashboard.json", import.meta.url), "utf8"));

test("generated dashboard data has valid core metrics", () => {
  assert.ok(data.sourceRows > 0);
  assert.ok(data.kpis.revenue > 0);
  assert.ok(data.kpis.customers > 0);
  assert.ok(data.kpis.products > 0);
  assert.ok(data.kpis.purchases > 0);
});

test("probabilities and prediction inputs are usable", () => {
  const total = data.probability.events.reduce((sum, event) => sum + event.value, 0);
  assert.ok(Math.abs(total - 1) < 0.000001);
  assert.ok(data.predictions.categories.length > 0);
  assert.ok(data.predictions.brands.length > 0);
  assert.ok(data.predictions.purchaseRate > 0 && data.predictions.purchaseRate < 1);
});
