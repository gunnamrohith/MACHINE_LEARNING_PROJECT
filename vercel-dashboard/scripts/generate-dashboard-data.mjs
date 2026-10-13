import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const inputPath = resolve(root, "../retail-sales-dashboard/data/raw/sample_ecommerce.csv");
const outputPath = resolve(root, "src/data/dashboard.json");

function parseCsv(text) {
  const [header, ...lines] = text.trim().split(/\r?\n/);
  const keys = header.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(keys.map((key, index) => [key, values[index] ?? ""]));
  });
}

function increment(map, key, amount = 1) {
  map.set(key, (map.get(key) ?? 0) + amount);
}

function sortedEntries(map, valueKey, limit) {
  return [...map.entries()]
    .map(([name, value]) => ({ name, [valueKey]: value }))
    .sort((a, b) => b[valueKey] - a[valueKey])
    .slice(0, limit);
}

const rows = parseCsv(await readFile(inputPath, "utf8"));
if (!rows.length) throw new Error(`No retail events found in ${inputPath}`);

const events = { view: 0, cart: 0, purchase: 0 };
const customers = new Set();
const products = new Set();
const buyers = new Set();
const buyerOrders = new Map();
const categoryEvents = new Map();
const categoryRevenue = new Map();
const dailyRevenue = new Map();
const monthlyRevenue = new Map();
const customerActivity = new Map();
const customerSpend = new Map();
const productViews = new Map();
const productPurchases = new Map();
const prices = [];
const weekdayPurchasePrices = [];
const weekendPurchasePrices = [];
const categoryPrices = new Map();
const brandPrices = new Map();
const categoryPurchase = new Map();
const brandPurchase = new Map();
const hourPurchase = new Map();
const weekdayPurchase = new Map();
const categories = new Set();
const brands = new Set();

function rateBucket(map, key, purchased) {
  const current = map.get(key) ?? { events: 0, purchases: 0 };
  current.events += 1;
  current.purchases += purchased ? 1 : 0;
  map.set(key, current);
}

function priceBucket(map, key, price) {
  const current = map.get(key) ?? { total: 0, count: 0 };
  current.total += price;
  current.count += 1;
  map.set(key, current);
}

for (const row of rows) {
  const date = new Date(row.event_time.replace(" UTC", "Z"));
  const price = Number(row.price);
  const event = row.event_type;
  const category = (row.category_code || "unknown").split(".")[0];
  const brand = row.brand || "unknown";
  const day = (date.getUTCDay() + 6) % 7;
  const hour = date.getUTCHours();
  const purchased = event === "purchase";

  if (event in events) events[event] += 1;
  customers.add(row.user_id);
  products.add(row.product_id);
  categories.add(category);
  brands.add(brand);
  increment(categoryEvents, category);
  increment(customerActivity, row.user_id);
  prices.push(price);
  rateBucket(categoryPurchase, category, purchased);
  rateBucket(brandPurchase, brand, purchased);
  rateBucket(hourPurchase, String(hour), purchased);
  rateBucket(weekdayPurchase, String(day), purchased);

  if (event === "view") increment(productViews, row.product_id);
  if (!purchased) continue;

  buyers.add(row.user_id);
  increment(buyerOrders, row.user_id);
  increment(customerSpend, row.user_id, price);
  increment(productPurchases, row.product_id);
  increment(categoryRevenue, category, price);
  increment(dailyRevenue, date.toISOString().slice(0, 10), price);
  increment(monthlyRevenue, date.toISOString().slice(0, 7), price);
  priceBucket(categoryPrices, category, price);
  priceBucket(brandPrices, brand, price);
  (day >= 5 ? weekendPurchasePrices : weekdayPurchasePrices).push(price);
}

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const standardDeviation = (values, average) =>
  Math.sqrt(values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1));
const priceMean = mean(prices);
const priceStd = standardDeviation(prices, priceMean);
const margin = 1.96 * priceStd / Math.sqrt(prices.length);
const purchaseCount = events.purchase;
const repeatBuyers = [...buyerOrders.values()].filter((count) => count > 1).length;
const activityBins = new Map();
for (const count of customerActivity.values()) {
  const label = count === 1 ? "1" : count <= 3 ? "2–3" : count <= 5 ? "4–5" : count <= 10 ? "6–10" : "11+";
  increment(activityBins, label);
}

const toSeries = (map, valueKey) =>
  [...map.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([name, value]) => ({ name, [valueKey]: Number(value.toFixed(2)) }));
const toAverages = (map) =>
  Object.fromEntries([...map].map(([key, value]) => [key, Number((value.total / value.count).toFixed(2))]));
const toRates = (map) =>
  Object.fromEntries([...map].map(([key, value]) => [key, Number((value.purchases / value.events).toFixed(6))]));

const data = {
  generatedAt: new Date().toISOString(),
  sourceRows: rows.length,
  dateRange: {
    start: rows.reduce((min, row) => row.event_time < min ? row.event_time : min, rows[0].event_time).slice(0, 10),
    end: rows.reduce((max, row) => row.event_time > max ? row.event_time : max, rows[0].event_time).slice(0, 10),
  },
  kpis: {
    revenue: Number([...categoryRevenue.values()].reduce((a, b) => a + b, 0).toFixed(2)),
    customers: customers.size,
    products: products.size,
    purchases: purchaseCount,
    buyers: buyers.size,
    repeatBuyerRate: Number((repeatBuyers / buyers.size * 100).toFixed(2)),
    averageOrderValue: Number(([...categoryRevenue.values()].reduce((a, b) => a + b, 0) / purchaseCount).toFixed(2)),
  },
  sales: {
    byCategory: sortedEntries(categoryRevenue, "value"),
    daily: toSeries(dailyRevenue, "value"),
    monthly: toSeries(monthlyRevenue, "value"),
  },
  customers: {
    activity: ["1", "2–3", "4–5", "6–10", "11+"].map((name) => ({ name, value: activityBins.get(name) ?? 0 })),
    top: sortedEntries(customerSpend, "value", 10).map((item) => ({
      ...item,
      name: `User …${item.name.slice(-5)}`,
      orders: buyerOrders.get(item.name) ?? 0,
    })),
  },
  products: {
    topViewed: sortedEntries(productViews, "value", 10),
    topPurchased: sortedEntries(productPurchases, "value", 10),
    categoryPopularity: sortedEntries(categoryEvents, "value"),
  },
  probability: {
    events: [
      { name: "View", value: events.view / rows.length },
      { name: "Cart", value: events.cart / rows.length },
      { name: "Purchase", value: events.purchase / rows.length },
    ],
    funnel: [
      { name: "Views", value: events.view },
      { name: "Cart", value: events.cart },
      { name: "Purchases", value: events.purchase },
    ],
    viewToCartRate: events.cart / events.view,
    cartToPurchaseRate: events.purchase / events.cart,
    overallConversionRate: events.purchase / events.view,
  },
  statistics: {
    priceMean: Number(priceMean.toFixed(2)),
    confidenceLower: Number((priceMean - margin).toFixed(2)),
    confidenceUpper: Number((priceMean + margin).toFixed(2)),
    weekdayMean: Number(mean(weekdayPurchasePrices).toFixed(2)),
    weekendMean: Number(mean(weekendPurchasePrices).toFixed(2)),
  },
  predictions: {
    categories: [...categories].sort(),
    brands: [...brands].sort(),
    overallPrice: Number(mean([...weekdayPurchasePrices, ...weekendPurchasePrices]).toFixed(2)),
    purchaseRate: purchaseCount / rows.length,
    categoryPrices: toAverages(categoryPrices),
    brandPrices: toAverages(brandPrices),
    categoryRates: toRates(categoryPurchase),
    brandRates: toRates(brandPurchase),
    hourRates: toRates(hourPurchase),
    weekdayRates: toRates(weekdayPurchase),
  },
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(data, null, 2)}\n`);
console.log(`Generated ${outputPath} from ${rows.length.toLocaleString()} events.`);
