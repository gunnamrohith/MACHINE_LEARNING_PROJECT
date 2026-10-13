"use client";

import { FormEvent, ReactNode, useState } from "react";
import { Bars, Donut, Point, TrendChart } from "./Charts";

type DashboardData = {
  sourceRows: number;
  dateRange: { start: string; end: string };
  kpis: Record<string, number>;
  sales: { byCategory: Point[]; daily: Point[]; monthly: Point[] };
  customers: { activity: Point[]; top: (Point & { orders: number })[] };
  products: { topViewed: Point[]; topPurchased: Point[]; categoryPopularity: Point[] };
  probability: {
    events: Point[];
    funnel: Point[];
    viewToCartRate: number;
    cartToPurchaseRate: number;
    overallConversionRate: number;
  };
  statistics: Record<string, number>;
  predictions: {
    categories: string[];
    brands: string[];
    overallPrice: number;
    purchaseRate: number;
    categoryPrices: Record<string, number>;
    brandPrices: Record<string, number>;
    categoryRates: Record<string, number>;
    brandRates: Record<string, number>;
    hourRates: Record<string, number>;
    weekdayRates: Record<string, number>;
  };
};

const views = [
  ["overview", "⌂", "Overview"],
  ["sales", "↗", "Sales Analysis"],
  ["customers", "♙", "Customers"],
  ["products", "◇", "Products"],
  ["probability", "◉", "Probability & Stats"],
  ["price", "⌁", "Price Prediction"],
  ["purchase", "◎", "Purchase Prediction"],
] as const;
type View = (typeof views)[number][0];

const rupees = (value: number, decimals = 0) =>
  `₹${value.toLocaleString("en-IN", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function Card({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return <article className="metric-card" style={{ "--accent": accent ?? "#7c6cf2" } as React.CSSProperties}>
    <span>{label}</span><strong>{value}</strong>
  </article>;
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return <section className="panel"><h2>{title}</h2>{children}</section>;
}

function Header({ eyebrow, title, copy }: { eyebrow: string; title: string; copy: string }) {
  return <header className="page-header"><span>{eyebrow}</span><h1>{title}</h1><p>{copy}</p></header>;
}

export default function Dashboard({ data }: { data: DashboardData }) {
  const [view, setView] = useState<View>("overview");
  const [menuOpen, setMenuOpen] = useState(false);
  const [category, setCategory] = useState(data.predictions.categories[0]);
  const [brand, setBrand] = useState(data.predictions.brands[0]);
  const [hour, setHour] = useState(12);
  const [weekday, setWeekday] = useState(0);
  const [priceResult, setPriceResult] = useState<number>();
  const [purchaseResult, setPurchaseResult] = useState<number>();
  const prediction = data.predictions;

  function navigate(id: View) {
    setView(id);
    setMenuOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function predictPrice(event: FormEvent) {
    event.preventDefault();
    const categoryPrice = prediction.categoryPrices[category] ?? prediction.overallPrice;
    const brandPrice = prediction.brandPrices[brand] ?? prediction.overallPrice;
    const timeAdjustment = 1 + (hour - 12) * 0.002 + (weekday >= 5 ? 0.01 : -0.01);
    setPriceResult((categoryPrice * 0.55 + brandPrice * 0.35 + prediction.overallPrice * 0.1) * timeAdjustment);
  }

  function predictPurchase(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const price = Number(new FormData(event.currentTarget).get("price"));
    const rates = [
      prediction.categoryRates[category], prediction.brandRates[brand],
      prediction.hourRates[String(hour)], prediction.weekdayRates[String(weekday)],
      prediction.purchaseRate,
    ].filter(Number.isFinite);
    const base = rates.reduce((sum, rate) => sum + rate, 0) / rates.length;
    const priceFactor = Math.max(0.65, Math.min(1.25, prediction.overallPrice / Math.max(price, 1)));
    setPurchaseResult(Math.max(0.01, Math.min(0.95, base * priceFactor)));
  }

  const selectInputs = <>
    <label>Product category<select value={category} onChange={(event) => setCategory(event.target.value)}>
      {prediction.categories.map((item) => <option key={item}>{item}</option>)}
    </select></label>
    <label>Brand<select value={brand} onChange={(event) => setBrand(event.target.value)}>
      {prediction.brands.map((item) => <option key={item}>{item}</option>)}
    </select></label>
  </>;

  return <div className="app-shell">
    <aside className={menuOpen ? "sidebar open" : "sidebar"}>
      <div className="brand"><div className="brand-mark">R</div><div><strong>Retail IQ</strong><span>Analytics suite</span></div></div>
      <nav aria-label="Dashboard sections">
        {views.map(([id, icon, label]) => <button key={id} className={view === id ? "active" : ""} onClick={() => navigate(id)}>
          <span>{icon}</span>{label}
        </button>)}
      </nav>
      <div className="data-note"><span>Live dataset</span><strong>{data.sourceRows.toLocaleString()} events</strong><small>{data.dateRange.start} — {data.dateRange.end}</small></div>
    </aside>

    <main>
      <div className="topbar">
        <button className="menu-button" onClick={() => setMenuOpen(!menuOpen)} aria-label="Toggle navigation">☰</button>
        <span>{views.find(([id]) => id === view)?.[2]}</span>
        <div className="status"><i /> Dataset ready</div>
      </div>
      <div className="content">
        {view === "overview" && <>
          <Header eyebrow="Executive overview" title="Retail performance, at a glance." copy="A focused view of revenue, customers, products, and buying behaviour across your commerce dataset." />
          <div className="metrics four">
            <Card label="Total revenue" value={rupees(data.kpis.revenue)} accent="#2dd4bf" />
            <Card label="Customers" value={data.kpis.customers.toLocaleString()} accent="#7c6cf2" />
            <Card label="Products" value={data.kpis.products.toLocaleString()} accent="#38bdf8" />
            <Card label="Purchases" value={data.kpis.purchases.toLocaleString()} accent="#fb7185" />
          </div>
          <div className="grid"><Panel title="Revenue momentum"><TrendChart data={data.sales.daily} /></Panel><Panel title="Revenue mix"><Donut data={data.sales.byCategory} /></Panel></div>
          <div className="insight">Revenue is concentrated in the leading categories. Use the category mix to prioritise inventory and campaign investment.</div>
        </>}

        {view === "sales" && <>
          <Header eyebrow="Commercial performance" title="Sales analysis" copy="Track revenue across categories and time to identify momentum, seasonality, and growth opportunities." />
          <div className="metrics three"><Card label="Total revenue" value={rupees(data.kpis.revenue)} /><Card label="Orders" value={data.kpis.purchases.toLocaleString()} /><Card label="Average order value" value={rupees(data.kpis.averageOrderValue, 2)} /></div>
          <div className="grid"><Panel title="Revenue by category"><Bars data={data.sales.byCategory} currency horizontal /></Panel><Panel title="Monthly revenue"><TrendChart data={data.sales.monthly} /></Panel></div>
          <Panel title="Daily revenue trend"><TrendChart data={data.sales.daily} /></Panel>
        </>}

        {view === "customers" && <>
          <Header eyebrow="Audience intelligence" title="Customer analysis" copy="Understand visitor activity, buyer quality, and the customers contributing the most revenue." />
          <div className="metrics three"><Card label="Unique visitors" value={data.kpis.customers.toLocaleString()} /><Card label="Unique buyers" value={data.kpis.buyers.toLocaleString()} /><Card label="Repeat buyer rate" value={`${data.kpis.repeatBuyerRate.toFixed(1)}%`} /></div>
          <div className="grid"><Panel title="Activity distribution"><Bars data={data.customers.activity} /></Panel><Panel title="Top customers by spend"><Bars data={data.customers.top} currency horizontal /></Panel></div>
        </>}

        {view === "products" && <>
          <Header eyebrow="Merchandising intelligence" title="Product analysis" copy="See which products attract attention, convert into purchases, and lead category engagement." />
          <div className="metrics three"><Card label="Unique products" value={data.kpis.products.toLocaleString()} /><Card label="Total views" value={Math.round(data.probability.events[0].value * data.sourceRows).toLocaleString()} /><Card label="Total purchases" value={data.kpis.purchases.toLocaleString()} /></div>
          <div className="grid"><Panel title="Most viewed products"><Bars data={data.products.topViewed} horizontal /></Panel><Panel title="Most purchased products"><Bars data={data.products.topPurchased} horizontal /></Panel></div>
          <Panel title="Category popularity"><Donut data={data.products.categoryPopularity} /></Panel>
        </>}

        {view === "probability" && <>
          <Header eyebrow="Decision science" title="Probability & statistics" copy="Translate event behaviour into conversion probabilities and statistically grounded pricing insights." />
          <div className="metrics three">{data.probability.events.map((item) => <Card key={item.name} label={`P(${item.name})`} value={percent(item.value)} />)}</div>
          <div className="grid"><Panel title="Event probabilities"><Bars data={data.probability.events.map((item) => ({ ...item, value: item.value * 100 }))} /></Panel><Panel title="Conversion funnel"><Bars data={data.probability.funnel} horizontal /></Panel></div>
          <div className="metrics three"><Card label="95% CI — lower" value={rupees(data.statistics.confidenceLower, 2)} /><Card label="Average price" value={rupees(data.statistics.priceMean, 2)} /><Card label="95% CI — upper" value={rupees(data.statistics.confidenceUpper, 2)} /></div>
          <div className="insight">View → cart: {percent(data.probability.viewToCartRate)} · Cart → purchase: {percent(data.probability.cartToPurchaseRate)} · Overall conversion: {percent(data.probability.overallConversionRate)}</div>
        </>}

        {view === "price" && <>
          <Header eyebrow="Predictive analytics" title="Estimate product price" copy="Use dataset-derived category and brand patterns to create an explainable price estimate." />
          <div className="prediction-layout">
            <form className="panel form" onSubmit={predictPrice}>{selectInputs}
              <label>Hour of day <b>{hour}:00</b><input type="range" min="0" max="23" value={hour} onChange={(event) => setHour(Number(event.target.value))} /></label>
              <label>Day of week<select value={weekday} onChange={(event) => setWeekday(Number(event.target.value))}>{["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].map((day, index) => <option key={day} value={index}>{day}</option>)}</select></label>
              <button type="submit" className="primary">Estimate price</button>
            </form>
            <div className="result-card"><span>Estimated product price</span><strong>{priceResult ? rupees(priceResult, 2) : "—"}</strong><p>Based on observed category and brand averages. Use this educational estimate as a starting point, not an exact valuation.</p></div>
          </div>
        </>}

        {view === "purchase" && <>
          <Header eyebrow="Purchase propensity" title="Will this interaction convert?" copy="Estimate purchase likelihood from observed category, brand, time, and price patterns." />
          <div className="metrics three"><Card label="Baseline purchase rate" value={percent(prediction.purchaseRate)} /><Card label="Training interactions" value={Math.round(data.sourceRows * .8).toLocaleString()} /><Card label="Test interactions" value={Math.round(data.sourceRows * .2).toLocaleString()} /></div>
          <div className="prediction-layout">
            <form className="panel form" onSubmit={predictPurchase}>{selectInputs}
              <label>Product price<input name="price" type="number" min="1" step="0.01" defaultValue={prediction.overallPrice} required /></label>
              <label>Hour of day <b>{hour}:00</b><input type="range" min="0" max="23" value={hour} onChange={(event) => setHour(Number(event.target.value))} /></label>
              <button type="submit" className="primary">Predict purchase</button>
            </form>
            <div className="result-card"><span>Purchase likelihood</span><strong>{purchaseResult == null ? "—" : percent(purchaseResult)}</strong>
              <em className={purchaseResult != null && purchaseResult >= prediction.purchaseRate ? "positive" : ""}>{purchaseResult == null ? "Enter interaction details" : purchaseResult >= prediction.purchaseRate ? "Above-baseline intent" : "Below-baseline intent"}</em>
              <p>This transparent propensity score uses aggregate historical rates and does not identify individual customers.</p>
            </div>
          </div>
        </>}
      </div>
    </main>
  </div>;
}
