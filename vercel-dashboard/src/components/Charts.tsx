"use client";

import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

export type Point = { name: string; value: number };
const COLORS = ["#7c6cf2", "#2dd4bf", "#fb7185", "#fbbf24", "#38bdf8", "#a78bfa"];
const money = (value: number) => `₹${value.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;

export function TrendChart({ data, currency = true }: { data: Point[]; currency?: boolean }) {
  return <div className="chart"><ResponsiveContainer width="100%" height="100%">
    <LineChart data={data}>
      <CartesianGrid stroke="#28314a" strokeDasharray="3 3" />
      <XAxis dataKey="name" stroke="#8993aa" minTickGap={28} />
      <YAxis stroke="#8993aa" tickFormatter={currency ? money : undefined} width={72} />
      <Tooltip formatter={(value) => currency ? money(Number(value)) : Number(value).toLocaleString()} />
      <Line type="monotone" dataKey="value" stroke="#7c6cf2" strokeWidth={3} dot={false} />
    </LineChart>
  </ResponsiveContainer></div>;
}

export function Bars({ data, currency = false, horizontal = false }: {
  data: Point[]; currency?: boolean; horizontal?: boolean;
}) {
  return <div className="chart"><ResponsiveContainer width="100%" height="100%">
    <BarChart data={data} layout={horizontal ? "vertical" : "horizontal"} margin={{ left: horizontal ? 24 : 0 }}>
      <CartesianGrid stroke="#28314a" strokeDasharray="3 3" />
      <XAxis type={horizontal ? "number" : "category"} dataKey={horizontal ? undefined : "name"} stroke="#8993aa" tickFormatter={horizontal && currency ? money : undefined} />
      <YAxis type={horizontal ? "category" : "number"} dataKey={horizontal ? "name" : undefined} stroke="#8993aa" width={horizontal ? 95 : 64} tickFormatter={!horizontal && currency ? money : undefined} />
      <Tooltip formatter={(value) => currency ? money(Number(value)) : Number(value).toLocaleString()} />
      <Bar dataKey="value" fill="#7c6cf2" radius={[5, 5, 0, 0]} />
    </BarChart>
  </ResponsiveContainer></div>;
}

export function Donut({ data }: { data: Point[] }) {
  return <div className="chart"><ResponsiveContainer width="100%" height="100%">
    <PieChart>
      <Pie data={data} dataKey="value" nameKey="name" innerRadius={62} outerRadius={105} paddingAngle={2}>
        {data.map((item, index) => <Cell key={item.name} fill={COLORS[index % COLORS.length]} />)}
      </Pie>
      <Tooltip formatter={(value) => Number(value).toLocaleString()} />
    </PieChart>
  </ResponsiveContainer></div>;
}
