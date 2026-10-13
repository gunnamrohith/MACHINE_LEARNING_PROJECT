import Dashboard from "@/components/Dashboard";
import data from "@/data/dashboard.json";

export default function Home() {
  return <Dashboard data={data} />;
}
