# Retail Sales Dashboard for Vercel

This is the Vercel-compatible rewrite of the Streamlit dashboard. The original Python application remains in `../retail-sales-dashboard`, while this application generates static analytics from the same committed sample CSV.

## Local development

```bash
npm install
npm run data:generate
npm run dev
```

Open `http://localhost:3000`.

## Deploy on Vercel

1. Import `gunnamrohith/MACHINE_LEARNING_PROJECT` in Vercel.
2. Set **Root Directory** to `vercel-dashboard`.
3. Keep **Framework Preset** as `Next.js`.
4. Keep the default install and build commands.
5. Click **Deploy**.

No environment variables, Python runtime, or external services are required.

## Validation

```bash
npm test
npm run lint
npm run build
```
