# PRODUCT.md — Tokenfold

## Register

product — a self-hosted usage/cost telemetry dashboard. Design serves the
data; the brand voice is strong but never at the expense of legibility.

## Users & Purpose

One power user (the operator) plus occasional viewers. Context: glancing at
cost/usage on desktop or phone, often mid-work. Primary jobs: "how much have
I spent (today / this month / per model / per machine)?", "is ingestion
healthy?", "does our measured cost match Anthropic's official billing?".
Numbers must be trustworthy, comparable, and scannable in seconds.

## Brand & Personality

Bauhaus / De Stijl: flat planes of primary color (red #e63329, blue #1a4b8c,
yellow #f5c518) on warm paper, heavy 2-3px #1a1a1a borders, Archivo Black
display + DM Sans body, uppercase micro-labels with wide tracking, tabular
numerals for all figures. Confident, geometric, honest. Established system —
preserve it; do not soften it into generic SaaS.

## Anti-references

Generic admin-template look (Bootstrap/AntD tables), soft shadows and
rounded glassy cards, purple-gradient SaaS, sparkline soup. No fabricated
numbers anywhere: unknown = em dash or explicit "unpriced/none" state.

## Accessibility & Constraints

Single self-contained server-rendered template (templates/dashboard.html),
vanilla JS + Chart.js, no build step. WCAG AA contrast (token --yellow-text
exists for yellow-on-paper), prefers-reduced-motion honored, mobile ≥390px
supported. All user-supplied strings go through esc(); all currency through
fC().
