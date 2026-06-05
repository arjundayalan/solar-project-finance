# Solar Project Finance Model (SPV)

Author: Arjun D | Finance Head | IFRS | Renewable Energy | M&A
LinkedIn: https://linkedin.com/in/arjundayalan

## What This Does

A Python-based SPV (Special Purpose Vehicle) project finance model for utility-scale solar assets.
Models the full 25-year financial life of a solar project including:

- Revenue with panel degradation and tariff escalation
- OPEX with annual inflation
- EBITDA and operating margins
- Debt service (interest + principal) over loan tenor
- DSCR: Debt Service Coverage Ratio (the critical lender bankability metric)
- Free Cash Flow to Equity (FCFE)
- Equity IRR and NPV

## Quick Start

pip install numpy pandas matplotlib
python spv_model.py

## Real-World Context

At Ciel Et Terre Solar (India arm of France-HQ global floating solar company) I structured
SPV-based capital and funding strategies for large-scale floating solar assets. Every project
required a detailed cash flow model to demonstrate bankability to lenders (minimum DSCR 1.25x),
justify equity returns to global investors, and model subsidy deployment.

## Key Concepts

DSCR (Debt Service Coverage Ratio) = EBITDA / Annual Debt Service
- Over 1.25x: Standard lender requirement for solar project finance
- Below 1.0x: Project cannot service debt - default risk
- Trending down: Red flag - generation degradation outpacing revenue

## Tech Stack
Python 3.10+ | numpy | pandas | matplotlib
