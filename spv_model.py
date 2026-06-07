"""
spv_model.py
------------
SPV (Special Purpose Vehicle) Project Finance Model for solar assets.

Built by: Arjun D
Real-world context: At Ciel Et Terre Solar (India arm of France HQ floating
solar company), I structured SPV-based capital and funding strategies for
large-scale floating solar assets, optimising subsidy deployment and
project bankability for lenders and investors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Project Parameters ───────────────────────────────────────────────────────
DEFAULT_PROJECT = {
    "name": "Floating Solar SPV — 10 MW",
    "capex_inr": 400_000_000,          # ₹40 Cr total project cost
    "debt_equity_ratio": 70 / 30,      # 70:30 D:E (typical solar project)
    "debt_interest_rate": 0.085,        # 8.5% p.a. (post-refinancing target)
    "debt_tenor_years": 15,
    "project_life_years": 25,
    "annual_generation_kwh": 15_000_000, # 15 million kWh/year
    "tariff_per_kwh": 3.50,             # ₹3.50/kWh PPA tariff
    "tariff_escalation": 0.00,          # Zero escalation (fixed tariff)
    "opex_per_kwh": 0.50,               # ₹0.50/kWh O&M
    "opex_escalation": 0.04,            # 4% annual O&M escalation
    "degradation_rate": 0.005,          # 0.5% annual panel degradation
    "tax_rate": 0.25,                   # 25% corporate tax
    "discount_rate": 0.12,              # 12% WACC / equity discount rate
}


def build_cashflow_model(params: dict) -> pd.DataFrame:
    """
    Build year-by-year project cash flows including:
    - Revenue (with degradation + tariff escalation)
    - OPEX (with escalation)
    - EBITDA
    - Debt service (interest + principal)
    - DSCR (Debt Service Coverage Ratio)
    - Free Cash Flow to Equity
    """
    capex = params["capex_inr"]
    debt_pct = params["debt_equity_ratio"] / (1 + params["debt_equity_ratio"])
    equity_pct = 1 - debt_pct

    total_debt = capex * debt_pct
    total_equity = capex * equity_pct
    interest_rate = params["debt_interest_rate"]
    tenor = params["debt_tenor_years"]
    life = params["project_life_years"]

    # Straight-line principal repayment over debt tenor
    annual_principal = total_debt / tenor

    rows = []
    remaining_debt = total_debt

    for year in range(1, life + 1):
        # Revenue
        gen = params["annual_generation_kwh"] * (1 - params["degradation_rate"]) ** year
        tariff = params["tariff_per_kwh"] * (1 + params["tariff_escalation"]) ** year
        revenue = gen * tariff

        # OPEX
        opex = gen * params["opex_per_kwh"] * (1 + params["opex_escalation"]) ** year

        ebitda = revenue - opex

        # Debt service
        interest = remaining_debt * interest_rate if year <= tenor else 0
        principal = annual_principal if year <= tenor else 0
        debt_service = interest + principal

        # DSCR = EBITDA / Debt Service (lender's key metric; must be >1.25x)
        dscr = ebitda / debt_service if debt_service > 0 else np.inf

        # Tax
        ebit = ebitda - interest
        tax = max(ebit * params["tax_rate"], 0)

        # Free Cash Flow to Equity
        fcfe = ebitda - debt_service - tax

        remaining_debt = max(remaining_debt - principal, 0)

        rows.append({
            "Year": year,
            "Generation (kWh M)": round(gen / 1e6, 2),
            "Revenue (₹ Cr)": round(revenue / 1e7, 2),
            "OPEX (₹ Cr)": round(opex / 1e7, 2),
            "EBITDA (₹ Cr)": round(ebitda / 1e7, 2),
            "Debt Service (₹ Cr)": round(debt_service / 1e7, 2),
            "DSCR": round(dscr, 2),
            "Tax (₹ Cr)": round(tax / 1e7, 2),
            "FCFE (₹ Cr)": round(fcfe / 1e7, 2),
            "Remaining Debt (₹ Cr)": round(remaining_debt / 1e7, 2),
        })

    df = pd.DataFrame(rows)
    return df, total_equity


def calculate_irr_npv(df: pd.DataFrame, total_equity: float, discount_rate: float):
    """Compute equity IRR and NPV from FCFE stream."""
    # Initial equity outflow at Year 0
    cashflows = [-total_equity] + list(df["FCFE (₹ Cr)"] * 1e7)

    # IRR via numpy
    irr = np.irr(cashflows) if hasattr(np, 'irr') else _manual_irr(cashflows)

    # NPV
    npv = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cashflows))

    return round(irr * 100, 2), round(npv / 1e7, 2)


def _manual_irr(cashflows: list, guess: float = 0.1) -> float:
    """Newton-Raphson IRR solver (fallback if numpy.irr unavailable)."""
    r = guess
    for _ in range(1000):
        npv = sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))
        dnpv = sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cashflows))
        if abs(dnpv) < 1e-10:
            break
        r -= npv / dnpv
        if r <= -1:
            r = guess
    return r


def plot_project(df: pd.DataFrame, project_name: str, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{project_name} — Financial Model", fontsize=13, fontweight="bold")

    # Revenue vs OPEX
    axes[0, 0].bar(df["Year"], df["Revenue (₹ Cr)"], label="Revenue", color="#27AE60", alpha=0.8)
    axes[0, 0].bar(df["Year"], df["OPEX (₹ Cr)"], label="OPEX", color="#E74C3C", alpha=0.8)
    axes[0, 0].set_title("Revenue vs OPEX (₹ Cr)")
    axes[0, 0].legend()

    # EBITDA
    axes[0, 1].plot(df["Year"], df["EBITDA (₹ Cr)"], marker="o", color="#3498DB")
    axes[0, 1].set_title("EBITDA Trend (₹ Cr)")
    axes[0, 1].set_ylabel("₹ Cr")

    # DSCR — critical lender metric
    dscr_active = df[df["DSCR"] != np.inf]
    axes[1, 0].bar(dscr_active["Year"], dscr_active["DSCR"], color="#9B59B6", alpha=0.8)
    axes[1, 0].axhline(1.25, color="red", linestyle="--", linewidth=1.5, label="Min DSCR (1.25x)")
    axes[1, 0].axhline(1.0, color="darkred", linestyle="-", linewidth=1, label="Default Threshold (1.0x)")
    axes[1, 0].set_title("DSCR — Debt Service Coverage Ratio")
    axes[1, 0].legend(fontsize=8)

    # FCFE
    colors_fcfe = ["#27AE60" if v >= 0 else "#E74C3C" for v in df["FCFE (₹ Cr)"]]
    axes[1, 1].bar(df["Year"], df["FCFE (₹ Cr)"], color=colors_fcfe, alpha=0.85)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Free Cash Flow to Equity (₹ Cr)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    params = DEFAULT_PROJECT
    print(f"=== {params['name']} ===")
    print(f"CAPEX: ₹{params['capex_inr']/1e7:.0f} Cr | "
          f"Debt: {params['debt_equity_ratio']/(1+params['debt_equity_ratio'])*100:.0f}% | "
          f"Equity: {100/(1+params['debt_equity_ratio']):.0f}%\n")

    df, equity = build_cashflow_model(params)
    irr, npv = calculate_irr_npv(df, equity, params["discount_rate"])

    print(df.to_string(index=False))
    print(f"\n{'='*50}")
    print(f"Equity IRR:        {irr:.1f}%")
    print(f"Equity NPV:        ₹{npv:.2f} Cr")
    print(f"Min DSCR:          {df[df['DSCR'] != float('inf')]['DSCR'].min():.2f}x")
    print(f"Equity Invested:   ₹{equity/1e7:.2f} Cr")
    print(f"Total FCFE (life): ₹{df['FCFE (₹ Cr)'].sum():.2f} Cr")

    plot_project(df, params["name"], save_path="spv_cashflows.png")
    print("\nChart saved: spv_cashflows.png")
