import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(page_title="Markovian Queueing Systems", layout="wide")


# ── Constants ────────────────────────────────────────────────────────────────
WATERMARK_TEXT = "by M.Sc. Dilan Mogollón"

COLORS = [
    "#3266ad", "#5DCAA5", "#AFA9EC", "#D85A30",
    "#EF9F27", "#E24B4A", "#7F77DD", "#1D9E75"
]


# ── Styles ───────────────────────────────────────────────────────────────────
watermark_html = f"""
<style>
.watermark {{
    position: fixed;
    top: 150px;
    right: 25px;
    opacity: 0.95;
    font-size: 22px;
    font-weight: 900;
    color: #ff4b4b;
    text-shadow: 1px 1px 2px #000;
    z-index: 9999;
    pointer-events: none;
}}

.info-box {{
    background-color: rgba(49, 51, 63, 0.08);
    padding: 16px 20px;
    border-radius: 14px;
    border: 1px solid rgba(120, 120, 120, 0.25);
    margin-bottom: 18px;
}}

.metric-card {{
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    min-height: 142px;
}}

.metric-card h4 {{
    color: #e2e8f0;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 8px;
}}

.metric-card .metric-value {{
    color: #38bdf8;
    font-size: 28px;
    font-weight: 900;
    font-family: monospace;
}}

.metric-card .metric-label {{
    color: #94a3b8;
    font-size: 13px;
    margin-top: 5px;
    line-height: 1.35;
}}

.rate-card {{
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151;
    border-radius: 18px;
    padding: 20px;
    min-height: 150px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.30);
    margin-bottom: 12px;
}}

.rate-card h3 {{
    color: #ffffff;
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 12px;
}}

.rate-card p {{
    color: #d1d5db;
    font-size: 15px;
    line-height: 1.5;
}}
</style>

<div class="watermark">{WATERMARK_TEXT}</div>
"""

st.markdown(watermark_html, unsafe_allow_html=True)


# ── Queueing helper functions ────────────────────────────────────────────────
def total_service_rate(n: int, servers: int, mu: float) -> float:
    """Total departure rate μ_n from state n for an M/M/s queue."""
    if n <= 0:
        return 0.0
    return min(n, servers) * mu


def birth_rate(n: int, lam: float, finite: bool, K: int | None) -> float:
    """Arrival rate λ_n from state n."""
    if finite and K is not None and n >= K:
        return 0.0
    return lam


def finite_stationary_distribution(lam: float, mu: float, servers: int, K: int) -> np.ndarray:
    """Exact stationary distribution for M/M/s/K using birth-death recursion."""
    weights = np.ones(K + 1, dtype=float)
    for n in range(1, K + 1):
        lam_prev = birth_rate(n - 1, lam, True, K)
        mu_n = total_service_rate(n, servers, mu)
        if mu_n <= 0:
            raise ValueError("The service rate must be positive.")
        weights[n] = weights[n - 1] * lam_prev / mu_n

    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Could not normalize the stationary distribution.")
    return weights / total


def infinite_stationary_probabilities(
    lam: float,
    mu: float,
    servers: int,
    max_state: int,
) -> tuple[np.ndarray, float, float]:
    """Exact probabilities p_0,...,p_max_state and remaining tail for M/M/s."""
    a = lam / mu
    rho = lam / (servers * mu)

    if rho >= 1.0:
        raise ValueError("The infinite-capacity system is unstable because λ ≥ sμ.")

    if servers == 1:
        p0 = 1.0 - rho
    else:
        first = sum((a ** n) / math.factorial(n) for n in range(servers))
        tail_norm = (a ** servers) / (math.factorial(servers) * (1.0 - rho))
        p0 = 1.0 / (first + tail_norm)

    probs = []
    for n in range(max_state + 1):
        if n < servers:
            pn = p0 * (a ** n) / math.factorial(n)
        else:
            pn = p0 * (a ** n) / (math.factorial(servers) * (servers ** (n - servers)))
        probs.append(pn)

    probs = np.array(probs, dtype=float)
    tail = max(0.0, 1.0 - probs.sum())
    return probs, tail, p0


def infinite_metrics(lam: float, mu: float, servers: int) -> dict:
    """Exact steady-state metrics for stable M/M/1 and M/M/s systems."""
    a = lam / mu
    rho = lam / (servers * mu)
    if rho >= 1.0:
        return {
            "stable": False,
            "rho": rho,
        }

    if servers == 1:
        p0 = 1.0 - rho
        p_wait = rho
        Lq = rho ** 2 / (1.0 - rho)
    else:
        first = sum((a ** n) / math.factorial(n) for n in range(servers))
        tail_norm = (a ** servers) / (math.factorial(servers) * (1.0 - rho))
        p0 = 1.0 / (first + tail_norm)
        p_wait = tail_norm * p0  # Erlang C
        Lq = p_wait * rho / (1.0 - rho)

    L = Lq + a
    Wq = Lq / lam if lam > 0 else 0.0
    W = Wq + 1.0 / mu

    return {
        "stable": True,
        "rho": rho,
        "p0": p0,
        "p_wait": p_wait,
        "blocking_probability": 0.0,
        "lambda_effective": lam,
        "Lq": Lq,
        "L": L,
        "Wq": Wq,
        "W": W,
        "mean_busy_servers": a,
        "server_utilization": rho,
    }


def finite_metrics(lam: float, mu: float, servers: int, K: int, p: np.ndarray) -> dict:
    """Exact steady-state metrics for M/M/s/K systems."""
    states = np.arange(K + 1)
    p_block = float(p[K])
    lambda_eff = lam * (1.0 - p_block)

    L = float(np.sum(states * p))
    Lq = float(np.sum(np.maximum(states - servers, 0) * p))
    busy = float(np.sum(np.minimum(states, servers) * p))
    utilization = busy / servers

    W = L / lambda_eff if lambda_eff > 1e-15 else np.nan
    Wq = Lq / lambda_eff if lambda_eff > 1e-15 else np.nan

    if K < servers:
        p_all_busy = 0.0
        p_wait_admitted = 0.0
    else:
        p_all_busy = float(np.sum(p[servers:]))
        admitted_prob = 1.0 - p_block
        wait_mass = float(np.sum(p[servers:K])) if K > servers else 0.0
        p_wait_admitted = wait_mass / admitted_prob if admitted_prob > 1e-15 else np.nan

    return {
        "stable": True,
        "rho": lam / (servers * mu),
        "p0": float(p[0]),
        "p_wait": p_wait_admitted,
        "p_all_busy": p_all_busy,
        "blocking_probability": p_block,
        "lambda_effective": lambda_eff,
        "Lq": Lq,
        "L": L,
        "Wq": Wq,
        "W": W,
        "mean_busy_servers": busy,
        "server_utilization": utilization,
    }


def build_generator_matrix(
    lam: float,
    mu: float,
    servers: int,
    max_state: int,
    finite: bool,
    K: int | None,
) -> np.ndarray:
    """
    Build Q for a finite system, or the leading principal block of Q for an
    infinite system. For the infinite block, the last row retains the true
    diagonal -(λ + μ_n), while q_{N,N+1}=λ lies outside the displayed block.
    """
    n_states = max_state + 1
    Q = np.zeros((n_states, n_states), dtype=float)

    for n in range(n_states):
        lam_n = birth_rate(n, lam, finite, K)
        mu_n = total_service_rate(n, servers, mu)

        if n + 1 < n_states and lam_n > 0:
            Q[n, n + 1] = lam_n
        if n - 1 >= 0 and mu_n > 0:
            Q[n, n - 1] = mu_n

        # For an infinite system at the last displayed state, λ is included in
        # the diagonal even though the transition to N+1 is outside the block.
        Q[n, n] = -(lam_n + mu_n)

    return Q


def build_rate_table(
    lam: float,
    mu: float,
    servers: int,
    max_state: int,
    finite: bool,
    K: int | None,
) -> pd.DataFrame:
    rows = []
    for n in range(max_state + 1):
        lam_n = birth_rate(n, lam, finite, K)
        mu_n = total_service_rate(n, servers, mu)
        active = min(n, servers)
        waiting = max(n - servers, 0)
        rows.append({
            "State n": n,
            "Customers in service": active,
            "Customers waiting": waiting,
            "Birth rate λₙ": lam_n,
            "Death rate μₙ": mu_n,
        })
    return pd.DataFrame(rows)


def build_birth_death_figure(
    lam: float,
    mu: float,
    servers: int,
    max_state: int,
    finite: bool,
    K: int | None,
) -> go.Figure:
    fig = go.Figure()
    xs = np.arange(max_state + 1, dtype=float)
    ys = np.zeros(max_state + 1)

    # Nodes
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=[str(i) for i in range(max_state + 1)],
        textposition="middle center",
        marker=dict(
            size=44,
            color=[COLORS[i % len(COLORS)] for i in range(max_state + 1)],
            line=dict(width=2, color="white"),
        ),
        hovertemplate="State %{text}<extra></extra>",
        showlegend=False,
    ))

    # Birth and death arrows between adjacent states
    for n in range(max_state):
        lam_n = birth_rate(n, lam, finite, K)
        mu_next = total_service_rate(n + 1, servers, mu)

        if lam_n > 0:
            fig.add_annotation(
                x=n + 0.88,
                y=0.11,
                ax=n + 0.12,
                ay=0.11,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=2,
                text="",
            )
            fig.add_annotation(
                x=n + 0.5,
                y=0.23,
                text=f"λ = {lam_n:g}",
                showarrow=False,
                font=dict(size=12),
            )

        if mu_next > 0:
            fig.add_annotation(
                x=n + 0.12,
                y=-0.11,
                ax=n + 0.88,
                ay=-0.11,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=2,
                text="",
            )
            fig.add_annotation(
                x=n + 0.5,
                y=-0.23,
                text=f"μ{n+1} = {mu_next:g}",
                showarrow=False,
                font=dict(size=12),
            )

    # Infinite tail
    if not finite:
        fig.add_annotation(
            x=max_state + 0.72,
            y=0,
            text="⋯",
            showarrow=False,
            font=dict(size=34),
        )
        fig.add_annotation(
            x=max_state + 0.95,
            y=0.14,
            ax=max_state + 0.25,
            ay=0.14,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            text=f"λ = {lam:g}",
            font=dict(size=11),
        )

    fig.update_layout(
        title="Birth-death transition-rate diagram",
        height=390,
        margin=dict(l=20, r=30, t=65, b=20),
        xaxis=dict(visible=False, range=[-0.6, max_state + (1.25 if not finite else 0.6)]),
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        showlegend=False,
    )
    return fig


def build_probability_figure(probabilities: np.ndarray, tail: float, finite: bool) -> go.Figure:
    states = [str(i) for i in range(len(probabilities))]
    values = list(probabilities)

    if not finite and tail > 1e-12:
        states.append(f"> {len(probabilities) - 1}")
        values.append(tail)

    fig = go.Figure(go.Bar(
        x=states,
        y=values,
        text=[f"{x:.5f}" for x in values],
        textposition="outside",
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(values))],
    ))
    fig.update_layout(
        title="Steady-state probability distribution",
        xaxis_title="Number of customers in the system n",
        yaxis_title="Probability pₙ",
        yaxis=dict(range=[0, max(0.05, max(values) * 1.22)]),
        height=470,
        margin=dict(b=70),
    )
    return fig


def build_flow_balance_table(
    probabilities: np.ndarray,
    lam: float,
    mu: float,
    servers: int,
    finite: bool,
    K: int | None,
) -> pd.DataFrame:
    rows = []
    max_state = len(probabilities) - 1
    for n in range(max_state):
        lam_n = birth_rate(n, lam, finite, K)
        mu_next = total_service_rate(n + 1, servers, mu)
        left = probabilities[n] * lam_n
        right = probabilities[n + 1] * mu_next
        rows.append({
            "Transition": f"{n} ↔ {n+1}",
            "pₙ λₙ": left,
            "pₙ₊₁ μₙ₊₁": right,
            "Absolute difference": abs(left - right),
        })
    return pd.DataFrame(rows)


def fmt(value, decimals=6):
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def metric_card(title: str, value: str, label: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Queue configuration")

model = st.sidebar.radio(
    "Queueing model",
    ["M/M/1", "M/M/s"],
    index=0,
)

capacity_type = st.sidebar.radio(
    "Queue capacity",
    ["Infinite", "Finite"],
    index=0,
)

lam = st.sidebar.number_input(
    "Arrival rate λ",
    min_value=0.000001,
    value=2.0,
    step=0.1,
    format="%.6f",
)

mu = st.sidebar.number_input(
    "Service rate μ per server",
    min_value=0.000001,
    value=3.0,
    step=0.1,
    format="%.6f",
)

if model == "M/M/s":
    servers = int(st.sidebar.number_input(
        "Number of servers s",
        min_value=2,
        max_value=100,
        value=3,
        step=1,
    ))
else:
    servers = 1

finite = capacity_type == "Finite"

if finite:
    waiting_capacity = int(st.sidebar.number_input(
        "Finite waiting-room capacity",
        min_value=0,
        max_value=500,
        value=5,
        step=1,
        help="Number of customers that can wait, excluding those currently in service.",
    ))
    K = servers + waiting_capacity
    display_max_state = K
    st.sidebar.caption(f"Total system capacity: K = s + waiting places = {K}")
else:
    K = None
    extra_states = int(st.sidebar.number_input(
        "States to display after all servers are active",
        min_value=2,
        max_value=100,
        value=5,
        step=1,
        help="This only controls the displayed birth-death diagram and Q block. It does not truncate the infinite queue mathematically.",
    ))
    display_max_state = servers + extra_states
    st.sidebar.caption(
        f"Service capacity becomes constant from state n = {servers}. "
        f"The display will show states 0 through {display_max_state}."
    )

st.sidebar.markdown("---")
solve = st.sidebar.button("Analyze queueing system", use_container_width=True, type="primary")


# ── Main header ──────────────────────────────────────────────────────────────
st.title("Analysis of Markovian Queueing Systems")
st.markdown(
    """
    <div class="info-box">
        This application models <b>M/M/1</b> and <b>M/M/s</b> queues as continuous-time
        birth-death Markov chains. The transition structure is generated from the arrival
        rate λ, the service rate μ per server, the number of servers s, and the queue capacity.
        No transition matrix needs to be uploaded.
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Configuration signature ─────────────────────────────────────────────────
current_signature = (
    model,
    capacity_type,
    round(float(lam), 12),
    round(float(mu), 12),
    int(servers),
    int(K) if K is not None else None,
    int(display_max_state),
)


# ── Solve ────────────────────────────────────────────────────────────────────
if solve:
    try:
        if lam <= 0 or mu <= 0:
            raise ValueError("λ and μ must be positive.")

        if finite:
            probabilities = finite_stationary_distribution(lam, mu, servers, K)
            tail = 0.0
            metrics = finite_metrics(lam, mu, servers, K, probabilities)
        else:
            metrics = infinite_metrics(lam, mu, servers)
            if not metrics.get("stable", False):
                st.session_state.pop("queue_solution", None)
                st.error(
                    "The infinite-capacity system does not have a stationary distribution because "
                    f"λ = {lam:g} ≥ sμ = {servers * mu:g}. Reduce λ, increase μ, or increase s."
                )
                st.stop()

            probabilities, tail, _ = infinite_stationary_probabilities(
                lam, mu, servers, display_max_state
            )

        Q = build_generator_matrix(
            lam=lam,
            mu=mu,
            servers=servers,
            max_state=display_max_state,
            finite=finite,
            K=K,
        )

        rate_df = build_rate_table(
            lam=lam,
            mu=mu,
            servers=servers,
            max_state=display_max_state,
            finite=finite,
            K=K,
        )

        balance_df = build_flow_balance_table(
            probabilities=probabilities,
            lam=lam,
            mu=mu,
            servers=servers,
            finite=finite,
            K=K,
        )

        st.session_state["queue_solution"] = {
            "signature": current_signature,
            "probabilities": probabilities,
            "tail": tail,
            "metrics": metrics,
            "Q": Q,
            "rate_df": rate_df,
            "balance_df": balance_df,
        }
        st.success("Queueing system analyzed successfully.")

    except Exception as exc:
        st.session_state.pop("queue_solution", None)
        st.error(f"The queueing system could not be analyzed: {exc}")


# ── Recover solution ─────────────────────────────────────────────────────────
solution = st.session_state.get("queue_solution")
solution_is_valid = solution is not None and solution.get("signature") == current_signature

if solution_is_valid:
    probabilities = solution["probabilities"]
    tail = solution["tail"]
    metrics = solution["metrics"]
    Q = solution["Q"]
    rate_df = solution["rate_df"]
    balance_df = solution["balance_df"]
else:
    probabilities = tail = metrics = Q = rate_df = balance_df = None


def require_solution_message():
    if solution is None:
        st.info("Configure the queue and click **Analyze queueing system** in the sidebar.")
    else:
        st.warning("The configuration changed. Click **Analyze queueing system** again.")


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_bd, tab_q, tab_ss, tab_metrics, tab_probs, tab_balance = st.tabs([
    "Birth-death process",
    "Generator matrix Q",
    "Steady state",
    "Queue metrics",
    "State probabilities",
    "Flow balance",
])


# ── TAB 1: Birth-death process ───────────────────────────────────────────────
with tab_bd:
    st.markdown("## Birth-death structure")

    if model == "M/M/1":
        st.latex(r"\lambda_n=\lambda, \qquad \mu_n=\begin{cases}0,&n=0\\ \mu,&n\ge 1\end{cases}")
        st.info(
            "For M/M/1, service capacity is already saturated from state n = 1: "
            "once the server is busy, the total service rate remains μ."
        )
    else:
        st.latex(r"\lambda_n=\lambda, \qquad \mu_n=\min(n,s)\mu")
        st.info(
            f"For M/M/s with s = {servers}, the total service rate increases as nμ for n < s. "
            f"From state n = {servers} onward, all servers are active and the total service rate remains sμ = {servers * mu:g}."
        )

    if finite:
        st.latex(r"\lambda_K=0")
        st.caption(
            f"This is an M/M/{servers}/{K} finite-capacity system. State K = {K} is full, so new arrivals are blocked."
        )
    else:
        st.caption(
            "The state space is infinite. Only a finite number of states is displayed; the analytical model itself remains infinite."
        )

    if not solution_is_valid:
        require_solution_message()
    else:
        fig_bd = build_birth_death_figure(
            lam, mu, servers, display_max_state, finite, K
        )
        st.plotly_chart(fig_bd, use_container_width=True)

        st.markdown("### State-dependent transition rates")
        st.dataframe(rate_df, use_container_width=True, hide_index=True)


# ── TAB 2: Generator matrix Q ────────────────────────────────────────────────
with tab_q:
    st.markdown("## Infinitesimal generator matrix Q")
    st.latex(
        r"q_{n,n+1}=\lambda_n,\qquad q_{n,n-1}=\mu_n,\qquad "
        r"q_{n,n}=-(\lambda_n+\mu_n)"
    )

    if not solution_is_valid:
        require_solution_message()
    else:
        state_labels = [str(i) for i in range(display_max_state + 1)]
        Q_df = pd.DataFrame(np.round(Q, 6), index=state_labels, columns=state_labels)
        st.dataframe(Q_df, use_container_width=True, height=520)

        if finite:
            row_sums = Q.sum(axis=1)
            max_error = float(np.max(np.abs(row_sums)))
            st.caption(
                f"Finite system: every row of Q sums to 0. Maximum numerical row-sum error = {max_error:.3e}."
            )
        else:
            st.warning(
                f"This is the leading block of the infinite generator, showing states 0,…,{display_max_state}. "
                f"The transition {display_max_state} → {display_max_state + 1} with rate λ = {lam:g} is outside the displayed block. "
                "Therefore the last displayed row is not expected to sum to zero inside this finite block."
            )

        st.markdown("### Matrix pattern")
        st.latex(
            r"Q=\begin{bmatrix}"
            r"-\lambda_0 & \lambda_0 & 0 & 0 & \cdots\\"
            r"\mu_1 & -(\lambda_1+\mu_1) & \lambda_1 & 0 & \cdots\\"
            r"0 & \mu_2 & -(\lambda_2+\mu_2) & \lambda_2 & \cdots\\"
            r"\vdots & \ddots & \ddots & \ddots & \ddots"
            r"\end{bmatrix}"
        )


# ── TAB 3: Steady state ─────────────────────────────────────────────────────
with tab_ss:
    st.markdown("## Steady-state analysis")
    st.latex(r"\boldsymbol{\pi}Q=0,\qquad \sum_n \pi_n=1")
    st.latex(r"\pi_n\lambda_n=\pi_{n+1}\mu_{n+1}")

    if not solution_is_valid:
        require_solution_message()
    else:
        if finite:
            st.success(
                "A finite-capacity birth-death queue always has a stationary distribution when all modeled rates are finite and positive."
            )
        else:
            rho = metrics["rho"]
            st.success(
                f"Stability condition satisfied: ρ = λ/(sμ) = {rho:.6f} < 1."
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("P₀", fmt(metrics["p0"]), "Probability that the system is empty.")
        with c2:
            metric_card(
                "Traffic intensity ρ",
                fmt(metrics["rho"]),
                "Offered load relative to nominal service capacity sμ.",
            )
        with c3:
            metric_card(
                "Mean busy servers",
                fmt(metrics["mean_busy_servers"]),
                "Expected number of servers simultaneously busy.",
            )

        st.markdown("### Stationary probabilities shown")
        p_df = pd.DataFrame({
            "State n": np.arange(len(probabilities)),
            "pₙ": probabilities,
        })
        st.dataframe(p_df, use_container_width=True, hide_index=True)

        if not finite:
            st.caption(
                f"Probability mass beyond state {display_max_state}: {tail:.8f}. "
                "This tail is included in the exact infinite model even though those states are not listed individually."
            )


# ── TAB 4: Queue metrics ─────────────────────────────────────────────────────
with tab_metrics:
    st.markdown("## Queue performance metrics")

    if not solution_is_valid:
        require_solution_message()
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("L", fmt(metrics["L"]), "Expected number of customers in the system.")
        with c2:
            metric_card("Lq", fmt(metrics["Lq"]), "Expected number of customers waiting in queue.")
        with c3:
            metric_card("W", fmt(metrics["W"]), "Expected time a customer spends in the system.")
        with c4:
            metric_card("Wq", fmt(metrics["Wq"]), "Expected waiting time before service begins.")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            metric_card(
                "Effective arrival rate",
                fmt(metrics["lambda_effective"]),
                "Throughput λ_eff. For finite systems, blocked arrivals are excluded.",
            )
        with c6:
            metric_card(
                "Server utilization",
                fmt(metrics["server_utilization"]),
                "Average fraction of service capacity that is busy.",
            )
        with c7:
            metric_card(
                "Blocking probability",
                fmt(metrics["blocking_probability"]),
                "Probability that an arrival finds the finite system full.",
            )
        with c8:
            wait_label = (
                "Probability that an admitted arrival must wait."
                if finite
                else "Erlang-C delay probability: probability an arrival must wait."
            )
            metric_card("Probability of waiting", fmt(metrics["p_wait"]), wait_label)

        st.markdown("### Little's law checks")
        check_rows = [
            {
                "Relationship": "L = λ_eff W",
                "Left side": metrics["L"],
                "Right side": metrics["lambda_effective"] * metrics["W"],
            },
            {
                "Relationship": "Lq = λ_eff Wq",
                "Left side": metrics["Lq"],
                "Right side": metrics["lambda_effective"] * metrics["Wq"],
            },
        ]
        check_df = pd.DataFrame(check_rows)
        check_df["Absolute difference"] = np.abs(check_df["Left side"] - check_df["Right side"])
        st.dataframe(check_df, use_container_width=True, hide_index=True)

        if finite:
            st.info(
                "For finite capacity, λ_eff = λ(1 − p_K). This is the rate that must be used in Little's law, "
                "because arrivals finding the system full never enter it."
            )


# ── TAB 5: State probabilities ───────────────────────────────────────────────
with tab_probs:
    st.markdown("## Probability distribution by system population")

    if not solution_is_valid:
        require_solution_message()
    else:
        fig_p = build_probability_figure(probabilities, tail, finite)
        st.plotly_chart(fig_p, use_container_width=True)

        prob_rows = []
        for n, pn in enumerate(probabilities):
            prob_rows.append({
                "State n": n,
                "Customers in service": min(n, servers),
                "Customers waiting": max(n - servers, 0),
                "pₙ": pn,
            })
        prob_df = pd.DataFrame(prob_rows)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

        if not finite and tail > 1e-12:
            st.info(
                f"The displayed bars omit individual states above {display_max_state}; their exact combined probability is {tail:.8f}."
            )


# ── TAB 6: Flow balance ──────────────────────────────────────────────────────
with tab_balance:
    st.markdown("## Birth-death flow-balance verification")
    st.latex(r"\pi_n\lambda_n=\pi_{n+1}\mu_{n+1}")
    st.info(
        "In steady state, the probability flow from n to n+1 equals the reverse flow from n+1 to n for every adjacent pair of states."
    )

    if not solution_is_valid:
        require_solution_message()
    else:
        st.dataframe(balance_df, use_container_width=True, hide_index=True)
        max_balance_error = float(balance_df["Absolute difference"].max()) if not balance_df.empty else 0.0
        st.caption(f"Maximum local balance error in the displayed states: {max_balance_error:.3e}.")
