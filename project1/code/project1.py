# =============================================================================
# STA 2503 Project 1: Discrete-Time vs. Continuous-Time Financial Models
# Alec Neme
#
# This script reproduces every figure and number in the report.
# Run it from the project1/ folder (python code/project1.py), or open it in
# VS Code and use "Run All" on the # %% cells.
# The lines "# BEGIN ..." / "# END ..." mark the parts shown in the appendix.
# =============================================================================

# %% Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

# Work from the project1/ folder whether this is run as a script or cell by cell
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "code" else Path.cwd()
FIG_DIR = ROOT / "report" / "figures"
DATA_DIR = ROOT / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# %% ===========================================================================
# PART I: Brownian motion and scaled random walks
# ==============================================================================

# BEGIN rw
def simulate_bm(N, n_paths, rng):
    dt = 1 / N
    t = np.linspace(0, 1, N + 1)
    dW = rng.normal(0, np.sqrt(dt), size=(n_paths, N))   # W_{t+dt} - W_t ~ N(0, dt)
    W = np.cumsum(dW, axis=1)                             # running sum along each path
    W = np.hstack([np.zeros((n_paths, 1)), W])            # W_0 = 0
    return t, W


def scaled_random_walk(N, n_paths, rng):
    t = np.linspace(0, 1, N + 1)
    Z = rng.choice([-1, 1], size=(n_paths, N))            # fair +-1 coin flips
    X = np.cumsum(Z, axis=1)                              # X_n = Z_1 + ... + Z_n
    X = np.hstack([np.zeros((n_paths, 1)), X])            # X_0 = 0
    W = (1 / np.sqrt(N)) * X                              # scale by 1/sqrt(N)
    return t, W


rng = np.random.default_rng(2503)                         # one generator for Part I
N1 = 10_000
N2_list = [10, 50, 200, 1000]
t_bm, W_bm = simulate_bm(N1, n_paths=5, rng=rng)
t10, W10 = scaled_random_walk(10, n_paths=5, rng=rng)
# END rw

# %% Part I figures (Figures 1 and 2)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_bm, W_bm.T, linewidth=0.8)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("$t$", fontsize=13); ax.set_ylabel("$W_t$", fontsize=13)
ax.set_title(f"Brownian motion, 5 sample paths ($N_1 = {N1:,}$ steps)", fontsize=13)
fig.savefig(FIG_DIR / "part1_bm_paths.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t10, W10.T, marker="o", markersize=3)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("$t$", fontsize=13); ax.set_ylabel("$W^{(N)}_t$", fontsize=13)
ax.set_title("Scaled symmetric random walk, 5 sample paths ($N_2 = 10$)", fontsize=13)
fig.savefig(FIG_DIR / "part1_rw_N10.pdf", bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
for ax, N2 in zip(axes.flat, N2_list):
    t_N, W_N = scaled_random_walk(N2, n_paths=5, rng=rng)
    ax.plot(t_bm, W_bm.T, color="lightgrey", linewidth=0.6)   # BM reference
    ax.plot(t_N, W_N.T, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"$N_2 = {N2}$", fontsize=13)
for ax in axes[1, :]: ax.set_xlabel("$t$", fontsize=13)
for ax in axes[:, 0]: ax.set_ylabel("$W^{(N_2)}_t$", fontsize=13)
fig.suptitle("Scaled random walks (colour) vs Brownian motion (grey)", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "part1_rw_vs_bm.pdf", bbox_inches="tight")
plt.show()


# %% ===========================================================================
# PART II: Market data and parameters
# ==============================================================================

# BEGIN data
REFRESH = False      # True re-downloads the data; False reuses the saved CSV files
CLOSE_FILE = DATA_DIR / "nvda_close.csv"
HIST_FILE = DATA_DIR / "nvda_history.csv"

if REFRESH or not CLOSE_FILE.exists() or not HIST_FILE.exists():
    import yfinance as yf
    last = yf.Ticker("NVDA").history(period="5d", auto_adjust=False)
    pd.DataFrame({"date": [last.index[-1].date()],
                  "close": [round(float(last["Close"].iloc[-1]), 2)],
                  "source": ["Yahoo Finance (yfinance)"]}).to_csv(CLOSE_FILE, index=False)
    hist = yf.Ticker("NVDA").history(start="2025-09-30", end="2026-10-01",
                                     auto_adjust=True)            # end date is exclusive
    hist[["Close"]].to_csv(HIST_FILE)

S0 = float(pd.read_csv(CLOSE_FILE)["close"].iloc[0])             # 228.38 (30 Sep 2026)
closes = pd.read_csv(HIST_FILE, index_col=0)["Close"].to_numpy()  # 252 adjusted closes


def estimate_volatility(closes, periods_per_year=252):
    log_returns = np.diff(np.log(closes))
    return np.sqrt(periods_per_year) * log_returns.std(ddof=1)


K = S0                               # at-the-money
T = 1.0                              # years
r = 0.04                             # risk-free rate, continuously compounded
sigma = estimate_volatility(closes)  # 0.3760
print(f"S0 = {S0},  sigma_hat = {sigma:.4f},  SE approx {sigma / np.sqrt(2 * (len(closes) - 1)):.4f}")
# END data


# %% ===========================================================================
# PART II: CRR binomial model
# ==============================================================================

# BEGIN crr
def crr_parameters(sigma, r, T, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    growth = np.exp(r * dt)
    if not (d < growth < u):
        raise ValueError("No-arbitrage condition d < e^{r dt} < u fails")
    p = (growth - d) / (u - d)
    return dt, u, d, p


def prices_at_step(S0, u, d, n):
    j = np.arange(n + 1)
    return S0 * u**j * d**(n - j)


def call_payoff(S, K):
    return np.maximum(S - K, 0)


def backward_induction(V_T, p, disc_fctr):
    V = V_T.copy()
    q = 1 - p
    for i in range(len(V_T) - 1):
        V = disc_fctr * (p * V[1:] + q * V[:-1])
    return V[0]


def binomial_call(S0, K, T, r, sigma, N):
    dt, u, d, p = crr_parameters(sigma, r, T, N)
    V_N = call_payoff(prices_at_step(S0, u, d, N), K)
    return backward_induction(V_N, p, np.exp(-r * dt))


def simulate_tree_paths(S0, sigma, r, T, N, n_paths, rng):
    dt, u, d, p = crr_parameters(sigma, r, T, N)
    t = np.linspace(0, T, N + 1)
    up = rng.random(size=(n_paths, N)) < p
    factors = np.where(up, u, d)
    S = S0 * np.cumprod(factors, axis=1)
    S = np.hstack([np.full((n_paths, 1), S0), S])
    return t, S
# END crr


# %% ===========================================================================
# PART II: Black-Scholes benchmark
# ==============================================================================

# BEGIN bs
def black_scholes_call(S0, K, T, r, sigma):
    d_plus = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d_minus = d_plus - sigma * np.sqrt(T)
    return S0 * norm.cdf(d_plus) - K * np.exp(-r * T) * norm.cdf(d_minus)
# END bs


# %% ===========================================================================
# PART II: GBM (Euler-Maruyama) and Monte Carlo
# ==============================================================================

# BEGIN gbm
def simulate_gbm_euler(S0, mu, sigma, T, N, n_paths, rng):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    Z = rng.standard_normal(size=(n_paths, N))
    factors = 1 + mu * dt + sigma * np.sqrt(dt) * Z
    S = S0 * np.cumprod(factors, axis=1)
    S = np.hstack([np.full((n_paths, 1), S0), S])
    return t, S


def mc_call_price(S0, K, T, r, sigma, N, M, rng):
    t, S = simulate_gbm_euler(S0, r, sigma, T, N, M, rng)
    disc_payoffs = np.exp(-r * T) * call_payoff(S[:, -1], K)
    price = disc_payoffs.mean()
    se = disc_payoffs.std(ddof=1) / np.sqrt(M)
    return price, se
# END gbm


# %% ===========================================================================
# PART II: Verification (tests against known answers)
# ==============================================================================

# BEGIN tests
# Lecture 2 three-period example (S0 = 4, u = 2, d = 1/2, K = 6, r = 25%): V0 = 2.048
V_T = call_payoff(prices_at_step(4, 2, 0.5, 3), 6)
print("Lecture check:", backward_induction(V_T, p=0.5, disc_fctr=1 / 1.25))

# Martingale check under Q: E[S_T] should equal S0 e^{rT} = 237.70
_, S_tree = simulate_tree_paths(S0, sigma, r, T, 100, 100_000, np.random.default_rng(0))
_, S_gbm = simulate_gbm_euler(S0, r, sigma, T, 100, 100_000, np.random.default_rng(0))
print("E[S_T]: tree", S_tree[:, -1].mean(), " Euler", S_gbm[:, -1].mean(),
      " theory", S0 * np.exp(r * T))
# END tests


# %% ===========================================================================
# PART II: Binomial results (Figure 3)
# ==============================================================================

# BEGIN conv
C_bs = black_scholes_call(S0, K, T, r, sigma)
for N in [10, 50, 100, 101, 500, 1000]:
    price = binomial_call(S0, K, T, r, sigma, N)
    print(f"N = {N:5d}   price = {price:.4f}   error vs BS = {price - C_bs:+.4f}")

N_values = np.arange(1, 301)
prices = [binomial_call(S0, K, T, r, sigma, N) for N in N_values]
# END conv

# CRR tree with N = 4 (Figure 3a)
N_small = 4
dt_s, u_s, d_s, p_s = crr_parameters(sigma, r, T, N_small)
fig, ax = plt.subplots(figsize=(8, 5))
for n in range(N_small + 1):
    t_n = n * dt_s
    S_n = prices_at_step(S0, u_s, d_s, n)
    if n < N_small:
        S_next = prices_at_step(S0, u_s, d_s, n + 1)
        for j in range(n + 1):
            ax.plot([t_n, t_n + dt_s], [S_n[j], S_next[j + 1]], color="grey", linewidth=0.8)
            ax.plot([t_n, t_n + dt_s], [S_n[j], S_next[j]], color="grey", linewidth=0.8)
    ax.scatter(np.full(n + 1, t_n), S_n, color="C0", zorder=3)
    for j in range(n + 1):
        ax.annotate(f"{S_n[j]:.2f}", (t_n, S_n[j]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=9)
ax.set_xlabel("$t$ (years)", fontsize=12)
ax.set_ylabel("$S_t$ (\\$)", fontsize=12)
ax.set_title(f"CRR tree for NVDA, $N = {N_small}$ "
             f"($u = {u_s:.4f}$, $d = {d_s:.4f}$, $\\tilde p = {p_s:.4f}$)", fontsize=12)
fig.savefig(FIG_DIR / "part2_crr_tree.pdf", bbox_inches="tight")
plt.show()

# Price against N (Figure 3b)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(N_values, prices, linewidth=0.8, label="CRR binomial price")
ax.axhline(C_bs, color="black", linestyle="--", label=f"Black\u2013Scholes = {C_bs:.4f}")
ax.set_xlabel("Number of steps $N$", fontsize=12)
ax.set_ylabel("Call price (\\$)", fontsize=12)
ax.set_title("CRR binomial price vs. number of steps", fontsize=13)
ax.legend()
fig.savefig(FIG_DIR / "part2_binomial_convergence.pdf", bbox_inches="tight")
plt.show()


# %% ===========================================================================
# PART II: Sample paths under both models (Figure 4)
# ==============================================================================

# BEGIN paths
rng = np.random.default_rng(2503)          # one generator for Part II
N_plot = 100
t_tree, S_tree = simulate_tree_paths(S0, sigma, r, T, N_plot, n_paths=10, rng=rng)
t_gbm, S_gbm = simulate_gbm_euler(S0, r, sigma, T, N_plot, n_paths=10, rng=rng)
# END paths

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_tree, S_tree.T, linewidth=0.9)
ax.axhline(K, color="black", linestyle="--", linewidth=0.8, label=f"strike $K = {K:.2f}$")
ax.set_xlabel("$t$ (years)", fontsize=12)
ax.set_ylabel("$S_t$ (\\$)", fontsize=12)
ax.set_title(f"Simulated NVDA price paths on a CRR tree ($N = {N_plot}$, 10 paths)", fontsize=12)
ax.legend(loc="upper left")
fig.savefig(FIG_DIR / "part2_tree_paths.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_gbm, S_gbm.T, linewidth=0.9)
ax.set_xlabel("$t$ (years)", fontsize=12)
ax.set_ylabel("$S_t$ (\\$)", fontsize=12)
ax.set_title(f"Simulated NVDA price paths under GBM (Euler\u2013Maruyama, $N = {N_plot}$, 10 paths)",
             fontsize=12)
fig.savefig(FIG_DIR / "part2_gbm_paths.pdf", bbox_inches="tight")
plt.show()


# %% ===========================================================================
# PART II: Comparison of the three prices (Table 2) and checks
# ==============================================================================

# BEGIN compare
N_steps, M_mc = 100, 100_000
C_tree = binomial_call(S0, K, T, r, sigma, N_steps)
C_mc, se_mc = mc_call_price(S0, K, T, r, sigma, N_steps, M_mc, rng)
print(f"Black-Scholes         : {C_bs:.4f}")
print(f"CRR binomial (N={N_steps}) : {C_tree:.4f}   diff vs BS = {C_tree - C_bs:+.4f}")
print(f"Monte Carlo (M={M_mc:,}): {C_mc:.4f} +- {se_mc:.4f}   "
      f"95% CI [{C_mc - 1.96 * se_mc:.4f}, {C_mc + 1.96 * se_mc:.4f}]   "
      f"diff vs BS = {C_mc - C_bs:+.4f} ({(C_mc - C_bs) / se_mc:.1f} SE)")

# Coverage of the 95% CI over 100 independent seeds
hits = 0
for seed in range(100):
    price, se = mc_call_price(S0, K, T, r, sigma, N_steps, M_mc, np.random.default_rng(seed))
    hits += (price - 1.96 * se <= C_bs <= price + 1.96 * se)
print(f"{hits}/100 confidence intervals contain the Black-Scholes price")

# Sensitivity of the Black-Scholes price to sigma (+- one standard error of sigma_hat)
print("BS price at sigma -+ 0.017:", black_scholes_call(S0, K, T, r, sigma - 0.017),
      black_scholes_call(S0, K, T, r, sigma + 0.017))
# END compare
