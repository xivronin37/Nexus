# Nexus Engine: Raw Evaluation Mathematical Specification

This document provides the formalized mathematical model for the raw heuristic evaluation function used in **Nexus** (a 12x12 high-complexity abstract strategy game), prior to the neural network. The entire system is anchored to a **Player 1 ($P_1$) perspective** (positive scores favor $P_1$; negative scores favor $P_2$).

---

## 1. The Core Evaluation Function

The overall position evaluation $\text{Eval}(s)$ for a given board state $s$ is computed as the sum of a non-linearly scaled material balance and four distinct tactical/strategic signals:

$$\text{Eval}(s) = 0.3 \cdot \Big( \Delta_{\text{stones}} + \Delta_{\text{sticks}} + \mathcal{G}_{\text{res}} \Big) \cdot \mathcal{V}^{\mathcal{R}} + \mathcal{M} + \mathcal{S} + \mathcal{P} + \mathcal{F}$$

Where:
*   $\Delta_{\text{stones}} = s_1 - s_2$ — Active board stone differential.
*   $\Delta_{\text{sticks}} = S_1 - S_2$ — Active board stick placement differential.
*   $\mathcal{G}_{\text{res}}$ — Resource Banking Gap signal.
*   $\mathcal{V}^{\mathcal{R}}$ — Non-linear risk-adjusted volatility scaling factor.
*   $\mathcal{M}$ — Hyperbolic Mercy Rule boundary signal.
*   $\mathcal{S}$ — Inventory Scarcity penalty/reward signal.
*   $\mathcal{P}$ — Absolute scoreboard point differential.
*   $\mathcal{F}$ — Localized 3-stick flippable stone threat signal.

---

## 2. Component Specifications

### A. Core Adaptive Parameters

#### Risk Scaling Density ($\mathcal{R}$)
Compresses total active board piece density into a strictly bounded exponential window:
$$\mathcal{R} = \max\left(0.8, \min\left(1.2, \frac{s_1 + s_2 + S_1 + S_2}{200.0}\right)\right)$$

#### Volatility Factor ($\mathcal{V}$)
Measures the square-root density of 3-stick flippable states relative to total placed stones:
$$\mathcal{V} = \begin{cases} 0.6 + \sqrt{\frac{F_{\text{total}}}{s_1 + s_2}} & \text{if } (s_1 + s_2) > 0 \\ 0.8 & \text{otherwise} \end{cases}$$
*(Where $F_{\text{total}}$ is the total number of flippable stones across both players).*

#### Resource Banking Gap ($\mathcal{G}_{\text{res}}$)
Evaluates resource bank disparities relative to a dynamic system scarcity ceiling:
$$\delta_{\text{pts}} = \begin{cases} \min(s_1 + S_1, s_2 + S_2) & \text{if } \min(s_1 + S_1, s_2 + S_2) > 0 \\ 1000.0 & \text{otherwise} \end{cases}$$

$$\mathcal{G}_{\text{res}} = \text{sgn}\Big((s_1 + S_1) - (s_2 + S_2)\Big) \cdot \frac{25.0}{\delta_{\text{pts}}}$$

---

### B. Asymptotic Signals

#### Mercy Rule Boundary ($\mathcal{M}$)
Acts as a severe hyperbolic penalty/reward curve that explodes asymptotically toward $\pm\infty$ when either player approaches the 30-point doubling threshold while the game is active ($G = 0$):
$$\mathcal{M} = \begin{cases} \frac{50.0}{\frac{1}{2}P_1 - P_2 + 0.1} & \text{if } P_1 > 30 \land G = 0 \\ \frac{50.0}{P_1 - \frac{1}{2}P_2 - 0.1} & \text{if } P_2 > 30 \land G = 0 \\ 0 & \text{otherwise} \end{cases}$$
*(Where $P_1, P_2$ are active player points, and $G$ is the game-over terminal flag).*

#### Material Inventory Scarcity ($\mathcal{S}$)
Tracks remaining unplaced inventory reserves, pulling the evaluation score down aggressively if an individual supply chain runs empty:
$$\mathcal{S} = \frac{15.0}{(\text{stones}_2 + \text{sticks}_2) + 0.5} - \frac{15.0}{(\text{stones}_1 + \text{sticks}_1) + 0.5}$$

#### Scoreboard Point Differential ($\mathcal{P}$)
Pure line-item game score variance including active chain extensions:
$$\mathcal{P} = P_1 - P_2$$

#### Tactical Trap Potential ($\mathcal{F}$)
Calculates relative immediate exposure to 3-stick flippable stone configurations:
$$\mathcal{F} = 0.6 \cdot (F_{\text{opp}} - F_{\text{self}})$$