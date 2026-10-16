# Nexus
A high-complexity abstract strategy game played on a 12 by 12 grid.

## Rules:

### Core Mechanics and Components
**Pieces:** Stones (value: 2) and Sticks (value: 1)  
**Initial Pool:** Each player starts with 5 stones and 4 sticks.  
**Resource Genration:** Connecting 2 stones allows a choice of 1 additional stone **OR** 1 extra stick.
> [NOTE] Forming a chain grants **BOTH.**

### The Chain Rule and Scoring
**Definition:** A contiguous group of 4+ stones connected by sticks.  
**Scoring:** Creating a chain grants a bonus point for every additional stone that extends it.

### The Adjacency Rule and Volatility
A stone's state shifts based on the number of connected sticks (can be done by opponent as well):  
**0 Sticks:** Dormant (Worth 0 points, inactive).  
**1-2 Sticks:** Alive / Filled (Active contributor to chains and scores).  
**3 Sticks:** Volatile / Flippable (Opponent can choose to flip this stone and its directly connected sticks to their possession on any future turn).  
**4 Sticks:** Dead (The stone and its direct connections are permanently removed from the board).  

### Flip Window
You cannot make a flip move that has already been done within the last 6 ply.

### Win Conditions
1. **Mercy Rule:** A player wins instantly if they have more than 30 points and their opponent's total points fall below half of their own.
2. **Surrender:** A player wins if their opponent has no legal moves to make.
3. **Resignation:** A player wins if their opponent resigns the game.

## Command Line Interface (CLI):
The Nexus executable strictly operates via the **Nexus Communcation Protocol (NCP), designed for easy integration for engines.** To read more about it, click [here][ncp-link].

## Arcturus:
Arcturus is an engine specifically built to play Nexus. 

**v1.0 (Legacy):** Relies on a rapid, superficial hand-crafted heuristic evaluation formula. Learn more about the raw mathematics in the [Evaluation Formula Specs][eval-link].

**v2.0 (Current):** Transitioning into a deep reinforcement learning phase. Driven by an asynchronous neural network policy/value evaluation framework heavily inspired by AlphaGo Zero's self-play training loop.
Track active generation milestones, passing thresholds, loss optimization matrices, and SPRT match logs in the [Neural Network Training Log][nnlog-link].


[eval-link]: NexusEval.md
[ncp-link]: ncp.md
[nnlog-link]: /src/nnlog.md

