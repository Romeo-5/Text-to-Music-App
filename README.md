# 🎵 Text-to-Music Generation: AI Music Synthesis

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange?style=flat-square)
![Transformers](https://img.shields.io/badge/Transformers-GPT--2-green?style=flat-square)
![MIDI](https://img.shields.io/badge/MIDI-Processing-purple?style=flat-square)

An AI-powered system that generates original music from text descriptions using deep learning. Input a text prompt like "A calm and peaceful melody" and the system composes a corresponding MIDI sequence that captures the sentiment and theme of your input.

## 🎯 Project Overview

This project explores the intersection of Natural Language Processing and music generation by:
- Processing text descriptions using a pre-trained GPT-2 language model
- Mapping textual sentiment and structure to musical note sequences
- Generating MIDI files that represent the mood and theme of input text
- Demonstrating multi-modal AI (text → music) capabilities

## ✨ Key Features

- **Text-to-MIDI generation** using GPT-2 and custom mapping algorithms
- **Sentiment-aware composition** - musical output reflects text mood and theme
- **Customizable parameters** - control tempo, note range, and sequence length
- **MIDI output** - compatible with any DAW or MIDI player
- **Experimental framework** - easily modify text inputs to generate diverse musical outputs

## 🛠️ Tech Stack

**AI/ML:**
- PyTorch - Deep learning framework
- Transformers (Hugging Face) - GPT-2 language model
- Natural Language Processing

**Music Processing:**
- `mido` - MIDI file manipulation
- `pretty_midi` - Advanced MIDI processing and analysis


# Minecraft Hierarchical Reinforcement Learning Agent
## Project Hub Document

---

| **Course** | USC CSCI 566 - Deep Learning and its Applications |
|------------|--------------------------------------------------|
| **Team Size** | 5-7 members |
| **Repository** | github.com/Romeo-5/minecraft-hrl-agent |
| **Status** | ✅ Boilerplate Complete - Ready for Development |

---

# 1. Project Overview

We are building a research-grade Minecraft agent that uses **Hierarchical Reinforcement Learning (HRL)** to navigate the game's technology tree. Unlike traditional RL approaches that operate on low-level motor controls (move forward, turn left, attack), our agent reasons at the level of **skills and goals**.

## 1.1 The Problem with Low-Level RL

Training an agent to craft a diamond pickaxe from raw motor commands is extremely sample-inefficient. The agent must discover, through random exploration, the exact sequence of thousands of low-level actions needed. State-of-the-art research (Voyager, Plan4MC, DEPS) has shown that hierarchical approaches dramatically improve learning efficiency.

## 1.2 Our Approach: Options Framework

We implement the **Options Framework** (Sutton, Precup, Singh 1999), treating complex behaviors as temporally-extended actions called "options" or "skills." Each skill has:

1. An **initiation set** (preconditions)
2. An **internal policy** (execution logic)
3. A **termination condition**

The high-level agent learns *which skill to invoke*, not how to execute low-level controls.

## 1.3 Research Contribution: Action Novelty Heuristics

Our primary research contribution focuses on how the agent selects which skill to execute. We propose **Action Novelty Heuristics** that combine:

- **UCB-style exploration bonuses** — try less-visited skills
- **Tech tree awareness** — prefer skills that unlock more options
- **Success rate tracking** — prefer reliable skills

This creates a curriculum that naturally progresses through the Minecraft tech tree.

---

# 2. System Architecture

The system consists of three main components connected via WebSocket:

| Component | Technology | Responsibility |
|-----------|------------|----------------|
| **Minecraft Server** | Paper 1.20.1 | Game world simulation, physics, game rules |
| **Mineflayer Bot** | Node.js + Mineflayer | Skill execution, world interaction, state observation |
| **RL Backend** | Python + SB3 + Gymnasium | High-level planning, learning, novelty tracking |

### Data Flow

```
Python Agent                    Mineflayer Bot                 Minecraft Server
     │                               │                               │
     │──── Skill ID (action) ───────▶│                               │
     │                               │──── Execute skill ───────────▶│
     │                               │◀─── World state ──────────────│
     │◀─── (state, reward, done) ────│                               │
     │                               │                               │
```

## 2.1 Current Skill Library (13 Skills)

| ID | Skill Name | Description | Preconditions |
|----|------------|-------------|---------------|
| 0 | `idle` | No operation (baseline action) | None |
| 1 | `harvest_wood` | Find and chop nearest tree | None |
| 2 | `mine_stone` | Mine cobblestone | Wooden pickaxe |
| 3 | `craft_planks` | Craft wooden planks from logs | Has logs |
| 4 | `craft_sticks` | Craft sticks from planks | Has planks |
| 5 | `craft_crafting_table` | Craft a crafting table | 4+ planks |
| 6 | `craft_wooden_pickaxe` | Craft wooden pickaxe at crafting table | Sticks + planks + table |
| 7 | `craft_stone_pickaxe` | Craft stone pickaxe | Cobblestone + sticks |
| 8 | `eat_food` | Consume food to restore hunger | Has food + hungry |
| 9 | `explore` | Move to random nearby location | None |
| 10 | `place_crafting_table` | Place crafting table in world | Has crafting table |
| 11 | `mine_iron` | Mine iron ore | Stone pickaxe |
| 12 | `smelt_iron` | Smelt raw iron in furnace | Raw iron + furnace (placeholder) |

---

# 3. Research Foundation

Our approach builds on three influential papers in the Minecraft AI space:

## 3.1 Voyager (Wang et al., 2023)

- **Key Idea:** Code-as-action paradigm where skills are JavaScript functions
- **What we adopt:** Skill library structure, Mineflayer as execution engine
- **What we extend:** Replace LLM planning with learned RL policy + novelty heuristics

## 3.2 Plan4MC (Yuan et al., 2023)

- **Key Idea:** Decompose tasks into basic skills, use graph search for planning
- **What we adopt:** Skill dependency graph encoding tech tree prerequisites
- **What we extend:** Dynamic unlock potential scoring for curriculum learning

## 3.3 DEPS (Wang et al., 2023)

- **Key Idea:** Describe-Explain-Plan-Select loop for interactive planning
- **What we adopt:** Error feedback mechanism (`skill_success`, `skill_message` in info dict)
- **What we extend:** Automatic success rate tracking for skill reliability estimation

---

# 4. Work Distribution

The project naturally divides into several independent workstreams. Below are suggested roles for a team of 5-7 members. Each role has clear deliverables and interfaces with other components.

## Team Roles

| Role | Members | Responsibilities | Deliverables |
|------|---------|------------------|--------------|
| **Skill Engineer** | 1-2 | Implement new skills in Mineflayer, improve existing skill reliability, handle edge cases | 15+ working skills, combat skills, building skills, smelting implementation |
| **RL Researcher** | 1-2 | Design and tune novelty heuristics, experiment with reward shaping, ablation studies | Novelty formulation, hyperparameter configs, learning curves, paper section |
| **Environment Engineer** | 1 | Improve observation space, implement hard reset, add parallel environments | RCON reset, vectorized env, improved state encoding |
| **Evaluation Lead** | 1 | Design evaluation metrics, run experiments, create visualizations, benchmark baselines | Eval suite, TensorBoard dashboards, baseline comparisons |
| **Integration Lead** | 1 | Manage codebase, CI/CD, documentation, coordinate interfaces between components | Clean repo, Docker setup, API docs, testing infrastructure |

## 4.1 Suggested Sprint Plan

| Sprint | Timeline | Goals |
|--------|----------|-------|
| **Sprint 1** | Week 1-2 | Environment setup for all members, run first training, identify gaps in current implementation |
| **Sprint 2** | Week 3-4 | Expand skill library to 20+ skills, implement hard reset, first baseline experiments |
| **Sprint 3** | Week 5-6 | Novelty heuristic ablations, reward shaping experiments, evaluation framework |
| **Sprint 4** | Week 7-8 | Final experiments, paper writing, video demo, presentation preparation |

---

# 5. Development Tasks (Backlog)

The following tasks are organized by priority. Team members should claim tasks and track progress in GitHub Issues.

## 5.1 🔴 High Priority (Core Functionality)

| Task | Owner | Status |
|------|-------|--------|
| Implement RCON hard reset (kill/tp bot between episodes) | TBD | ⬜ Not Started |
| Complete `smelt_iron` skill with furnace interaction | TBD | ⬜ Not Started |
| Add `craft_furnace` skill | TBD | ⬜ Not Started |
| Improve pathfinding error handling and retries | TBD | ⬜ Not Started |
| Add prismarine-viewer integration for debugging | TBD | ⬜ Not Started |
| Fix Health/Food showing as undefined on spawn | TBD | ⬜ Not Started |

## 5.2 🟡 Medium Priority (Research Extensions)

| Task | Owner | Status |
|------|-------|--------|
| Ablation: UCB constant tuning (c parameter) | TBD | ⬜ Not Started |
| Ablation: Novelty weight in hybrid mode | TBD | ⬜ Not Started |
| Implement count-based state novelty bonus | TBD | ⬜ Not Started |
| Compare PPO vs DQN vs SAC performance | TBD | ⬜ Not Started |
| Implement curiosity-driven exploration baseline (ICM) | TBD | ⬜ Not Started |
| Add Random Network Distillation (RND) baseline | TBD | ⬜ Not Started |

## 5.3 🟢 Lower Priority (Nice to Have)

| Task | Owner | Status |
|------|-------|--------|
| Parallel environments with SubprocVecEnv | TBD | ⬜ Not Started |
| Docker Compose for full stack deployment | TBD | ⬜ Not Started |
| Combat skills (attack mob, flee from danger) | TBD | ⬜ Not Started |
| Building skills (place block, build shelter) | TBD | ⬜ Not Started |
| MCP integration (replace WebSocket with mcpmc) | TBD | ⬜ Not Started |
| Implement farming/food gathering skills | TBD | ⬜ Not Started |

---

# 6. Getting Started Guide

Follow these steps to set up your development environment:

## 6.1 Prerequisites

- ☐ Minecraft Java Edition (for visual debugging)
- ☐ Node.js 18+ (for Mineflayer bot)
- ☐ Python 3.10+ (for RL backend)
- ☐ Java 17+ (for Minecraft server)
- ☐ Git (for version control)

## 6.2 Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Romeo-5/minecraft-hrl-agent.git
   cd minecraft-hrl-agent
   ```

2. **Install Node dependencies:**
   ```bash
   cd mineflayer
   npm install
   ```

3. **Install Python dependencies:**
   ```bash
   cd ../python
   pip install -r requirements.txt
   ```

4. **Download Paper 1.20.1 server:**
   - Go to https://papermc.io/downloads/paper
   - Download version 1.20.1
   - Place in a separate `minecraft-server` folder

5. **Configure server:**
   - Run once: `java -Xmx2G -jar paper.jar`
   - Accept EULA: change `eula=false` to `eula=true` in `eula.txt`
   - Edit `server.properties`: set `online-mode=false`

## 6.3 Running the System

Open three terminal windows:

**Terminal 1 — Minecraft Server:**
```bash
cd minecraft-server
java -Xmx2G -jar paper.jar
```

**Terminal 2 — Mineflayer Bot:**
```bash
cd minecraft-hrl-agent/mineflayer
npm start
```

**Terminal 3 — Training:**
```bash
cd minecraft-hrl-agent/python
python main.py --mode hybrid --timesteps 10000
```

## 6.4 Verifying Setup

✅ Server shows: `HRL_Agent joined the game`  
✅ Bot shows: `Ready for Python agent connection!`  
✅ Training shows: `Connected! Action space: 13 skills`

**Optional:** Join the server yourself via Minecraft → Multiplayer → Direct Connect → `localhost` to watch the bot in real-time.

---

# 7. Key Metrics & Evaluation

## 7.1 Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Tech Tree Depth** | Highest tier item crafted | Stone pickaxe → Iron pickaxe |
| **Sample Efficiency** | Steps to reach milestone | Lower is better |
| **Success Rate** | % of episodes reaching goal | > 50% |
| **Skill Utilization** | Entropy of skill selection | Higher = more exploration |

## 7.2 Ablation Studies

1. **Pure RL vs Hybrid:** Does novelty bonus improve learning?
2. **UCB constant (c):** Optimal exploration-exploitation tradeoff
3. **Tech tree weighting:** Does unlock potential help curriculum?
4. **State representation:** Which features matter most?

---

# 8. Resources and References

## 8.1 Key Papers

- **Voyager:** An Open-Ended Embodied Agent with LLMs (Wang et al., 2023)
- **Plan4MC:** Skill RL and Planning for Open-World Minecraft (Yuan et al., 2023)
- **DEPS:** Describe, Explain, Plan and Select (Wang et al., 2023)
- **Options Framework:** Between MDPs and Semi-MDPs (Sutton et al., 1999)

## 8.2 Documentation Links

- **Mineflayer:** https://github.com/PrismarineJS/mineflayer
- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io
- **Gymnasium:** https://gymnasium.farama.org
- **Paper Server:** https://papermc.io/documentation

## 8.3 Team Communication

- **GitHub Issues** — For task tracking and bug reports
- **GitHub Discussions** — For design decisions and questions
- **Pull Requests** — For code review before merging

---

# 9. FAQ

**Q: Can I use a different Minecraft version?**  
A: The bot is configured for 1.20.1. Other versions may work but aren't tested.

**Q: How do I add a new skill?**  
A: Add it to `mineflayer/src/skillManager.js` and update the dependency graph in `python/agent/planner.py`.

**Q: Why WebSocket instead of MCP?**  
A: WebSocket is simpler and sufficient for our needs. MCP integration is a stretch goal.

**Q: Can I train on GPU?**  
A: Yes! Use `python main.py --device cuda` if you have CUDA installed.

---

*Document last updated: January 2025*

**Python Libraries:**
- NumPy - Numerical operations
- Standard library modules
  

