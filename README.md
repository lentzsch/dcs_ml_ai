# DCS ML AI

An experimental project using reinforcement learning to train a human-like AI pilot for Digital Combat Simulator (DCS).

## Goals

- Develop a high-difficulty, human-like fighter pilot AI using modern reinforcement learning techniques.
- Implement scalable difficulty levels with customizable error modeling and "pilot personality" traits.
- Utilize TacView telemetry and supervised imitation learning to enhance realism and learning efficiency.
- Maintain modularity to allow iterative upgrades and fallback to vanilla DCS AI logic during partial deployment.

## Training Gameplan

### 🛫 Phase 1: Low-Fidelity Environment Bootstrapping
- Begin training in simplified environments (e.g., `CartPole-v1`) to validate model architecture and integration.
- Develop progressively more complex environments that simulate basic flight mechanics and dogfight decision trees (e.g., Unity-based or custom Gym environments).
- Eventually create an open-ended gym-compatible flight simulation capable of supporting DCS-like scenarios.

### 🧠 Phase 2: Curriculum Learning & Domain Transfer
- Train agents in a curriculum: starting with takeoff, then navigation, formation flying, and basic air combat maneuvers.
- Use domain transfer to scale from low-fidelity trainers (like a Cessna or T-6) to high-fidelity aircraft approximations.

### 🛰️ Phase 3: TacView Integration
- Use TacView data to fine-tune agents based on real human-vs-AI engagements.
- Partner with or host multiplayer servers to collect high-quality telemetry during PvE sessions.
- Develop tools to export agent telemetry in a TacView-compatible format for visualization and post-mortem analysis.

### 🎮 Phase 4: Incremental Integration with DCS
- Initially deploy AI to control a single aircraft type with fallback to DCS vanilla AI for unsupported roles or failures.
- Gradually expand coverage to more aircraft and mission profiles.
- Explore potential collaboration or modding access with Eagle Dynamics to enable deeper integration.

## Stack

- **Python 3.11**
- **Gymnasium** – Environment API and wrappers
- **Stable-Baselines3** – RL algorithm library (using PPO)
- **PyTorch** – Neural network backend
- **MoviePy** – Training visualization (turnkey video export)
- **TensorBoard** – Metrics logging and performance visualization
- **Poetry** – Dependency and environment management
- **TacView** (Planned) – Flight telemetry recording and visualization
- **Unity** or **Custom Gym Environments** (Planned) – For flight dynamics simulation

## Features

- Automatic video recording of training episodes
- Manual or automatic saving of training sessions
- Session tagging and archival with timestamped logs
- Optional video visualization toggle to preserve storage during long sessions
- Planned support for TacView-compatible telemetry export

## Status

🚧 In early development. Currently validating training pipeline and setting up visualization tools.

