# DCS ML AI

An experimental project using reinforcement learning to train a human-like AI pilot for Digital Combat Simulator (DCS).

## Goals

- Develop a high-difficulty, human-like fighter pilot AI using modern reinforcement learning techniques.
- Implement scalable difficulty levels with customizable error modeling and "pilot personality" traits.
- Utilize TacView telemetry and supervised imitation learning to enhance realism and learning efficiency.
- Maintain modularity to allow iterative upgrades and fallback to vanilla DCS AI logic during partial deployment.

## Training Gameplan

### 🛫 Phase 1: Low-Fidelity Environment Bootstrapping *(In Progress)*
- ✅ Training pipeline validated using `LunarLanderContinuous-v2` as a stand-in for basic flight dynamics.
- ✅ PPO-based agent achieving safe landings in some training runs.
- ✅ Video recording integrated via `RecordVideo`.
- ✅ Model checkpointing and final model saving in place.
- ✅ Custom metrics callback implemented (placeholder phase) to support domain-specific tracking.
- 🔧 Environment unwrapping utility added to support robust metric extraction from wrapped Gym environments.
- ✅ Custom 2D Flight Environment implemented with energy management fundamentals (pitch/throttle control).
- ✅ Comprehensive training pipeline with logging, evaluation, and checkpointing for `TwoDFlightEnv`.
- ✅ Real-time matplotlib visualization with aircraft markers, pitch vectors, and flight path trails.
- ✅ TacView-compatible telemetry logging system with CSV and basic ACMI export capabilities.
- 🔜 Next: Begin curriculum learning with takeoff and basic flight maneuvers.

### 🧠 Phase 2: Curriculum Learning & Domain Transfer
- Train agents in a curriculum: takeoff → navigation → formation flying → basic air combat maneuvers.
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
- **Matplotlib** – Real-time flight visualization and telemetry display
- **MoviePy** – Training visualization (turnkey video export)
- **TensorBoard** – Metrics logging and performance visualization
- **Poetry** – Dependency and environment management
- **TacView** (Planned) – Flight telemetry recording and visualization
- **Unity** or **Custom Gym Environments** (Planned) – For flight dynamics simulation

## Features

- ✅ Automatic video recording of training episodes
- ✅ Manual and automatic saving of training sessions, with timestamped run directories
- ✅ Custom `FinalModelCallback` for consistent model archival
- ✅ `CustomMetricCallback` with placeholders for domain-specific metrics (thrust, fuel efficiency, landing quality)
- ✅ Run ID tagging for training sessions
- ✅ TensorBoard logging for core and custom metrics
- ✅ Environment unwrapping utility for safe metric extraction from wrapped environments
- ✅ Optional video toggle to conserve storage
- ✅ Real-time matplotlib visualization with aircraft position tracking and flight path trails
- ✅ Comprehensive telemetry logging with CSV export and basic TacView ACMI format support
- ✅ Energy state visualization showing kinetic/potential energy management in real-time
- 🔜 Enhanced TacView integration for full ACMI replay compatibility

## Status

🚀 Actively developing Phase 1.
Current focus: collecting architecture feedback, refining metrics, and preparing to move toward a custom 2D flight simulation environment.

