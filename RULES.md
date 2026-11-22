# Engineering & Operational Rules for VPS Phase 1

---

### Rule 1: Workspace & Filesystem Isolation
1. **Strictly Work Within `e:\VPS\`**:
   - All source code, configs, test assets, tools, synthetic datasets, and outputs must reside inside `e:\VPS\`.
   - Never write to or modify files in `C:\` (apart from internal agent scratch/artifact directories managed automatically by the agent harness).

### Rule 2: Physical Drone & Remote Access Safety
1. **Zero Direct Network / SSH Access to Drone**:
   - The user is connected to the drone via SSH. The companion computer has active software installed.
   - **Never** attempt to execute SSH, SCP, or remote commands targeting the drone or any remote hardware directly.
   - All code development, testing, and validation must occur in the local laptop environment first.

### Rule 3: Execution Environment & Multi-Platform Compatibility
1. **Laptop-First Verification**:
   - Every module, tool, unit test, and simulator must run and pass locally on this laptop (Windows / Python 3.10+) before considering deployment.
2. **Raspberry Pi 5 Target Alignment**:
   - The primary target companion computer is **Raspberry Pi 5** (ARM64 / Linux) connected to the flight controller (ArduPilot) via serial UART.
   - Jetson Orin is also supported via configurable assumptions.
   - Avoid architecture-specific proprietary binaries. Rely on standard Python, NumPy, OpenCV, and pymavlink.
   - Video capture must support standard OpenCV V4L2 and mock streams, with clear configuration for PiCamera2 / V4L2 on Raspberry Pi OS.

### Rule 4: Physical Testing & Hand-Held Baseline Height
1. **Hand-Held Ground Calibration**:
   - Physical testing will involve holding the drone by hand and raising/lowering it to verify AGL measurements.
   - The system must incorporate a baseline ground calibration offset (vertical distance from camera optical center to landing gear/ground contact plane when resting or held at baseline).
   - Low-altitude estimation ($0.1\text{ m} - 2\text{ m}$) must handle ground baseline offset accurately so resting on the ground measures 0.0 m AGL.

### Rule 5: Scope & Architectural Boundaries
1. **Phase 1 Height (AGL) Exclusivity**:
   - Phase 1 scope is strictly vertical height (AGL) estimation.
   - Do NOT implement horizontal position estimation or visual odometry in Phase 1.
   - Keep internal data structures, state vectors, and tracker interfaces modular and extensible for Phase 2 integration.
2. **Zero ROS Dependency in Core**:
   - Core libraries (`vps.camera`, `vps.tracker`, `vps.geometry`, `vps.mavlink`, `vps.fusion`, `vps.tools`) must have **zero** ROS / ROS 2 dependencies.
   - The ROS 2 node is an optional, decoupled, thin wrapper scheduled exclusively for Milestone 8.

### Rule 6: Code Quality & Mathematical Rigor
1. **Type Annotations & Documentation**:
   - All code must feature strict PEP 484 type annotations and descriptive docstrings detailing coordinate frames, units, and mathematical assumptions.
2. **No Invented Intrinsics**:
   - Never hardcode or invent camera intrinsics. Always load from calibrated YAML configuration files.
3. **Deterministic Unit Testing**:
   - Mathematical modules (de-rotation, undistortion, height-from-flow, scale-expansion, Kalman filter) must be verified via unit tests with known analytic ground truth.

### Rule 7: Milestone Governance & Approval Gate
1. **Step-by-Step Approval Protocol**:
   - Work strictly in designated milestones (M1 through M8).
   - After completing each milestone, provide a concise summary of results, generated plots/metrics, and open questions.
   - **Always halt and wait for explicit user approval before proceeding to code the next milestone.**

