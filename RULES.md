# Engineering & Operational Rules for VPS (Phase 1 & Phase 2)

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
2. **Raspberry Pi 5 Target Alignment (Design for Pi 5 from Day 1)**:
   - The primary target companion computer is **Raspberry Pi 5** (ARM64 / Linux, 4-core Cortex-A76, NO GPU/CUDA) connected to ArduPilot via serial UART at 921,600 baud.
   - Budget constraints: tracking loop $\ge 25\text{ fps}$, end-to-end latency $< 100\text{ ms}$, map matching $\ge 1\text{ Hz}$, total CPU $< 80\%$.
   - Multi-process pipeline with shared memory IPC (`multiprocessing.shared_memory`) and core pinning (`os.sched_setaffinity`).
   - Run tracking at reduced resolution ($728 \times 544$, scale intrinsics accordingly), cap features (300–500), use OpenCV with NEON optimization.
   - Power and thermal monitoring: poll undervoltage and CPU throttling flags (`vcgencmd get_throttled`).

### Rule 4: Dynamic Altitude Envelope (10 m to 400 m AGL) & Hand-Held Testing
1. **10 m to 400 m Operating Envelope**:
   - Optical flow, adaptive keyframing, and GSD calculations must be valid across $10\text{ m}$ to $400\text{ m}$ AGL.
2. **Hand-Held Ground Baseline Zeroing**:
   - Physical testing will involve holding the drone by hand and raising/lowering it to verify AGL measurements.
   - The system must incorporate a calibrated ground baseline offset ($15\text{ cm}$ clearance from camera optical center to landing gear contact plane).
   - Low-altitude estimation ($0.1\text{ m} - 2\text{ m}$) must subtract the ground baseline offset so resting on the ground measures exactly $0.000\text{ m}$ AGL.

### Rule 5: Phase Boundaries & Sensor Modalities
1. **Phase 1 (Height AGL) vs Phase 2 (Horizontal GNSS-Denied Navigation)**:
   - Do NOT start Phase 2 until Phase 1 simulator acceptance criteria are met and approved by the user.
2. **Sensor Modalities**:
   - Ingest IMU (high-rate gyro + accel), Compass (magnetometer heading / FC yaw), and VIO (visual-inertial odometry), alongside baro.
3. **Zero ROS Dependency in Core**:
   - Core libraries must have **zero** ROS / ROS 2 dependencies. The ROS 2 node is an optional decoupled wrapper.

### Rule 6: Code Quality & Mathematical Rigor
1. **Type Annotations & Documentation**:
   - All code must feature strict PEP 484 type annotations and descriptive docstrings detailing coordinate frames, units, and mathematical assumptions.
2. **No Invented Intrinsics**:
   - Never hardcode or invent camera intrinsics. Always load from calibrated YAML configuration files.
3. **Deterministic Unit Testing**:
   - Mathematical modules must be verified via unit tests with known analytic ground truth.

### Rule 7: Milestone Governance & Approval Gate
1. **Step-by-Step Approval Protocol**:
   - Work strictly in designated milestones (M1 through M12).
   - After completing each milestone, provide a concise summary of results, generated plots/metrics, and open questions.
   - **Always halt and wait for explicit user approval before proceeding to code the next milestone.**
