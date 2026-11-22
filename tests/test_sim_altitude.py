"""Unit tests for synthetic aerial simulation and height estimation accuracy."""

from pathlib import Path
import numpy as np
import pytest
import cv2

from vps.config import CameraConfig, TrackerConfig
from vps.tools.terrain_generator import DeccanTerrainGenerator
from vps.tools.simulator import DroneCameraSimulator, DroneState, SimulatorDegradationConfig
from vps.tools.evaluate_sim import run_altitude_benchmark


@pytest.fixture(scope="module")
def shared_terrain(tmp_path_factory):
    fn = tmp_path_factory.mktemp("sim_data") / "test_terrain.png"
    gen = DeccanTerrainGenerator(width_px=1024, height_px=1024, meters_per_pixel=0.1, seed=123)
    img = gen.generate(fn)
    return fn, img


class TestSimulator:
    def test_terrain_generation(self, shared_terrain):
        fn, img = shared_terrain
        assert img.shape == (1024, 1024, 3)
        assert img.dtype == np.uint8
        # Ensure image has sufficient contrast / variance
        assert np.std(img) > 15.0

    def test_simulator_render_validity(self, shared_terrain):
        _, ortho = shared_terrain
        sim = DroneCameraSimulator(orthophoto=ortho, meters_per_pixel=0.1)

        state = DroneState(
            timestamp_s=0.0,
            pos_ned_m=np.array([50.0, 50.0, -100.0]),
            vel_ned_mps=np.array([10.0, 0.0, 0.0]),
            attitude_rpy_rad=np.array([0.0, 0.0, 0.0]),
            angular_rates_rad_s=np.array([0.0, 0.0, 0.0]),
        )
        frame = sim.render_frame(state)
        assert frame.shape == (1088, 1456, 3)
        assert frame.dtype == np.uint8
        assert not np.isnan(frame).any()

    def test_haze_attenuation(self, shared_terrain):
        _, ortho = shared_terrain
        deg_cfg = SimulatorDegradationConfig(sensor_noise_sigma=0.0, exposure_time_s=0.0, haze_beta=0.005)
        sim = DroneCameraSimulator(orthophoto=ortho, meters_per_pixel=0.1, degradation_config=deg_cfg)

        # Low altitude (high contrast, low haze)
        state_low = DroneState(
            timestamp_s=0.0,
            pos_ned_m=np.array([50.0, 50.0, -20.0]),
            vel_ned_mps=np.array([0.0, 0.0, 0.0]),
            attitude_rpy_rad=np.array([0.0, 0.0, 0.0]),
            angular_rates_rad_s=np.array([0.0, 0.0, 0.0]),
        )
        frame_low = sim.render_frame(state_low)

        # High altitude (low contrast, high haze)
        state_high = DroneState(
            timestamp_s=0.0,
            pos_ned_m=np.array([50.0, 50.0, -300.0]),
            vel_ned_mps=np.array([0.0, 0.0, 0.0]),
            attitude_rpy_rad=np.array([0.0, 0.0, 0.0]),
            angular_rates_rad_s=np.array([0.0, 0.0, 0.0]),
        )
        frame_high = sim.render_frame(state_high)

        std_low = np.std(frame_low)
        std_high = np.std(frame_high)
        # High altitude haze reduces contrast
        assert std_high < std_low


class TestSimulationBenchmark:
    def test_sim_height_from_flow_accuracy(self, tmp_path):
        """Test height-from-flow in simulator across 60 to 300 m meets <= 5% error requirement."""
        terrain_file = tmp_path / "bench_terrain.png"
        gen = DeccanTerrainGenerator(width_px=2048, height_px=2048, meters_per_pixel=0.1, seed=42)
        gen.generate(terrain_file)

        plot_path = tmp_path / "test_benchmark.png"
        test_alts = [60.0, 100.0, 200.0, 300.0]

        res = run_altitude_benchmark(
            terrain_path=terrain_file,
            output_plot_path=plot_path,
            altitudes=test_alts,
        )

        assert plot_path.is_file()

        for r in res["results"]:
            assert r["is_valid"], f"Altitude {r['alt_true']}m failed validity check"
            # Acceptance requirement: <= 5% altitude error at 60-300 m on textured imagery
            assert r["error_pct"] <= 5.0, (
                f"Altitude {r['alt_true']}m error {r['error_pct']:.2f}% exceeds 5% limit! "
                f"Est: {r['alt_est']:.1f}m"
            )
            assert r["confidence"] >= 0.5
            assert r["inliers"] >= 15

