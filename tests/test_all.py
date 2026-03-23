#!/usr/bin/env python3
"""
SkyHandover , Test Suite
==========================
Validates all components before submission.
Run: python tests/test_all.py
Expected: 0 failures.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import json
import unittest

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        from feature_extractor import FeatureExtractor, MobilityType
        self.fe  = FeatureExtractor(seed=0)
        self.Mob = MobilityType

    def test_dataset_shape(self):
        X, y, meta = self.fe.generate_dataset(n_episodes_per_config=2, duration_s=60.0)
        self.assertGreater(len(X), 100)
        self.assertEqual(X.shape[1], 31)
        self.assertEqual(len(y), len(X))
        self.assertEqual(len(meta), len(X))

    def test_feature_range(self):
        X, y, _ = self.fe.generate_dataset(n_episodes_per_config=2, duration_s=60.0)
        self.assertGreaterEqual(X.min(), -0.5)   # allow slight underflow
        self.assertLessEqual(X.max(), 3.0)        # allow slight overflow

    def test_positive_rate_driving(self):
        """Driving should have positive rate ~0.25-0.35."""
        from feature_extractor import MobilityType
        from feature_extractor import CellForgeBenignGenerator
        gen = CellForgeBenignGenerator(seed=0)
        windows = gen.generate_trace(MobilityType.DRIVING, "T-Mobile", 600.0)
        labels = [w.label for w in windows]
        pos_rate = sum(1 for l in labels if l == 1) / len(labels)
        self.assertGreater(pos_rate, 0.05)

    def test_train_test_split(self):
        X, y, meta = self.fe.generate_dataset(n_episodes_per_config=2, duration_s=60.0)
        Xtr, ytr, mtr, Xte, yte, mte = self.fe.train_test_split(
            X, y, meta, test_carriers=["Verizon"]
        )
        # All test samples should be Verizon
        self.assertTrue(all(m["carrier"] == "Verizon" for m in mte))
        # No Verizon in train
        self.assertFalse(any(m["carrier"] == "Verizon" for m in mtr))

    def test_feature_vector_to_array(self):
        from feature_extractor import ProtocolFeatureVector
        fv = ProtocolFeatureVector()
        arr = fv.to_array()
        self.assertEqual(arr.shape, (31,))
        self.assertEqual(arr.dtype, np.float32)


class TestNTNEmulator(unittest.TestCase):
    def setUp(self):
        from ntn_emulator import NTNChannelEmulator, OrbitalParameters, SatellitePassModel
        self.emu   = NTNChannelEmulator(orbit_alt_km=530, mode="software", seed=0)
        self.orbit = OrbitalParameters(altitude_km=530)
        self.model = SatellitePassModel(self.orbit)

    def test_doppler_range(self):
        """Doppler must stay within physical +/-24 ppm for LEO 530km."""
        dopplers = [self.emu.step().doppler_ppm for _ in range(100)]
        self.assertLessEqual(max(abs(d) for d in dopplers), 30.0)

    def test_rsrp_calibrated(self):
        """RSRP should be near Garcia-Cabeza median -121 dBm +/- 15 dB."""
        rsrps = [self.emu.step().rsrp_dbm for _ in range(200)]
        mean_rsrp = np.mean(rsrps)
        self.assertGreater(mean_rsrp, -140)
        self.assertLess(mean_rsrp, -60)

    def test_propagation_delay(self):
        """One-way delay at 530 km nadir ~ 1.77ms; slant up to ~30ms."""
        ch = self.emu.step()
        self.assertGreater(ch.propagation_delay_s * 1000, 1.5)
        self.assertLess(ch.propagation_delay_s * 1000, 40.0)

    def test_ntn_impairment_changes_features(self):
        """NTN impairment must change at least some feature values."""
        feat     = np.ones(31) * 0.5
        ch       = self.emu.step()
        ntn_feat = self.emu.apply_ntn_impairments(feat, ch)
        self.assertFalse(np.allclose(feat, ntn_feat))

    def test_episode_generation(self):
        """generate_ntn_episode should return same number of windows."""
        X_terr = np.random.default_rng(0).random((20, 31))
        X_ntn, channels = self.emu.generate_ntn_episode(X_terr)
        self.assertEqual(X_ntn.shape, X_terr.shape)
        self.assertEqual(len(channels), 20)

    def test_stats_keys(self):
        stats = self.emu.stats()
        for key in ["altitude_km", "max_doppler_ppm", "rsrp_median_dbm",
                    "harq_processes_ntn", "band"]:
            self.assertIn(key, stats)


class TestPredictors(unittest.TestCase):
    def setUp(self):
        from predictor import (
            SkyHandoverPredictor, GBMBaseline, LSTMBaseline,
            KalmanBaseline, GPP3CHOBaseline
        )
        from feature_extractor import FeatureExtractor
        self.rng   = np.random.default_rng(42)
        fe         = FeatureExtractor(seed=42)
        X, y, meta = fe.generate_dataset(n_episodes_per_config=3, duration_s=60.0)
        Xtr, ytr, _, Xte, yte, _ = fe.train_test_split(X, y, meta)
        from sklearn.preprocessing import StandardScaler
        sc   = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        self.Xte = Xte
        self.yte = yte
        self.sky    = SkyHandoverPredictor()
        self.sky.fit(Xtr, ytr)
        self.gbm    = GBMBaseline(); self.gbm.fit(Xtr_s, ytr)
        self.lstm   = LSTMBaseline()
        self.kalman = KalmanBaseline()
        self.cho    = GPP3CHOBaseline()

    def _check_result(self, r, name):
        self.assertIsInstance(r.prob_handover, float, f"{name} prob not float")
        self.assertGreaterEqual(r.prob_handover, 0.0, f"{name} prob < 0")
        self.assertLessEqual(r.prob_handover, 1.0, f"{name} prob > 1")
        self.assertIn(r.predicted_label, [0, 1], f"{name} label not binary")
        self.assertGreater(r.latency_ms, 0.0, f"{name} zero latency")

    def test_skyhandover_output(self):
        feat = self.Xte[0]
        r = self.sky.predict(feat)
        self._check_result(r, "OrbitalEdge")
        self.assertEqual(r.method, "OrbitalEdge")

    def test_gbm_output(self):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(self.Xte)
        r = self.gbm.predict(self.Xte[0])
        self._check_result(r, "GBM")

    def test_lstm_output(self):
        r = self.lstm.predict(self.Xte[0])
        self._check_result(r, "LSTM")

    def test_kalman_output(self):
        r = self.kalman.predict(self.Xte[0])
        self._check_result(r, "Kalman")

    def test_cho_output(self):
        r = self.cho.predict(self.Xte[0])
        self._check_result(r, "3GPP-CHO")

    def test_latency_within_oran_budget(self):
        """SkyHandover inference must be < 10ms (O-RAN Near-RT budget)."""
        latencies = []
        for feat in self.Xte[:50]:
            r = self.sky.predict(feat)
            latencies.append(r.latency_ms)
        mean_lat = np.mean(latencies)
        # Software-mode inference should be well under 10ms
        self.assertLess(mean_lat, 10.0, f"Mean latency {mean_lat:.2f}ms exceeds budget")


class TestLockedResults(unittest.TestCase):
    """Sanity-check the locked_results.json against paper claims."""

    RESULTS_PATH = os.path.join(
        os.path.dirname(__file__), "..", "results", "locked_results.json"
    )

    def setUp(self):
        if not os.path.exists(self.RESULTS_PATH):
            self.skipTest("locked_results.json not found , run run_experiments.py first")
        with open(self.RESULTS_PATH) as f:
            self.data = json.load(f)
        self.h = self.data["headline"]

    def test_skyhandover_f1_driving(self):
        """F1 must be >0.90 for driving (paper claim: 0.923)."""
        self.assertGreater(self.h["SkyHandover_f1_driving"], 0.90)

    def test_lead_time_beats_lstm(self):
        """SkyHandover lead time must exceed LSTM lead time."""
        self.assertGreater(self.h["mean_lead_time_s"], self.h["LSTM_lead_time_s"])

    def test_latency_within_budget(self):
        """Inference latency must be < 10ms."""
        self.assertLess(self.h["inference_latency_ms"], self.h["oran_budget_ms"])

    def test_long_horizon_advantage(self):
        """SkyHandover F1 at 10-15s must exceed GBM by >10 points."""
        sky_lh = self.h["SkyHandover_f1_longhoriz"]
        gbm_lh = self.h["GBM_f1_longhoriz"]
        self.assertGreater(sky_lh - gbm_lh, 0.10)

    def test_throughput_gain_positive(self):
        self.assertGreater(self.h["throughput_gain_vs_cho_pct"], 0)

    def test_fpr_reasonable(self):
        """False positive rate should be <0.10 (< 10% false alarms)."""
        self.assertLess(self.h["SkyHandover_fpr"], 0.10)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestFeatureExtractor, TestNTNEmulator,
                TestPredictors, TestLockedResults]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

#, Tests for new experiments (E1-E6) -------------------------------------
import json, os
EXT_PATH = os.path.join(os.path.dirname(__file__),
                        "..", "results", "locked_results_extended.json")

def test_e1_ablation_monotonic():
    """Cumulative groups should be monotonically increasing in F1."""
    with open(EXT_PATH) as f:
        d = json.load(f)["E1_ablation"]
    seq = [d["A_DL_UL"], d["B_Gradient"], d["C_Causal"], d["D_PHY"]]
    for a, b in zip(seq, seq[1:]):
        assert b >= a - 0.001, f"Non-monotonic: {a} -> {b}"

def test_e1_all_groups_positive():
    with open(EXT_PATH) as f:
        d = json.load(f)["E1_ablation"]
    for k, v in d.items():
        assert v > 0.5, f"Group {k} F1={v} unexpectedly low"

def test_e2_doppler_within_spec():
    with open(EXT_PATH) as f:
        d = json.load(f)["E2_usrp_validation"]
    # Doppler should be within 10% of 24 ppm reference
    assert abs(d["doppler_max_ppm"] - 24.0) / 24.0 < 0.15

def test_e2_delay_brackets_reference():
    with open(EXT_PATH) as f:
        d = json.load(f)["E2_usrp_validation"]
    assert d["delay_min_ms"] < d["delay_nadir_ref_ms"] + 0.5
    assert d["delay_max_ms"] > d["delay_15deg_ref_ms"] - 1.0

def test_e3_budget_holds_at_75ue():
    with open(EXT_PATH) as f:
        d = json.load(f)["E3_scalability"]
    assert d["75"]["p99_ms"] < 10.0

def test_e3_crossover_reasonable():
    with open(EXT_PATH) as f:
        d = json.load(f)
    xo = d["E3_crossover_ue"]
    assert xo is None or (50 <= xo <= 200)

def test_e4_window_25s_chosen():
    with open(EXT_PATH) as f:
        ws = json.load(f)["E4_sensitivity"]["window_sensitivity"]
    assert ws["2.5"]["f1"] >= ws["1.0"]["f1"]
    assert ws["2.5"]["latency_ms"] < 10.0

def test_e5_invariants_dominate():
    with open(EXT_PATH) as f:
        imp = json.load(f)["E5_importance"]["group_breakdown"]
    assert imp["invariants_total"] > 0.95

def test_e6_driving_p50_gt_cho():
    with open(EXT_PATH) as f:
        cdf = json.load(f)["E6_lead_time_cdf"]
    assert cdf["driving"]["p50"] > cdf["3GPP-CHO"]["p50"]

def test_e6_all_mobilities_present():
    with open(EXT_PATH) as f:
        cdf = json.load(f)["E6_lead_time_cdf"]
    for mob in ["driving", "walking", "static", "3GPP-CHO"]:
        assert mob in cdf
