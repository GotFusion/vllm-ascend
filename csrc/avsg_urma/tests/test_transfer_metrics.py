# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import unittest

from avsg_urma.protocol import device_timing_stats, transfer_plan, host_timing_stats


class TransferMetricsTest(unittest.TestCase):
    def test_host_copy_wall_time_and_fp8_payload(self):
        payload = 2048 * (512 + 64)
        stats = host_timing_stats(2_000_000, payload)
        self.assertEqual(stats["host_copy_us"], 2000)
        self.assertAlmostEqual(stats["host_bandwidth_gib_s"], payload / 0.002 / 1024 ** 3)
        self.assertEqual(host_timing_stats(1234, 0)["host_bandwidth_gib_s"], 0)
        for elapsed, size in ((0, payload), (-1, payload), (1000, -1)):
            with self.subTest(elapsed=elapsed, size=size), self.assertRaises(ValueError):
                host_timing_stats(elapsed, size)

    def test_host_stage_breakdown_and_zero_transfer(self):
        stats = host_timing_stats(2_000_000, 2048 * 576, 400_000, 800_000)
        self.assertEqual(stats["host_descriptor_us"], 400)
        self.assertEqual(stats["host_transfer_us"], 800)
        self.assertEqual(stats["host_other_us"], 800)
        zero = host_timing_stats(1_000_000, 0, 300_000, 0)
        self.assertEqual(zero["host_transfer_us"], 0)
        self.assertEqual(zero["host_bandwidth_gib_s"], 0)
        for desc, transfer in ((-1, 1), (1, -1), (600_000, 600_000)):
            with self.subTest(desc=desc, transfer=transfer), self.assertRaises(ValueError):
                host_timing_stats(1_000_000, 100, desc, transfer)

    def test_exclusive_and_shared_groups(self):
        self.assertEqual(transfer_plan(4)["strategy"], "exclusive")
        self.assertEqual(transfer_plan(4, 8, "shared")["active_jettys"], 4)
        # HIXL ceil grouping does not always use every configured jetty.
        self.assertEqual(transfer_plan(4, 5)["active_jettys"], 3)
        self.assertEqual(transfer_plan(1, 64)["active_jettys"], 1)

    def test_reject_incompatible_strategy_and_grid(self):
        for args in ((4, 8, "exclusive"), (4, 4, "shared"), (4, 3), (1, 65),
                     (0,), (33,), (1, True), (1, 1, "unknown")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                transfer_plan(*args)

    def stats(self, rows, tokens=3):
        return device_timing_stats(rows, transfer_plan(2), tokens=tokens, kv_dim=512,
                                   rope_dim=64, element_bytes=2, lanes=32)

    def test_max_completion_not_sum_or_absolute_timestamps(self):
        rows = [[0] * 16 for _ in range(32)]
        rows[0][:5] = [1000, 2000, 0, 4, 1]
        rows[1][:5] = [2000, 4000, 0, 2, 1]
        rows[0][12:14] = [1000000, 2000000] # absolute slots are not the denominator
        stats = self.stats(rows)
        self.assertEqual(stats["completion_us"], 4)
        self.assertEqual(stats["issue_us"], 2)
        self.assertEqual(stats["payload_bytes"], 3 * 576 * 2)
        self.assertEqual(stats["descriptors"], 6)
        self.assertAlmostEqual(stats["devbw_gib_s"], 3456 / 4e-6 / 1024 ** 3)

    def test_zero_error_and_missing_timing(self):
        rows = [[0] * 16 for _ in range(32)]
        stats = self.stats(rows, tokens=0)
        self.assertFalse(stats["valid"])
        self.assertEqual(stats["devbw_gib_s"], 0)
        with self.assertRaises(RuntimeError):
            self.stats(rows)
        rows[0][3] = 6
        with self.assertRaises(RuntimeError):
            self.stats(rows)  # work exists but completion timing is absent
        rows[0][1] = 1000
        rows[1][2] = 4
        with self.assertRaises(RuntimeError):
            self.stats(rows)  # device error cannot be reported as valid bandwidth

    def test_shared_uses_leader_rows_and_kv_only(self):
        plan = transfer_plan(4, 5, "shared")
        rows = [[0] * 16 for _ in range(32)]
        for row in rows[:3]:
            row[:5] = [100, 200, 0, 1, 1]
        stats = device_timing_stats(rows, plan, tokens=3, kv_dim=8, rope_dim=0,
                                    element_bytes=1, lanes=8)
        self.assertEqual(stats["payload_bytes"], 24)
        self.assertEqual(stats["publish_count"], 3)
