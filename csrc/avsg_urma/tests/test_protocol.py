# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import struct
import unittest

from avsg_urma.protocol import HEADER_WORDS, MAGIC, VERSION, RemoteCache, pack_context, descriptor_bytes, graph_bitmap_words


def remote(**kwargs):
    args = dict(kv_base=0x100000, kv_bytes=4 * 128 * 1024, blocks=4, block_size=128,
                kv_dim=512, kv_stride0=128 * 1024, kv_stride1=1024, dtype="bfloat16",
                rope_base=0x200000, rope_bytes=4 * 128 * 128, rope_dim=64,
                rope_stride0=128 * 128, rope_stride1=128)
    return RemoteCache(**(args | kwargs))


class ProtocolTest(unittest.TestCase):
    def test_graph_bitmap_alignment(self):
        self.assertEqual(graph_bitmap_words(1025, 1), 33)
        self.assertEqual(graph_bitmap_words(1024, 4), 4 * 32)
        self.assertEqual(graph_bitmap_words(1025, 4), 4 * 64)

    def test_fp8_layout_and_descriptor_byte_width(self):
        for dtype, code in (("float8_e4m3fn", 3), ("float8_e5m2", 4)):
            with self.subTest(dtype=dtype):
                r = remote(dtype=dtype, blocks=64, kv_bytes=8192 * 512,
                           kv_stride0=128 * 512, kv_stride1=512,
                           rope_bytes=8192 * 64, rope_stride0=128 * 64, rope_stride1=64)
                kv, rope = r.addresses(8191, 4095, 0x900000, 0xc00000)
                self.assertEqual(kv, (r.kv_base + 8191 * 512, 0x900000 + 4095 * 512, 512))
                self.assertEqual(rope, (r.rope_base + 8191 * 64, 0xc00000 + 4095 * 64, 64))
                blob = pack_context(r, [123], generation=1, batch=32,
                                    local_kv=0x900000, local_kv_bytes=4096 * 512,
                                    local_rope=0xc00000, local_rope_bytes=4096 * 64)
                words = struct.unpack(f"<{HEADER_WORDS + 1}Q", blob)
                self.assertEqual(words[6], 1)
                self.assertEqual(words[11], code)

    def test_physical_block_and_distinct_rope_stride(self):
        r = remote()
        kv, rope = r.addresses(259, 7, 0x300000, 0x400000)
        self.assertEqual(kv, (0x100000 + 2 * 128 * 1024 + 3 * 1024, 0x300000 + 7 * 1024, 1024))
        self.assertEqual(rope, (0x200000 + 2 * 128 * 128 + 3 * 128, 0x400000 + 7 * 128, 128))

    def test_strided_source(self):
        r = remote(kv_stride1=2048, kv_stride0=128 * 2048, kv_bytes=4 * 128 * 2048)
        self.assertEqual(r.addresses(129, 0, 4096, 8192)[0][0], 0x100000 + 129 * 2048)

    def test_opaque_high_bit_pointer_roundtrip(self):
        channels = (0xF000000000000100, 0xF000000000000200)
        blob = pack_context(remote(), channels, generation=1, batch=32,
                            local_kv=0x300000, local_kv_bytes=128 * 1024,
                            local_rope=0x400000, local_rope_bytes=128 * 128)
        words = struct.unpack(f"<{HEADER_WORDS + 2}Q", blob)
        self.assertEqual(words[:3], (MAGIC, VERSION, len(blob)))
        self.assertEqual(words[-2:], channels)
        self.assertEqual(words[20], 0x300000)

    def test_no_rope(self):
        r = remote(rope_base=0, rope_bytes=0, rope_dim=0, rope_stride0=0, rope_stride1=0)
        self.assertEqual(len(r.addresses(0, 0, 4096, 0)), 1)
        blob = pack_context(r, [0x1000], generation=1, batch=8,
                            local_kv=0x300000, local_kv_bytes=1024)
        self.assertEqual(struct.unpack_from("<Q", blob, 10 * 8)[0], 0)

    def test_reject_bad_ranges_strides_and_dimensions(self):
        for kwargs in ({"kv_bytes": 1024}, {"block_size": 127}, {"kv_stride1": 128},
                       {"kv_base": 2**64 - 100}, {"rope_stride0": 1}, {"kv_dim": 1025},
                       {"dtype": "float32"}, {"rope_base": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                remote(**kwargs)

    def test_reject_shared_channel_and_invalid_indices(self):
        with self.assertRaises(ValueError):
            pack_context(remote(), [123, 123], generation=1, batch=32, local_kv=4096,
                         local_kv_bytes=1024, local_rope=8192, local_rope_bytes=128)
        for src, dst in ((-1, 0), (512, 0), (0, -1)):
            with self.subTest(src=src, dst=dst), self.assertRaises(ValueError):
                remote().addresses(src, dst, 4096, 8192)

    def test_manifest_roundtrip(self):
        r = remote()
        self.assertEqual(RemoteCache(**r.to_dict()), r)

    def test_full_descriptor_capacity_and_alignment(self):
        self.assertEqual(descriptor_bytes(0, True), 128)
        for entries in (1, 3, 97, 1024):
            for rope in (False, True):
                size = descriptor_bytes(entries, rope)
                self.assertEqual(size % 128, 0)
                self.assertGreaterEqual(size, entries * (2 if rope else 1) * 24)
        with self.assertRaises(ValueError):
            descriptor_bytes(-1, False)
        with self.assertRaises(ValueError):
            pack_context(remote(), [123], generation=1, batch=4, local_kv=4096,
                         local_kv_bytes=1024, local_rope=8192, local_rope_bytes=128)


if __name__ == "__main__":
    unittest.main()
