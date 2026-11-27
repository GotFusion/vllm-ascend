# SPDX-License-Identifier: Apache-2.0
"""Eager HCOMM KV gather using the vendored Ascend multi-SGE vector kernel."""

from .session import KVGatherClient, KVGatherPeer, parse_endpoint

__all__ = ("KVGatherClient", "KVGatherPeer", "parse_endpoint")
