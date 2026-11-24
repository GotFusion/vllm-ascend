# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
from .protocol import RemoteCache

__all__ = ["AvsgUrmaSession", "AvsgUrmaCacheServer", "RemoteCache"]


def __getattr__(name):
    if name in ("AvsgUrmaSession", "AvsgUrmaCacheServer"):
        from . import session
        return getattr(session, name)
    raise AttributeError(name)
