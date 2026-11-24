/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * CPU-only HIXL boundary fake; never used by the extension's CMake target.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#ifdef AVSG_TEST_TORCH_GE
#error "HIXL and Torch GE headers must not share a translation unit"
#endif
// The second include resolves differently if a framework include root leaks
// ahead of CANN: this reproduces the user's failing hixl_bridge.cpp build.
#include "graph/error_codes.h"
#include "graph/ge_error_codes.h"
namespace hixl {
using AscendString = std::string;
using MemHandle = void *;
constexpr const char *OPTION_ENABLE_AIV_DIRECT = "EnableAivDirect";
constexpr const char *OPTION_ENABLE_AIV_UBMEM = "EnableAivUbmem";
constexpr uint32_t AIV_MEM_TYPE_URMA = 0;
enum MemType { MEM_DEVICE, MEM_HOST };
struct MemDesc { uintptr_t addr; size_t len; };
class Hixl {
public:
    uint32_t Initialize(const AscendString &local, const std::map<AscendString, AscendString> &options)
    {
        return local == "local" && options.at(OPTION_ENABLE_AIV_DIRECT) == "1" &&
               options.at(OPTION_ENABLE_AIV_UBMEM) == "0" ? 0 : 1;
    }
    uint32_t RegisterMem(const MemDesc &desc, MemType type, MemHandle &handle)
    {
        handle = reinterpret_cast<void *>(desc.addr);
        return desc.addr == 4096 && desc.len == 1024 && type == MEM_DEVICE ? 0 : 2;
    }
    uint32_t Connect(const AscendString &peer, int timeout) { return peer == "remote" && timeout == 123 ? 0 : 3; }
    uint32_t GetAivChannelPointers(const AscendString &peer, std::vector<uint64_t> &channels)
    {
        if (peer != "remote") return 4;
        channels = {0x1000, 0x2000};
        return 0;
    }
    uint32_t GetAddr(const AscendString &peer, uintptr_t base, size_t bytes, uint32_t type, uintptr_t &addr)
    {
        if (peer != "remote" || bytes != 1024 || type != AIV_MEM_TYPE_URMA) return 5;
        addr = base;
        return 0;
    }
    uint32_t Disconnect(const AscendString &peer) { return peer == "remote" ? 0 : 6; }
    void Finalize() {}
};
} // namespace hixl
