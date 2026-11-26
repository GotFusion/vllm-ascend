/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#define AVSG_TEST_TORCH_GE
#include <cstdint>
// Model a framework header defining the same GE symbols as the HIXL dependency.
// The bridge's public header must compile here without leaking HIXL/GE types.
namespace ge {
using graphStatus = uint32_t;
[[maybe_unused]] constexpr graphStatus GRAPH_FAILED = 0xffffffffU;
}
#include "hixl_bridge.h"
#include <cassert>
#include <iostream>

int main()
{
    avsg_urma::HixlBridge bridge;
    assert(bridge.Initialize("local", {{"EnableAivDirect", "0"}, {"EnableAivUbmem", "1"}}) == 0);
    assert(bridge.RegisterMem(4096, 1024) == 0);
    assert(bridge.RegisterMem(4096, 2048) == 2);
    assert(bridge.Connect("remote", 123) == 0);
    assert(bridge.Connect("remote", 0) == 3);
    std::vector<uint64_t> channels;
    assert(bridge.Channels("remote", channels) == 0);
    assert((channels == std::vector<uint64_t>{0x1000, 0x2000}));
    assert(bridge.Channels("unknown", channels) == 4);
    uint64_t addr = 0;
    assert(bridge.RemoteAddress("remote", 0x9000, 1024, addr) == 0 && addr == 0x9000);
    assert(bridge.RemoteAddress("remote", 0x9000, 2048, addr) == 5);
    assert(bridge.Disconnect("remote") == 0);
    assert(bridge.Disconnect("unknown") == 6);
    bridge.Finalize();
    std::cout << "PASS: isolated HIXL/GE bridge, option forcing, forwarding and error codes (CPU fake SDK)\n";
}
