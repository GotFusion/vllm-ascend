/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef AVSG_URMA_HIXL_BRIDGE_H
#define AVSG_URMA_HIXL_BRIDGE_H
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace avsg_urma {
// Keep HIXL/GE definitions out of translation units that include torch_npu.
// This boundary must only use standard C++ types, never hixl/ge types or headers.
class HixlBridge {
public:
    HixlBridge();
    ~HixlBridge();
    HixlBridge(const HixlBridge &) = delete;
    HixlBridge &operator=(const HixlBridge &) = delete;

    uint32_t Initialize(const std::string &local, const std::map<std::string, std::string> &options);
    uint32_t RegisterMem(uint64_t base, uint64_t bytes);
    uint32_t Connect(const std::string &peer, int timeout);
    uint32_t Channels(const std::string &peer, std::vector<uint64_t> &channels);
    uint32_t RemoteAddress(const std::string &peer, uint64_t base, uint64_t bytes, uint64_t &addr);
    uint32_t Disconnect(const std::string &peer);
    void Finalize();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace avsg_urma
#endif
