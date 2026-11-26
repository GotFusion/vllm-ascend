/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "hixl_bridge.h"
#include "hixl/hixl.h"

// Do not include Torch, torch_npu or their vendored GE headers in this file.
namespace avsg_urma {
struct HixlBridge::Impl {
    hixl::Hixl engine;
};

HixlBridge::HixlBridge() : impl_(std::make_unique<Impl>()) {}
HixlBridge::~HixlBridge() = default;

uint32_t HixlBridge::Initialize(const std::string &local, const std::map<std::string, std::string> &options)
{
    std::map<hixl::AscendString, hixl::AscendString> opts;
    for (const auto &entry : options) {
        opts.emplace(entry.first.c_str(), entry.second.c_str());
    }
    opts[hixl::OPTION_ENABLE_AIV_DIRECT] = "1";
    opts[hixl::OPTION_ENABLE_AIV_UBMEM] = "0";
    return impl_->engine.Initialize(hixl::AscendString(local.c_str()), opts);
}

uint32_t HixlBridge::RegisterMem(uint64_t base, uint64_t bytes)
{
    hixl::MemDesc desc{};
    desc.addr = base;
    desc.len = bytes;
    hixl::MemHandle handle = nullptr;
    // HIXL retains registered memory handles until Finalize; Endpoint retains the Tensor owners.
    return impl_->engine.RegisterMem(desc, hixl::MEM_DEVICE, handle);
}

uint32_t HixlBridge::Connect(const std::string &peer, int timeout)
{
    return impl_->engine.Connect(hixl::AscendString(peer.c_str()), timeout);
}

uint32_t HixlBridge::Channels(const std::string &peer, std::vector<uint64_t> &channels)
{
    return impl_->engine.GetAivChannelPointers(hixl::AscendString(peer.c_str()), channels);
}

uint32_t HixlBridge::RemoteAddress(const std::string &peer, uint64_t base, uint64_t bytes, uint64_t &addr)
{
    uintptr_t remote = 0;
    const auto st = impl_->engine.GetAddr(hixl::AscendString(peer.c_str()), base, bytes,
                                         hixl::AIV_MEM_TYPE_URMA, remote);
    addr = remote;
    return st;
}

uint32_t HixlBridge::Disconnect(const std::string &peer)
{
    return impl_->engine.Disconnect(hixl::AscendString(peer.c_str()));
}

void HixlBridge::Finalize()
{
    impl_->engine.Finalize();
}
} // namespace avsg_urma
