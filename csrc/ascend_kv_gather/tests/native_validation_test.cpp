// SPDX-License-Identifier: Apache-2.0
// Host-only boundary checks. This does not exercise HCOMM, CANN or a device.
#include "../native/validation.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>

using namespace ascend_kv_gather;

template <typename Function>
void Reject(Function function)
{
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

int main()
{
    const Region banks[] = {{0x10000, 1024 * 262144ULL}, {0x20010000, 128 * 262144ULL}};
    const std::uint64_t last[] = {262143};
    const std::uint64_t zero[] = {0};
    const auto max = std::numeric_limits<std::uint64_t>::max();
    assert(ValidateTransfer(banks, 2, banks[0].base, 0x40010000, 1024, last, 1) == 0);
    assert(ValidateTransfer(banks, 2, banks[1].base, 0x50010000, 128, last, 1) == 1);
    assert(ValidateTransfer(banks, 2, banks[0].base + banks[0].bytes - 1024, 0x40010000, 1024, zero, 1) == 0);
    const std::uint64_t outside[] = {262144};
    const std::uint64_t wrapping[] = {max / 1024 + 1};
    Reject([&] { ValidateTransfer(banks, 2, banks[0].base, 0x40010000, 1024, outside, 1); });
    Reject([&] { ValidateTransfer(banks, 2, banks[0].base, 0x40010000, 1024, wrapping, 1); });
    Reject([&] { ValidateTransfer(banks, 2, banks[0].base + 1024, 0x40010000, 1024, last, 1); });
    Reject([&] { ValidateTransfer(banks, 2, banks[0].base - 1, 0x40010000, 1024, zero, 1); });
    Reject([&] { ValidateTransfer(banks, 2, banks[0].base + 1, 0x40010000, 1024, zero, 1); });
    Reject([&] { ValidateTransfer(banks, 2, banks[1].base, 0x40010000, 1024, zero, 1); });
    Reject([&] { ValidateTransfer(banks, 2, banks[0].base, max - 511, 1024, zero, 1); });
    Reject([&] { ValidateTransfer(banks, 2, banks[0].base, 0, 1024, zero, 1); });
    Reject([&] { ValidateTransfer(banks, 2, banks[0].base, 0x40010000, 1024, nullptr, 1); });
    Reject([&] { ValidateTransfer(banks, 2, banks[0].base, 0x40010000, 1024, zero, 0); });
    Reject([&] { ValidateTransfer(banks, 2, banks[0].base, 0x40010000, 1024, zero, kMaxRows + 1); });
}
