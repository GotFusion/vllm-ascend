/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "avsg_urma_abi.h"
#include <iostream>
int main()
{
    using namespace avsg_urma;
    std::cout << sizeof(Context) << " " << offsetof(Context, localKv) << " "
              << sizeof(LaunchArgs) << "\n";
    return sizeof(Context) == 192 && offsetof(Context, localKv) == 160 && sizeof(LaunchArgs) == 72 ? 0 : 1;
}
