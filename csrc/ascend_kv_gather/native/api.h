// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// All calls use the caller's already initialized ACL device context. They do
// not initialize, finalize, or reset Torch's runtime. A session owns only its
// communicator, channel resources, stream, and small transfer workspaces.
size_t kvg_root_info_size(void);
int kvg_root_info(void* data, size_t capacity, char* error, size_t error_capacity);
int kvg_create(uint32_t rank, const void* root_info, const uint64_t* bases,
               const uint64_t* sizes, uint32_t regions, uint32_t cores,
               void** session, char* error, size_t error_capacity);
// On a create error, *session is normally null. If cleanup itself failed, a
// nonnull handle is returned and the caller must retain it and its bank tensors.
// Only rank 1 submits. Both ranks register caller-owned tensors in kvg_create.
// The caller validates remote_destination against the rank 0 receive region.
// Success means every AIV core has drained and the launch stream completed.
int kvg_transfer(void* session, uint64_t source, uint64_t remote_destination,
                 uint32_t row_bytes, const uint64_t* host_indices,
                 uint32_t count, char* error, size_t error_capacity);
// A failed destroy retains the handle and its bank tensors for safe cleanup or
// process shutdown. Only successful destruction invalidates the handle. Do not
// call concurrently with transfer; the Python owner serializes these calls.
int kvg_destroy(void* session, char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
