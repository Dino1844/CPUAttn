#ifndef MINI_CPUATTN_PACKING_H
#define MINI_CPUATTN_PACKING_H

#include <stdint.h>
#include <stddef.h>

static inline int cpuattn_owner_slot(
    int group, const int *owner_groups, int owner_count) {
    for (int slot = 0; slot < owner_count; ++slot) {
        if (owner_groups[slot] == group) return slot;
    }
    return -1;
}

static inline int cpuattn_local_packed_slot(
    int group, const int *owner_groups, int owner_count) {
    int slot = cpuattn_owner_slot(group, owner_groups, owner_count);
    return slot >= 0 ? slot : 0;
}

static inline void cpuattn_pack_k_for_owner(
    const float *restrict k,
    float *restrict packed_k,
    size_t packed_copy_bytes,
    int64_t B,
    int64_t HKV,
    int64_t SKV,
    int64_t D,
    int64_t packed_skv,
    const int *worker_groups,
    int workers,
    const int *owner_groups,
    int owner_count) {
    int worker = omp_get_thread_num();
    int group = worker_groups[worker];
    int slot = cpuattn_owner_slot(group, owner_groups, owner_count);
    if (slot < 0) return;

    int group_rank;
    int group_workers;
    cpuattn_group_position(
        worker_groups, workers, worker, &group_rank, &group_workers);
    int64_t begin;
    int64_t end;
    cpuattn_static_range(
        B * HKV * D, group_rank, group_workers, &begin, &end);
    float *copy = (float *)((unsigned char *)packed_k
        + (size_t)slot * packed_copy_bytes);
    for (int64_t panel = begin; panel < end; ++panel) {
        int64_t d = panel % D;
        int64_t hkv = (panel / D) % HKV;
        int64_t b = panel / (D * HKV);
        const float *source = k + ((b * HKV + hkv) * SKV) * D + d;
        float *target = copy + ((b * HKV + hkv) * D + d) * packed_skv;
        int64_t ki = 0;
        for (; ki < SKV; ++ki) target[ki] = source[ki * D];
        for (; ki < packed_skv; ++ki) target[ki] = 0.0f;
    }
}

#endif
