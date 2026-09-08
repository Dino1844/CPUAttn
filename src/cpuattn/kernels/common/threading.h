#ifndef MINI_CPUATTN_THREADING_H
#define MINI_CPUATTN_THREADING_H

#if defined(__linux__)
#include <sched.h>
typedef struct {
    cpu_set_t set;
    int valid;
} cpuattn_affinity_state;
static inline int cpuattn_pin_worker(
    const int *cpu_ids, int workers, cpuattn_affinity_state *original) {
    int worker = omp_get_thread_num();
    if (worker >= workers) return -1;
    original->valid = 0;
    if (sched_getaffinity(0, sizeof(original->set), &original->set) != 0) return -1;
    original->valid = 1;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_ids[worker], &set);
    return sched_setaffinity(0, sizeof(set), &set);
}
static inline int cpuattn_restore_worker(const cpuattn_affinity_state *original) {
    if (!original->valid) return -1;
    return sched_setaffinity(0, sizeof(original->set), &original->set);
}
#else
typedef struct { int valid; } cpuattn_affinity_state;
static inline int cpuattn_pin_worker(
    const int *cpu_ids, int workers, cpuattn_affinity_state *original) {
    (void)cpu_ids;
    (void)workers;
    original->valid = 1;
    return 0;
}
static inline int cpuattn_restore_worker(const cpuattn_affinity_state *original) {
    (void)original;
    return 0;
}
#endif

static inline void cpuattn_static_range(
    int64_t count, int rank, int workers, int64_t *begin, int64_t *end) {
    *begin = count * rank / workers;
    *end = count * (rank + 1) / workers;
}

static inline void cpuattn_group_position(
    const int *worker_groups,
    int workers,
    int worker,
    int *group_rank,
    int *group_workers) {
    int group = worker_groups[worker];
    int rank = 0;
    int count = 0;
    for (int index = 0; index < workers; ++index) {
        if (worker_groups[index] != group) continue;
        if (index < worker) ++rank;
        ++count;
    }
    *group_rank = rank;
    *group_workers = count;
}

#endif
