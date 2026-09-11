#ifndef MINI_CPUATTN_EXECUTOR_H
#define MINI_CPUATTN_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

static inline uint64_t cpuattn_now_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t)value.tv_sec * 1000000000ull + (uint64_t)value.tv_nsec;
}

static inline size_t cpuattn_page_size(void) {
    long value = sysconf(_SC_PAGESIZE);
    return value > 0 ? (size_t)value : 4096u;
}

static inline size_t cpuattn_align_size(size_t value, size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

int cpuattn_prepare_launch(const int *cpu_ids, int workers) {
    if (!cpu_ids || workers <= 0) return -1;
    int affinity_error = 0;
    #pragma omp parallel num_threads(workers) reduction(|:affinity_error)
    {
        cpuattn_affinity_state original_affinity;
        affinity_error |= omp_get_num_threads() != workers;
        affinity_error |= cpuattn_pin_worker(cpu_ids, workers, &original_affinity) != 0;
        #pragma omp barrier
        affinity_error |= cpuattn_restore_worker(&original_affinity) != 0;
    }
    return affinity_error ? 1 : 0;
}

#endif
