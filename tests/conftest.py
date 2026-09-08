import pytest

from cpuattn.hardware.host import Cache, Host, detect_host


@pytest.fixture(scope="session")
def runtime_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("runtime-cache")


@pytest.fixture(scope="session")
def runtime_host() -> Host:
    detected = detect_host()
    first = detected.cpus[0]
    same_node = [
        cpu
        for cpu in detected.cpus
        if cpu.numa_node == first.numa_node and cpu.core_id != first.core_id
    ]
    cpus = (first, same_node[0]) if same_node else (first,)
    cpu_ids = tuple(cpu.cpu_id for cpu in cpus)
    return Host(
        detected.architecture,
        detected.vendor,
        detected.model,
        detected.features,
        cpus,
        (
            Cache(1, "Data", 32 << 10, cpu_ids),
            Cache(2, "Unified", 256 << 10, cpu_ids),
            Cache(3, "Unified", 2 << 20, cpu_ids),
        ),
    )
