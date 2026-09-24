/**
 * tools/inspect_cpu_topology.cpp
 * Inspects Windows Logical Processor Information for exact SMT sibling masks.
 */
#include <windows.h>
#include <iostream>
#include <vector>

int main() {
    DWORD length = 0;
    GetLogicalProcessorInformation(nullptr, &length);
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    GetLogicalProcessorInformation(buffer.data(), &length);

    int core_idx = 0;
    for (const auto& info : buffer) {
        if (info.Relationship == RelationProcessorCore) {
            std::cout << "Physical Core " << core_idx++ 
                      << ": Affinity Mask = 0x" << std::hex << info.ProcessorMask 
                      << " (SMT Enabled: " << (info.ProcessorCore.Flags == 1 ? "Yes" : "No") << ")\n";
        }
    }
    return 0;
}
