#!/usr/bin/env python3
"""
tools/inspect_wddm.py
Step 5: Windows Graphics Driver & WDDM 2.x Boundary Investigation

Probes the OS-to-Silicon abstraction layer on Windows 11:
  1. DXGI 1.4+ / IDXGIAdapter3:
     - GPU hardware identification (Vendor, Device ID, LUID)
     - Dedicated VRAM vs Shared System Memory allocations
     - WDDM Memory Segment Budgets & Current OS Usage via QueryVideoMemoryInfo
  2. NVIDIA Driver WDDM vs TCC Operating Mode:
     - Driver version and execution model
     - WDDM Command Queue submission overhead
     - Host-visible page mapping and unified memory attributes
"""

import ctypes
from ctypes import wintypes
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "wddm_boundary"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# DXGI GUIDs and COM Structures
IID_IDXGIFactory4 = ctypes.c_char_p(b"\x1b\xc6\x3c\x1b\xee\xb4\x45\x24\xbf\x15\x26\x9e\x1d\x8e\x8d\x62")
DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0
DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL = 1


class LUID(ctypes.Structure):
    _fields_ = [
        ("LowPart", wintypes.DWORD),
        ("HighPart", wintypes.LONG)
    ]


class DXGI_ADAPTER_DESC2(ctypes.Structure):
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", LUID),
        ("Flags", wintypes.UINT),
        ("GraphicsPreemptionGranularity", wintypes.UINT),
        ("ComputePreemptionGranularity", wintypes.UINT)
    ]


class DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [
        ("Budget", ctypes.c_uint64),
        ("CurrentUsage", ctypes.c_uint64),
        ("AvailableForReservation", ctypes.c_uint64),
        ("CurrentReservation", ctypes.c_uint64)
    ]


def inspect_dxgi():
    adapters_info = []
    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
    except Exception as e:
        print(f"[!] Unable to load dxgi.dll: {e}")
        return adapters_info

    # CreateDXGIFactory1(REFIID riid, void **ppFactory)
    # IID_IDXGIFactory1 = {770aae78-f26f-4dba-a829-253c83d1b387}
    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", wintypes.BYTE * 8)
        ]

    IID_IDXGIFactory1 = GUID(
        0x770aae78, 0xf26f, 0x4dba,
        (wintypes.BYTE * 8)(0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87)
    )

    pFactory = ctypes.c_void_p()
    hr = dxgi.CreateDXGIFactory1(ctypes.byref(IID_IDXGIFactory1), ctypes.byref(pFactory))
    if hr != 0:
        print(f"[!] CreateDXGIFactory1 failed: hr = {hr:#x}")
        return adapters_info

    # IDXGIFactory1 Vtbl:
    # 0: QueryInterface, 1: AddRef, 2: Release
    # 7: EnumAdapters1(UINT Adapter, IDXGIAdapter1 **ppAdapter)
    factory_vtbl = ctypes.cast(ctypes.cast(pFactory, ctypes.POINTER(ctypes.c_void_p)).contents,
                               ctypes.POINTER(ctypes.c_void_p))
    EnumAdapters1_proto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))
    EnumAdapters1 = EnumAdapters1_proto(factory_vtbl[7])

    IID_IDXGIAdapter3 = GUID(
        0x64596774, 0x4151, 0x4fb8,
        (wintypes.BYTE * 8)(0xbb, 0x80, 0x5e, 0xa0, 0x42, 0xa9, 0x1b, 0x45)
    )

    idx = 0
    while True:
        pAdapter = ctypes.c_void_p()
        hr = EnumAdapters1(pFactory, idx, ctypes.byref(pAdapter))
        if hr != 0:
            break

        adapter_vtbl = ctypes.cast(ctypes.cast(pAdapter, ctypes.POINTER(ctypes.c_void_p)).contents,
                                   ctypes.POINTER(ctypes.c_void_p))
        
        # QueryInterface for IDXGIAdapter3 to access DXGI 1.4 memory queries
        QueryInterface_proto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))
        QueryInterface = QueryInterface_proto(adapter_vtbl[0])

        pAdapter3 = ctypes.c_void_p()
        has_adapter3 = (QueryInterface(pAdapter, ctypes.byref(IID_IDXGIAdapter3), ctypes.byref(pAdapter3)) == 0)

        # GetDesc2 if Adapter3, else GetDesc1
        # In IDXGIAdapter1: 8: GetDesc1(DXGI_ADAPTER_DESC1 *pDesc)
        class DXGI_ADAPTER_DESC1(ctypes.Structure):
            _fields_ = [
                ("Description", wintypes.WCHAR * 128),
                ("VendorId", wintypes.UINT),
                ("DeviceId", wintypes.UINT),
                ("SubSysId", wintypes.UINT),
                ("Revision", wintypes.UINT),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", LUID),
                ("Flags", wintypes.UINT)
            ]

        GetDesc1_proto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1))
        GetDesc1 = GetDesc1_proto(adapter_vtbl[8])

        desc = DXGI_ADAPTER_DESC1()
        GetDesc1(pAdapter, ctypes.byref(desc))

        vram_mb = desc.DedicatedVideoMemory / (1024 * 1024)
        sys_mb = desc.DedicatedSystemMemory / (1024 * 1024)
        shared_mb = desc.SharedSystemMemory / (1024 * 1024)

        adapter_dict = {
            "index": idx,
            "description": desc.Description,
            "vendor_id": f"0x{desc.VendorId:04X}",
            "device_id": f"0x{desc.DeviceId:04X}",
            "subsys_id": f"0x{desc.SubSysId:08X}",
            "revision": desc.Revision,
            "dedicated_vram_mb": vram_mb,
            "dedicated_sys_mb": sys_mb,
            "shared_sys_mb": shared_mb,
            "luid": f"0x{desc.AdapterLuid.HighPart:08X}:0x{desc.AdapterLuid.LowPart:08X}",
            "flags": desc.Flags
        }

        # Query WDDM Video Memory Info (Local VRAM vs Non-Local RAM)
        if has_adapter3:
            # IDXGIAdapter3 Vtbl:
            # 10: QueryVideoMemoryInfo(UINT NodeIndex, DXGI_MEMORY_SEGMENT_GROUP MemorySegmentGroup, DXGI_QUERY_VIDEO_MEMORY_INFO *pVideoMemoryInfo)
            adapter3_vtbl = ctypes.cast(ctypes.cast(pAdapter3, ctypes.POINTER(ctypes.c_void_p)).contents,
                                        ctypes.POINTER(ctypes.c_void_p))
            QueryVideoMemoryInfo_proto = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, wintypes.UINT, ctypes.c_int, ctypes.POINTER(DXGI_QUERY_VIDEO_MEMORY_INFO)
            )
            QueryVideoMemoryInfo = QueryVideoMemoryInfo_proto(adapter3_vtbl[10])

            # Local VRAM
            local_info = DXGI_QUERY_VIDEO_MEMORY_INFO()
            if QueryVideoMemoryInfo(pAdapter3, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, ctypes.byref(local_info)) == 0:
                adapter_dict["wddm_local_segment"] = {
                    "budget_mb": local_info.Budget / (1024 * 1024),
                    "current_usage_mb": local_info.CurrentUsage / (1024 * 1024),
                    "reservation_mb": local_info.CurrentReservation / (1024 * 1024),
                    "available_for_reservation_mb": local_info.AvailableForReservation / (1024 * 1024)
                }

            # Non-Local (GTT Shared System RAM)
            non_local_info = DXGI_QUERY_VIDEO_MEMORY_INFO()
            if QueryVideoMemoryInfo(pAdapter3, 0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, ctypes.byref(non_local_info)) == 0:
                adapter_dict["wddm_non_local_segment"] = {
                    "budget_mb": non_local_info.Budget / (1024 * 1024),
                    "current_usage_mb": non_local_info.CurrentUsage / (1024 * 1024),
                    "reservation_mb": non_local_info.CurrentReservation / (1024 * 1024),
                    "available_for_reservation_mb": non_local_info.AvailableForReservation / (1024 * 1024)
                }

        adapters_info.append(adapter_dict)
        idx += 1

    return adapters_info


def inspect_cuda_wddm_attributes():
    try:
        cuda = ctypes.WinDLL("nvcuda.dll")
    except Exception as e:
        return {"error": str(e)}

    cuda.cuInit(0)
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)

    # Device attributes per cuda.h (Driver API)
    attr_names = {
        1: "MAX_THREADS_PER_BLOCK",
        2: "MAX_BLOCK_DIM_X",
        19: "CAN_MAP_HOST_MEMORY",
        31: "CONCURRENT_KERNELS",
        33: "PCI_BUS_ID",
        34: "PCI_DEVICE_ID",
        35: "TCC_DRIVER",  # 0 = WDDM, 1 = TCC
        36: "MEMORY_CLOCK_RATE_KHZ",
        37: "GLOBAL_MEMORY_BUS_WIDTH_BITS",
        38: "L2_CACHE_SIZE_BYTES",
        39: "MAX_THREADS_PER_MULTIPROCESSOR",
        40: "ASYNC_ENGINE_COUNT",
        41: "UNIFIED_ADDRESSING",
        90: "COMPUTE_PREEMPTION_SUPPORTED",
        96: "DIRECT_MANAGED_MEM_ACCESS_FROM_HOST"
    }

    attrs = {}
    for attr_id, name in attr_names.items():
        val = ctypes.c_int()
        if cuda.cuDeviceGetAttribute(ctypes.byref(val), attr_id, dev) == 0:
            attrs[name] = val.value

    # Measure WDDM Command Submission Overhead
    # Measure time taken by CPU to enqueue a null kernel / event record
    ctx = ctypes.c_void_p()
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

    evt = ctypes.c_void_p()
    cuda.cuEventCreate(ctypes.byref(evt), 0)

    # Warmup
    cuda.cuEventRecord(evt, None)
    cuda.cuEventSynchronize(evt)

    # Profile CPU submission latency
    N_SAMPLES = 1000
    t0 = time.perf_counter()
    for _ in range(N_SAMPLES):
        cuda.cuEventRecord(evt, None)
    t1 = time.perf_counter()

    cuda.cuEventSynchronize(evt)
    avg_submit_us = ((t1 - t0) / N_SAMPLES) * 1e6
    attrs["wddm_cmd_submit_latency_us"] = round(avg_submit_us, 3)

    return attrs


def main():
    print("=" * 75)
    print(" Step 5: Windows Graphics Driver & WDDM 2.x Boundary Investigation")
    print("=" * 75)

    print("[*] Probing DXGI / WDDM Adapters...")
    dxgi_adapters = inspect_dxgi()

    for adp in dxgi_adapters:
        print(f"\n[+] Adapter [{adp['index']}]: {adp['description']}")
        print(f"    - Hardware IDs: Vendor={adp['vendor_id']}, Device={adp['device_id']}, SubSys={adp['subsys_id']}")
        print(f"    - Adapter LUID: {adp['luid']}")
        print(f"    - Dedicated VRAM: {adp['dedicated_vram_mb']:.1f} MB")
        print(f"    - Shared System RAM: {adp['shared_sys_mb']:.1f} MB")
        if "wddm_local_segment" in adp:
            loc = adp["wddm_local_segment"]
            print(f"    - WDDM Local VRAM Budget: {loc['budget_mb']:.1f} MB (Usage: {loc['current_usage_mb']:.1f} MB)")
        if "wddm_non_local_segment" in adp:
            nloc = adp["wddm_non_local_segment"]
            print(f"    - WDDM Non-Local SysRAM Budget: {nloc['budget_mb']:.1f} MB (Usage: {nloc['current_usage_mb']:.1f} MB)")

    print("\n" + "-" * 75)
    print("[*] Probing NVIDIA Driver UMD/KMD Operating Model...")
    cuda_attrs = inspect_cuda_wddm_attributes()

    is_tcc = cuda_attrs.get("TCC_DRIVER", 0)
    driver_mode = "Tesla Compute Cluster (TCC)" if is_tcc == 1 else "Windows Display Driver Model (WDDM)"
    print(f"    - Driver Execution Mode: {driver_mode} (TCC_DRIVER={is_tcc})")
    print(f"    - Async DMA Engines: {cuda_attrs.get('ASYNC_ENGINE_COUNT', 'N/A')}")
    print(f"    - Unified Addressing (UVA): {'Supported' if cuda_attrs.get('UNIFIED_ADDRESSING') else 'No'}")
    print(f"    - Hardware Preemption Supported: {'Yes' if cuda_attrs.get('COMPUTE_PREEMPTION_SUPPORTED') else 'No'}")
    print(f"    - Can Map Host Memory (Zero-Copy): {'Yes' if cuda_attrs.get('CAN_MAP_HOST_MEMORY') else 'No'}")
    print(f"    - Direct Managed Mem Access from Host: {'Yes' if cuda_attrs.get('DIRECT_MANAGED_MEM_ACCESS_FROM_HOST') else 'No'}")
    print(f"    - WDDM Command Buffer Enqueue Overhead: {cuda_attrs.get('wddm_cmd_submit_latency_us', 'N/A')} us / submission")

    summary = {
        "dxgi_adapters": dxgi_adapters,
        "cuda_wddm_attributes": cuda_attrs
    }

    out_file = RESULTS_DIR / "wddm_boundary_profile.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print(f"[SUCCESS] Saved WDDM boundary profile to: {out_file}\n")


if __name__ == "__main__":
    main()
