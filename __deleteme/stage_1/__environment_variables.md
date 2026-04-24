# @TODO:
1. Corrections for LLM-produced stuff:
   1. ~~Check `NCCL_CONFIG_DEFAULT` macro calls to discover the default values for different configuration parameters.~~ ~~✅ Done — all `NCCL_PARAM` and `NCCL_CONFIG_DEFAULT` defaults filled in.~~
2. ~~`NCCL_NVML_DIRECT` in `@/src/include/nvmlwrap.h`~~
    1. ~~@TODO: check if we need to pass this stuff to no rely on copy-pasted variant~~
        1.~~ NOTE: `NCCL_NVML_DIRECT` is not used in Ray or Torch.~~
    2. ~~Check how Ray, Torch, and other guys link NCCL - directly or over `dlopen()`~~
3. `@/src/graph/search.cc#L818`
    1. @TODO+: It's where NCCL parses topology file. We need to embed it into our container and AMI for Grace-Blackwell instances (it should be downloadable from NVIDIA or even be installed w/ drivers). GB10 has it's pre-installed on the host side, #    but we'll still need to put that into container to make it visible to the userspace NCCL installation.


# NCCL Environment Variables — Complete Reference

> **Generated from**: NVIDIA docs (v2.29) + exhaustive codebase scan of eugo-inc/nccl-cmake (v2.29.3-1)
>
> **Total**: 223 unique `NCCL_*` env vars (200 in `src/`, 23 in `ext-*/` plugins) + 1 non-NCCL (`CUDA_LAUNCH_BLOCKING`)
>
> **How NCCL reads env vars**: Most use the `NCCL_PARAM(Name, "SUFFIX", default)` macro which reads `NCCL_SUFFIX` from environment.
> String-valued vars use `ncclGetEnv("NCCL_FOO")`. A few use raw `getenv()`.
> Env vars can also be set in a conf file: `${NCCL_CONF_FILE}` if set, otherwise `~/.nccl.conf`.
>
> **Conf file loading — `initEnvFunc` (`src/misc/param.cc`)**:
> The conf file is loaded by `initEnvFunc()` in `src/misc/param.cc`, called once at startup.
> Load order: `$NCCL_CONF_FILE` (if set and non-empty) → `~/.nccl.conf` (if `$HOME` is set).
>
> **Known upstream bugs fixed by eugo (@EUGO_CHANGE in `src/misc/param.cc`)**:
> 1. **Unconditional `/etc/nccl.conf` override** — upstream always called `setEnvFile("/etc/nccl.conf")` after the user file, silently overriding any values already loaded from `$NCCL_CONF_FILE` or `~/.nccl.conf`. We removed those two calls.
> 2. **1024-byte path truncation** — `confFilePath` is a fixed `char[1024]`; paths longer than 1023 bytes are silently truncated (upstream limitation, not fixed).

> NCCL also implements a wrapper to SET enviroment variables, but as of now it's only used within test plugin so has no interest for us. `ncclOsSetEnv`
> However, we should be careful and future also examine all environment variables and system configuration parameters the given package sets to avoid collisions.

---

## Deployment Scenario Key

| Symbol | Scenario |
|--------|----------|
| **SN** | Single-node multi-GPU (dev/test, no IB) |
| **IB** | Multi-node with InfiniBand/RoCE (production HPC) |
| **TCP** | Multi-node with TCP/IP only (cloud VMs without RDMA) |
| **ALL** | Applies to all scenarios |

**Recommendation symbols**: ✅ = recommended to set, 🔧 = set if needed, ⚙️ = leave default, 🚫 = do not set in production

---

## Table of Contents

1. [Network & Socket Configuration](#1-network--socket-configuration)
2. [InfiniBand / RoCE Configuration](#2-infiniband--roce-configuration)
3. [Out-of-Band & Bootstrap](#3-out-of-band--bootstrap)
4. [Debug & Logging](#4-debug--logging)
5. [Topology & Graph](#5-topology--graph)
6. [Transport: P2P / SHM / Net](#6-transport-p2p--shm--net)
7. [GPU Direct RDMA (GDR)](#7-gpu-direct-rdma-gdr)
8. [NVLink / NVLS / MNNVL](#8-nvlink--nvls--mnnvl)
9. [Algorithms, Protocols & Tuning](#9-algorithms-protocols--tuning)
10. [CUDA Memory & Registration](#10-cuda-memory--registration)
11. [Kernel Launch & Threading](#11-kernel-launch--threading)
12. [CollNet & Plugins](#12-collnet--plugins)
13. [GIN (GPU-Initiated Networking)](#13-gin-gpu-initiated-networking)
14. [RAS (Reliability, Availability, Serviceability)](#14-ras-reliability-availability-serviceability)
15. [Miscellaneous Internal](#15-miscellaneous-internal)
16. [Profiler Plugin Env Vars (ext-profiler/)](#16-profiler-plugin-env-vars-ext-profiler)
17. [Inspector Plugin Env Vars (ext-profiler/inspector/)](#17-inspector-plugin-env-vars-ext-profilerinspector)
18. [Tuner Plugin Env Vars (ext-tuner/)](#18-tuner-plugin-env-vars-ext-tuner)
19. [Non-NCCL Env Vars Referenced](#19-non-nccl-env-vars-referenced)
20. [Build-Only Env Vars](#20-build-only-env-vars)
21. [Summary: Quick-Start Recommendations](#21-summary-quick-start-recommendations)

---

## 1. Network & Socket Configuration

These control TCP/IP socket behavior for bootstrap and socket-based transports.

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_SOCKET_IFNAME` | IP interface filter for NCCL communication. Comma-separated prefixes. `^` excludes, `=` for exact match. Default algo prefers `ib*` over others, excludes `lo`/`docker*`. | Auto-detected | — | `src/misc/socket.cc` | ✅ **IB/TCP**: Set to your data-plane interface (e.g., `=eth0` or `=ens1f0`). **SN**: ⚙️ leave default |
| `NCCL_SOCKET_FAMILY` | Force IPv4 (`AF_INET`) or IPv6 (`AF_INET6`) | Auto | — | `src/misc/socket.cc` | ⚙️ Leave default unless you have a mixed IPv4/IPv6 environment |
| `NCCL_SOCKET_RETRY_CNT` | Socket connection retry count after ETIMEDOUT/ECONNREFUSED/EHOSTUNREACH | `34` | 2.24 | `src/misc/socket.cc` | ⚙️ Default gives ~60s total retry. Increase for flaky networks |
| `NCCL_SOCKET_RETRY_SLEEP_MSEC` | Initial sleep (ms) before retry. Scales linearly: total = (N+1)*N/2 * sleep_ms | `100` | 2.24 | `src/misc/socket.cc` | ⚙️ Leave default |
| `NCCL_SOCKET_POLL_TIMEOUT_MSEC` | Poll timeout (ms) during bootstrap to reduce CPU usage. 0 = no poll (old behavior) | `0` | 2.28 | `src/misc/socket.cc` | 🔧 Set `100` for CPU-constrained environments |
| `NCCL_SOCKET_NTHREADS` | CPU helper threads per socket transport connection | `1` (AWS: `2`, GCP gVNIC: `4`) | 2.4.8 | `src/transport/net_socket.cc` | 🔧 **TCP 100G**: Set `4`. Product with `NSOCKS_PERTHREAD` ≤ 64 |
| `NCCL_NSOCKS_PERTHREAD` | Sockets per helper thread (for speed-limited sockets) | `1` (AWS: `8`) | 2.4.8 | `src/transport/net_socket.cc` | 🔧 **TCP 100G**: Set `4`. Product with `SOCKET_NTHREADS` ≤ 64 |
| `NCCL_SOCKET_RCVBUF` | Socket receive buffer size | OS default | — | `src/misc/socket.cc` | ⚙️ Leave default unless tuning TCP |
| `NCCL_SOCKET_SNDBUF` | Socket send buffer size | OS default | — | `src/misc/socket.cc` | ⚙️ Leave default unless tuning TCP |
| `NCCL_SOCKET_INLINE` | Socket inline threshold | `128` (1<<7) | — | `src/transport/net_socket.cc` | ⚙️ Internal |
| `NCCL_SOCKET_MIN_TASKSIZE` | Minimum task size for socket transport | `65536` (64 KiB) | — | `src/transport/net_socket.cc` | ⚙️ Internal |

---

## 2. InfiniBand / RoCE Configuration

Controls for IB Verbs transport. Only relevant with RDMA-capable NICs.

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_IB_DISABLE` | Disable IB/RoCE transport entirely; fall back to sockets | `0` | — | `src/transport/net_ib/init.cc` | ✅ **TCP/SN**: Set `1`. **IB**: Leave `0` |
| `NCCL_IB_HCA` | Filter IB Verbs devices. Comma-separated, `^` excludes, `=` exact match, `:port` for specific port. Max 32 HCAs. | All available | — | `src/transport/net_ib/init.cc` | 🔧 **IB**: Set if you need specific NICs (e.g., `=mlx5_0:1,mlx5_1:1`) |
| `NCCL_IB_TIMEOUT` | IB Verbs timeout. Timeout = 4.096µs × 2^value. Error 12 from ibv_poll_cq means too small. | `20` (was 18 pre-2.23, 14 pre-2.14) | — | `src/transport/net_ib/connect.cc` | ⚙️ Increase for very large fabrics. `0` or `≥32` = infinite |
| `NCCL_IB_RETRY_CNT` | IB retry count for failed transmissions | `7` | 2.1.15 | `src/transport/net_ib/connect.cc` | ⚙️ Leave default |
| `NCCL_IB_GID_INDEX` | Global ID index for RoCE mode | `-1` (auto) | 2.1.4 | `src/transport/net_ib/connect.cc` | 🔧 **IB/RoCE**: Set per `show_gids` output if auto-detect fails |
| `NCCL_IB_ADDR_FAMILY` | IP family for auto-selected GID when GID_INDEX unset | `AF_INET` | 2.21 | `src/transport/net_ib/connect.cc` | ⚙️ Leave default |
| `NCCL_IB_ADDR_RANGE` | CIDR range to filter GIDs when GID_INDEX unset | Unset (ignored) | 2.21 | `src/transport/net_ib/connect.cc` | 🔧 **IB**: Set if you need to restrict GID selection to a subnet |
| `NCCL_IB_ROCE_VERSION_NUM` | RoCE version for auto-selected GID | `2` | 2.21 | `src/transport/net_ib/connect.cc` | ⚙️ Leave default (RoCEv2) |
| `NCCL_IB_SL` | InfiniBand Service Level | `0` | 2.1.4 | `src/transport/net_ib/connect.cc` | 🔧 **IB**: Set per fabric admin instructions |
| `NCCL_IB_TC` | InfiniBand Traffic Class | `0` | 2.1.15 | `src/transport/net_ib/connect.cc` | 🔧 **IB**: Set per fabric QoS policy |
| `NCCL_IB_FIFO_TC` | IB Traffic Class for control messages (credit return). Allows high-priority control path. | Same as `NCCL_IB_TC` | 2.22.3 | `src/transport/net_ib/connect.cc` | 🔧 **IB**: Set to high-priority TC if available |
| `NCCL_IB_PKEY` | InfiniBand Partition Key | `0` | — | `src/transport/net_ib/connect.cc` | ⚙️ Leave default unless fabric uses partitioning |
| `NCCL_IB_RETURN_ASYNC_EVENTS` | Report fatal IB async events and stop IB comms | `1` | 2.23 | `src/transport/net_ib/common.cc` | ⚙️ Leave enabled |
| `NCCL_IB_QPS_PER_CONNECTION` | Queue pairs per connection. More QPs = better routing entropy on multi-level fabrics. | `1` | 2.10 | `src/transport/net_ib/connect.cc` | 🔧 **IB multi-tier**: Set `2-4` for multi-rail fabrics |
| `NCCL_IB_SPLIT_DATA_ON_QPS` | How multi-QP data is split. 0=round-robin, 1=split evenly per message. | `0` (since 2.20) | 2.18 | `src/transport/net_ib/connect.cc` | ⚙️ Leave `0` (round-robin) unless testing |
| `NCCL_IB_AR_THRESHOLD` | Message size threshold for adaptive-routing-friendly separate messages | `8192` | 2.6 | `src/transport/net_ib/p2p.cc` | ⚙️ Setting > BUFFSIZE disables AR entirely |
| `NCCL_IB_ADAPTIVE_ROUTING` | Enable adaptive routing capable data transfers | `1` (IB), `0` (RoCE) | 2.16 | `src/transport/net_ib/init.cc` | ⚙️ Enable on AR-capable fabrics |
| `NCCL_IB_PCI_RELAXED_ORDERING` | Enable PCIe Relaxed Ordering for IB. 0=off, 1=force, 2=auto | `2` (auto) | 2.12 | `src/transport/net_ib/init.cc` | ⚙️ Leave `2` for auto-detect. Helps in VMs |
| `NCCL_IB_ECE_ENABLE` | Enhanced Connection Establishment. Enables advanced features (CC, AR, SR). | `1` (since 2.19) | 2.23 | `src/transport/net_ib/connect.cc` | ⚙️ Leave enabled. Disable if ECE misconfigured at system level |
| `NCCL_IB_MERGE_NICS` | Combine dual-port NICs into single logical device for BW aggregation | `1` | 2.20 | `src/transport/net_ib/init.cc` | ⚙️ Leave enabled |
| `NCCL_IB_MERGE_VFS` | Merge VFs (Virtual Functions) in SR-IOV | `1` | — | `src/transport/net_ib/init.cc` | ⚙️ Internal, for VM/SR-IOV setups |
| `NCCL_IB_DATA_DIRECT` | Enable direct data path for IB | `1` | — | `src/transport/net_ib/init.cc` | ⚙️ Internal |
| `NCCL_IB_USE_INLINE` | Use inline data for small IB messages | `0` | — | `src/transport/net_ib/connect.cc` | ⚙️ Internal |
| `NCCL_IB_WARN_RAIL_LOCAL` | Warn about rail-local suboptimal NIC assignment | `0` | — | `src/transport/net_ib/connect.cc` | ⚙️ Internal debug |
| `NCCL_IB_ROUTABLE_FLID_GID_INDEX` | Routable FLID GID index | `1` | — | `src/transport/net_ib/connect.cc` | ⚙️ Internal |
| `NCCL_IB_CUDA_SUPPORT` | Force/disable GPU Direct RDMA (removed in 2.4, see `NCCL_NET_GDR_LEVEL`) | — | removed 2.4.0 | — | 🚫 Deprecated, use `NCCL_NET_GDR_LEVEL` |
| `NCCL_GDR_FLUSH_DISABLE` | Disable GDR flush | `0` | — | `src/transport/net_ib/connect.cc` | ⚙️ Internal |
| `NCCL_IB_MQP_RETRY_ALL` | Multi-QP retry for all operations | `0` | — | `src/misc/ibvwrap.cc` | ⚙️ Internal |
| `NCCL_IB_MQP_RETRY_CNT` | Multi-QP retry count | `34` | — | `src/misc/ibvwrap.cc` | ⚙️ Internal |
| `NCCL_IB_MQP_RETRY_SLEEP_MSEC` | Multi-QP retry sleep (ms) | `100` | — | `src/misc/ibvwrap.cc` | ⚙️ Internal |

---

## 3. Out-of-Band & Bootstrap

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_OOB_NET_ENABLE` | Use NCCL net (IB/Socket) for out-of-band init allgather | `0` | 2.23 | `src/bootstrap.cc` | 🔧 **IB**: Set `1` to use IB for OOB comms |
| `NCCL_OOB_NET_IFNAME` | Filter interfaces for OOB net (when OOB_NET_ENABLE=1) | — | 2.23 | `src/bootstrap.cc` | 🔧 Set if using OOB over specific IB device |
| `NCCL_UID_STAGGER_THRESHOLD` | Number of ranks above which bootstrap connections are staggered | `256` | 2.23 | `src/bootstrap.cc` | ⚙️ Reduce for smaller UniqueId sockets |
| `NCCL_UID_STAGGER_RATE` | Target message rate (msgs/sec) when staggering | `7000` | 2.23 | `src/bootstrap.cc` | ⚙️ Leave default |
| `NCCL_COMM_ID` | Manual communicator ID (host:port) for bootstrap | Auto | — | `src/bootstrap.cc`, `src/init.cc`, `src/misc/socket.cc` | 🔧 Set for manual bootstrap (no MPI/NCCL unique ID) |
| `NCCL_HOSTID` | Override host identification | Auto | — | `src/misc/utils.cc` | ⚙️ Internal, for environments where hostname detection fails |

---

## 4. Debug & Logging

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_DEBUG` | Debug verbosity: `VERSION`, `WARN`, `INFO`, `TRACE` | — (no debug) | — | `src/debug.cc` | ✅ **ALL**: Set `WARN` in production, `INFO` for troubleshooting, `TRACE` for deep debugging |
| `NCCL_DEBUG_FILE` | Redirect debug output to file. `%h`=hostname, `%p`=PID | stdout | 2.2.12 | `src/debug.cc` | 🔧 Set for multi-rank jobs: `/tmp/nccl-debug-%h-%p.log` |
| `NCCL_DEBUG_SUBSYS` | Filter INFO output by subsystem. CSV of: INIT, COLL, P2P, SHM, NET, GRAPH, TUNING, ENV, ALLOC, CALL, PROXY, NVLS, BOOTSTRAP, REG, PROFILE, RAS, ALL. `^` prefix excludes. | `INIT,BOOTSTRAP,ENV` | 2.3.4 | `src/debug.cc` | 🔧 Set `ALL` when debugging, `INIT,NET` for network issues |
| `NCCL_DEBUG_TIMESTAMP_FORMAT` | strftime format for log timestamps. `%Xf` for fractional seconds. Underscores render as spaces. | `[%F %T]` | 2.26 | `src/debug.cc` | ⚙️ Leave default. `[%F %T.%3f]` adds milliseconds |
| `NCCL_DEBUG_TIMESTAMP_LEVELS` | Which log levels get timestamps. CSV of VERSION, WARN, INFO, ABORT, TRACE, ALL. `^` inverts. | `WARN` | 2.26 | `src/debug.cc` | ⚙️ Leave default |
| `NCCL_WARN_ENABLE_DEBUG_INFO` | Include extra debug info in WARN messages | — | — | `src/debug.cc` | ⚙️ Internal |
| `NCCL_CONF_FILE` | Path to static configuration file (key=value lines); loaded by `initEnvFunc()` in `src/misc/param.cc`. Falls back to `~/.nccl.conf` if unset. **Note**: path is truncated to 1023 bytes (fixed `char[1024]` buffer — upstream limitation). **@EUGO_CHANGE**: upstream unconditionally called `setEnvFile("/etc/nccl.conf")` after the user file, overriding it; we removed that. | `~/.nccl.conf` | 2.23 | `src/misc/param.cc` | 🔧 Set to `/etc/nccl.conf` for system-wide config (upstream used to do this automatically but we removed it) |

---

## 5. Topology & Graph

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_TOPO_FILE` | Load XML topology file before auto-detection | `/var/run/nvidia-topologyd/virtualTopology.xml` if exists | 2.6 | `src/graph/topo.cc` | 🔧 **IB**: Set if auto-detect is wrong for your fabric |
| `NCCL_TOPO_DUMP_FILE` | Dump detected XML topology to file | — | 2.6 | `src/init.cc` | 🔧 Set for debugging topology issues |
| `NCCL_TOPO_DUMP_FILE_RANK` | Which rank dumps topology | `0` | — | `src/graph/topo.cc` | ⚙️ Set `0` to only dump from rank 0 |
| `NCCL_GRAPH_FILE` | Load communication graph from file | — | — | `src/graph/search.cc` | ⚙️ For custom graph injection |
| `NCCL_GRAPH_DUMP_FILE` | Dump searched communication graph to file | — | — | `src/graph/search.cc` | 🔧 For debugging graph search |
| `NCCL_GRAPH_DUMP_FILE_RANK` | Which rank dumps graph | `0` | — | `src/init.cc` | ⚙️ Set `0` |
| `NCCL_CROSS_NIC` | Cross-NIC ring/tree policy. 0=same NIC (rail-optimized), 1=allow cross, 2=prefer same | `2` | — | `src/graph/search.cc` | ⚙️ **IB rail-optimized**: `0`. **IB fat-tree**: `1`. Default `2` is safe |
| `NCCL_IGNORE_CPU_AFFINITY` | Ignore job CPU affinity, use GPU affinity only | `0` | 2.4.6 | `src/graph/topo.cc` | 🔧 Set `1` if CPU affinity causes suboptimal NIC selection |
| `NCCL_NET_FORCE_MERGE` | Force-merge network devices by name pattern | — | — | `src/graph/topo.cc` | ⚙️ Internal |
| `NCCL_NET_MERGE_LEVEL` | Network device merge level | — | — | `src/graph/topo.cc` | ⚙️ Internal |
| `NCCL_NETDEVS_POLICY` | Network device assignment: `AUTO`, `MAX:N`, `ALL` | `AUTO` | 2.28 | `src/graph/topo.cc` | 🔧 **IB**: Set `MAX:N` if AUTO causes sharing conflicts |
| `NCCL_NET_DISABLE_INTRA` | Disable intra-node net usage | `0` | — | `src/graph/paths.cc` | ⚙️ Internal |

---

## 6. Transport: P2P / SHM / Net

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_P2P_DISABLE` | Disable P2P (NVLink/PCIe direct GPU access) entirely | `0` | — | implied | 🚫 Only for debugging. Use `P2P_LEVEL` for fine control |
| `NCCL_P2P_LEVEL` | Max P2P distance: `LOC`, `NVL`, `PIX`, `PXB`, `PHB`, `SYS`. Legacy int: 0-4. | Auto per arch | 2.3.4 | `src/graph/paths.cc` | ⚙️ Leave auto. 🔧 Set `NVL` to restrict P2P to NVLink only |
| `NCCL_P2P_DIRECT_DISABLE` | Disable direct user buffer access through P2P across same-process GPUs | `0` | — | `src/transport/p2p.cc` | 🔧 Set `1` if buffers aren't P2P-accessible (custom allocators) |
| `NCCL_P2P_READ_ENABLE` | Enable P2P read mode | `-2` (auto) | — | `src/transport/p2p.cc` | ⚙️ Internal |
| `NCCL_P2P_USE_CUDA_MEMCPY` | Use CUDA memcpy for P2P | `0` | — | `src/transport/p2p.cc` | ⚙️ Internal |
| `NCCL_SHM_DISABLE` | Disable shared memory transport | `0` | — | `src/transport/shm.cc` | 🚫 Only for debugging |
| `NCCL_SHM_LOCALITY` | SHM locality control. 1=send-side, 2=recv-side | `2` (SHM_RECV_SIDE) | — | `src/transport/shm.cc` | ⚙️ Internal |
| `NCCL_SHM_MEMCPY_MODE` | SHM memory copy mode. 1=send-side, 2=recv-side | `1` (SHM_SEND_SIDE) | — | `src/transport/shm.cc` | ⚙️ Internal |
| `NCCL_SHM_USE_CUDA_MEMCPY` | Use CUDA memcpy for SHM | `0` | — | `src/transport/shm.cc` | ⚙️ Internal |
| `NCCL_NVB_DISABLE` | Disable NVLink-via-intermediate-GPU intra-node path | `0` | 2.11 | `src/graph/paths.cc` | 🚫 Only for debugging |
| `NCCL_PXN_DISABLE` | Disable ProXy Networking (NVLink + non-local NIC) | `0` | 2.12 | `src/graph/paths.cc` | 🚫 Only for debugging |
| `NCCL_PXN_C2C` | Allow PXN if peer GPU connected via C2C + PCIe to NIC | `1` (since 2.28; was 0 in 2.27) | 2.27 | `src/graph/paths.cc` | ⚙️ Leave default |
| `NCCL_P2P_PXN_LEVEL` | PXN for send/recv: 0=off, 1=when NIC not PCI-accessible, 2=always | `2` | 2.12 | `src/graph/search.cc` | ⚙️ Leave default |
| `NCCL_NET_SHARED_BUFFERS` | Use shared buffer pool for inter-node P2P (constant memory vs linear) | `1` | 2.8 | `src/transport/net.cc` | ⚙️ Leave enabled |
| `NCCL_NET_SHARED_COMMS` | Reuse connections in PXN context for message aggregation | `1` | 2.12 | `src/transport/net.cc` | ⚙️ Leave enabled |
| `NCCL_NET_OPTIONAL_RECV_COMPLETION` | Optional receive completion | `1` | — | `src/transport/net.cc` | ⚙️ Internal |
| `NCCL_NET_OVERHEAD` | Net overhead estimation for tuning | `-2` (auto) | — | `src/graph/tuning.cc` | ⚙️ Internal |
| `NCCL_NET_FORCE_FLUSH` | Force net flush | `0` | — | `src/graph/paths.cc` | ⚙️ Internal |
| `NCCL_NET_PLUGIN_REF_COUNT` | Net plugin reference counting | `0` | — | `src/plugin/net.cc` | ⚙️ Internal |
| `NCCL_RUNTIME_CONNECT` | Connect peers lazily during runtime instead of init | `1` | 2.22 | `src/init.cc` | ⚙️ Leave `1`. Set `0` for deterministic init-time connection |
| `NCCL_CONNECT_ROUND_MAX_PEERS` | Max peers per connection round | `128` | — | `src/transport.cc` | ⚙️ Internal |
| `NCCL_REPORT_CONNECT_PROGRESS` | Report connection progress during init | `0` | — | `src/transport.cc` | ⚙️ Internal, useful for debugging slow init |
| `NCCL_IGNORE_DISABLED_P2P` | Ignore disabled P2P settings | `0` | — | `src/graph/paths.cc` | ⚙️ Internal |
| `NCCL_P2P_PER_CHANNEL_NET_BW` | Per-channel net bandwidth for P2P paths (GB/s) | `14` | — | `src/graph/paths.cc` | ⚙️ Internal |
| `NCCL_MAX_P2P_NCHANNELS` | Max P2P channels | `64` (MAXCHANNELS) | — | `src/graph/paths.cc` | ⚙️ Internal |
| `NCCL_MIN_P2P_NCHANNELS` | Min P2P channels | `1` | — | `src/graph/paths.cc` | ⚙️ Internal |
| `NCCL_MULTI_SEGMENT_REGISTER` | Multi-segment registration for generic transport | `1` | — | `src/transport/generic.cc` | ⚙️ Internal |
| `NCCL_SINGLE_PROC_MEM_REG_ENABLE` | Single-process memory registration | `0` | — | `src/group.cc` | ⚙️ Internal |
| `NCCL_IPC_USE_ABSTRACT_SOCKET` | Use Linux Abstract Socket for IPC UDS handle exchange | `1` | 2.29 | `src/misc/ipcsocket.cc` | 🔧 Set `0` if containers have different network namespaces |
| `NCCL_P2P_EPOCH_ENABLE` | Enable P2P epoch mode | `1` | — | `src/enqueue.cc` | ⚙️ Internal |

---

## 7. GPU Direct RDMA (GDR)

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_NET_GDR_LEVEL` | Max GPU-NIC distance for GDR: `LOC`/`PIX`/`PXB`/`PHB`/`SYS`. Legacy int: 0-4. (formerly `NCCL_IB_GDR_LEVEL`) | Auto per arch | 2.3.4 | `src/graph/paths.cc` | ⚙️ Leave auto. **IB**: 🔧 Set `PHB` if GDR causes issues across NUMA |
| `NCCL_NET_GDR_READ` | Enable GPU Direct RDMA for send path (read from GPU) | `1` (NVLink), `0` (PCIe) since 2.4.2 | — | `src/graph/paths.cc` | ⚙️ Leave auto |
| `NCCL_NET_GDR_C2C` | Enable GDR via NIC on C2C-connected CPU (distance PHB) | `1` (since 2.27; was 0 in 2.26) | 2.26 | `src/graph/paths.cc` | ⚙️ Leave default |
| `NCCL_GDRCOPY_ENABLE` | Enable GDRCopy for small message optimization | `0` | — | `src/init.cc` | 🔧 **IB**: Enable if GDRCopy is installed |
| `NCCL_GDRCOPY_FIFO_ENABLE` | Enable GDRCopy for FIFO | `1` | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_GDRCOPY_FLUSH_ENABLE` | Enable GDRCopy flush | `0` | — | `src/transport/net.cc` | ⚙️ Internal |
| `NCCL_GDRCOPY_SYNC_ENABLE` | Enable GDRCopy sync | `1` | — | `src/transport/net.cc` | ⚙️ Internal |
| `NCCL_DMABUF_ENABLE` | Enable GPU Direct RDMA via Linux dma-buf | `1` (auto-disabled without support) | 2.13 | `src/init.cc` | ⚙️ Leave enabled |

---

## 8. NVLink / NVLS / MNNVL

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_NVLS_ENABLE` | NVLink SHARP offload. 0=off, 1=on (fail if can't allocate), 2=try | `2` | 2.17 | `src/transport/nvls.cc` | ⚙️ Leave `2`. Only on NVSwitch v3 (NVLink4) + Hopper+ |
| `NCCL_NVLS_NCHANNELS` | Number of NVLS channels | auto (UNDEF) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_NVLS_CHUNKSIZE` | NVLS chunk size | `131072` (128 KiB) | — | `src/transport/nvls.cc` | ⚙️ Internal |
| `NCCL_MNNVL_ENABLE` | Multi-Node NVLink. 0=off, 1=on (fail if unsupported), 2=auto | `2` | 2.21 | `src/init.cc` | ⚙️ Leave `2`. Requires IMEX domain + `CUMEM_ENABLE` |
| `NCCL_MNNVL_UUID` | Override MNNVL UUID (normally from Fabric Manager) | Auto | 2.25 | `src/init.cc` | ⚙️ Do not set unless debugging FM |
| `NCCL_MNNVL_CLIQUE_ID` | Override MNNVL Clique ID for soft partitioning | Auto | 2.25 | `src/init.cc` | 🔧 Set to partition MNNVL jobs |
| `NCCL_MNNVL_RAIL_PER_HOST` | Rails per host for MNNVL | `0` | — | `src/graph/search.cc` | ⚙️ Internal |
| `NCCL_MNNVL_SCATTER_NETS_ENABLE` | Scatter nets for MNNVL | `1` | — | `src/graph/search.cc` | ⚙️ Internal |

---

## 9. Algorithms, Protocols & Tuning

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_ALGO` | Allowed algorithms: Ring, Tree, CollnetChain, CollnetDirect, NVLS, NVLSTree, PAT. Semicolon-separated per-function rules (2.24+). `^` excludes. | All available (auto) | 2.5 | `src/enqueue.cc`, `src/graph/tuning.cc` | 🚫 Don't set in production. 🔧 Debug: `^Tree` to disable Tree |
| `NCCL_PROTO` | Allowed protocols: `LL`, `LL128`, `Simple`. `^` excludes. Per-function rules (2.24+). | All supported | 2.5 | `src/enqueue.cc`, `src/graph/tuning.cc` | 🚫 Don't set in production. 🔧 Debug: `^LL128` if data corruption suspected |
| `NCCL_BUFFSIZE` | Per-GPU-pair communication buffer size | `4194304` (4 MiB) | — | `src/init.cc` | 🚫 Only for memory-constrained or perf experiments |
| `NCCL_LL_BUFFSIZE` | Low-latency buffer size | `-2` (auto) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_LL128_BUFFSIZE` | LL128 buffer size | `-2` (auto) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_LL128_C2C` | LL128 for C2C paths | `1` | — | `src/graph/tuning.cc` | ⚙️ Internal |
| `NCCL_LL128_NTHREADS` | Thread count for LL128 | `-2` (auto) | — | `src/graph/tuning.cc` | ⚙️ Internal |
| `NCCL_NTHREADS` | CUDA threads per block per channel | `512` (newer GPUs), `256` (older) | — | `src/graph/tuning.cc` | 🚫 Only for debugging: 64, 128, 256, 512 |
| `NCCL_THREAD_THRESHOLDS` | Thread threshold configuration | — | — | `src/graph/tuning.cc` | ⚙️ Internal |
| `NCCL_MAX_NCHANNELS` | Max communication channels. Superseded by `NCCL_MAX_CTAS`. | `-2` (auto) | 2.5.0 | `src/graph/connect.cc` | 🚫 Use `NCCL_MAX_CTAS` instead |
| `NCCL_MIN_NCHANNELS` | Min communication channels. Superseded by `NCCL_MIN_CTAS`. | Platform-dependent | 2.5.0 | `src/graph/connect.cc` | 🚫 Use `NCCL_MIN_CTAS` instead |
| `NCCL_MAX_NRINGS` | Legacy alias for `MAX_NCHANNELS` | `-2` (auto) | 2.0.5 | `src/graph/connect.cc` | 🚫 Deprecated |
| `NCCL_MIN_NRINGS` | Legacy alias for `MIN_NCHANNELS` | `-2` (auto) | 2.2.0 | `src/graph/connect.cc` | 🚫 Deprecated |
| `NCCL_MAX_CTAS` | Max CTAs (CUDA Thread Block Arrays). Overrides `maxCTAs` config. | Auto | 2.17 | `src/init.cc` | 🔧 Set to limit GPU resource usage (1-64) |
| `NCCL_MIN_CTAS` | Min CTAs. Overrides `minCTAs` config. | Auto | 2.17 | `src/init.cc` | 🔧 Set to guarantee parallelism (1-64) |
| `NCCL_CTA_POLICY` | CTA policy: `DEFAULT`/`0`, `EFFICIENCY`/`1`, `ZERO`/`2`. Pipe with `\|`. | `DEFAULT` | 2.29 (legacy 2.27) | `src/init.cc` | ⚙️ Leave default |
| `NCCL_UNPACK_DOUBLE_NCHANNELS` | Double channels for unpacking | `1` | — | `src/graph/connect.cc` | ⚙️ Internal |
| `NCCL_NCHANNELS_PER_NET_PEER` | Channels per network peer | auto (UNDEF) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_P2P_NET_CHUNKSIZE` | Network chunk size for ncclSend/ncclRecv | `131072` (128K) | 2.14 | `src/init.cc` | ⚙️ Power of 2, next up: 262144 |
| `NCCL_P2P_NVL_CHUNKSIZE` | NVLink chunk size for P2P | `524288` (512 KiB) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_P2P_PCI_CHUNKSIZE` | PCI chunk size for P2P | `131072` (128 KiB) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_P2P_LL_THRESHOLD` | Max message size for LL protocol in P2P | `16384` | 2.14 | `src/enqueue.cc` | ⚙️ Leave default |
| `NCCL_ALLOC_P2P_NET_LL_BUFFERS` | Allocate dedicated LL buffers for all P2P net connections | `0` | 2.14 | `src/init.cc` | 🔧 Set `1` for latency-sensitive all-to-all at scale (high memory) |
| `NCCL_NVLSTREE_MAX_CHUNKSIZE` | Max chunk size for NVLSTree | `-2` (auto) | — | `src/enqueue.cc` | ⚙️ Internal |
| `NCCL_PAT_ENABLE` | Enable PAT algorithm. 0=off, 1=on, 2=auto | `2` (auto) | — | `src/graph/tuning.cc` | ⚙️ Internal |
| `NCCL_P2P_SCHEDULE_GROUP_SIZE` | P2P schedule group size | `8` (NCCL_MAX_DEV_WORK_P2P_PER_BATCH) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_CHUNK_SIZE` | Generic chunk size override | `0` (disabled) | — | `src/enqueue.cc` | ⚙️ Internal |
| `NCCL_SYM_CE_THRESHOLD` | Symmetric CE threshold | `8388608` (8 MiB) | — | `src/enqueue.cc` | ⚙️ Internal |
| `NCCL_ALLGATHERV_ENABLE` | Enable AllGatherV | `1` | — | `src/enqueue.cc` | ⚙️ Internal |
| `NCCL_SINGLE_RING_THRESHOLD` | Single-ring msg size limit (removed 2.3) | `262144` | 2.1 | — | 🚫 Removed |
| `NCCL_LL_THRESHOLD` | LL algorithm size limit (removed 2.5) | `16384` | 2.1 | — | 🚫 Removed |
| `NCCL_TREE_THRESHOLD` | Tree algorithm size limit (removed 2.5) | Rank-dependent | 2.4 | — | 🚫 Removed |

---

## 10. CUDA Memory & Registration

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_CUMEM_ENABLE` | Use CUDA cuMem* functions for device memory allocation | Auto-detect (since 2.19) | 2.18 | `src/misc/cudawrap.cc` | ⚙️ Leave auto. Required for MNNVL |
| `NCCL_CUMEM_HOST_ENABLE` | Use cuMem* for host memory | `1` if CUDA driver ≥ 12.6, runtime ≥ 12.2 (since 2.24) | 2.23 | `src/misc/cudawrap.cc` | ⚙️ Leave auto |
| `NCCL_GRAPH_REGISTER` | Auto-register buffers in CUDA Graphs (CollNet, P2P, single GPU/proc) | `1` | 2.11 | `src/enqueue.cc` | ⚙️ Leave enabled |
| `NCCL_LOCAL_REGISTER` | Enable explicit buffer registration via ncclCommRegister | `1` | 2.19 | `src/register/register.cc` | ⚙️ Leave enabled |
| `NCCL_LEGACY_CUDA_REGISTER` | Allow registration of legacy cudaMalloc buffers (can cause sync/hang!) | `0` | 2.24 | `src/transport/p2p.cc` | 🚫 Leave `0`. Use cuMem-based allocators instead |
| `NCCL_WIN_ENABLE` | Enable window memory registration | `1` | 2.27 | `src/init.cc` | ⚙️ Leave enabled |
| `NCCL_WIN_STRIDE` | Window stride | `-1` (auto) | — | `src/dev_runtime.cc` | ⚙️ Internal |
| `NCCL_MEM_SYNC_DOMAIN` | CUDA Memory Sync Domain for NCCL kernels (sm90+). 0=Default, 1=Remote | `1` (Remote) | 2.16 | `src/enqueue.cc` | ⚙️ Leave `1`. Eliminates compute/NCCL interference |

---

## 11. Kernel Launch & Threading

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_LAUNCH_MODE` | CUDA launch mode: `PARALLEL` or `GROUP` (deprecated) | `PARALLEL` | 2.1.0 | `src/init.cc` | ⚙️ Leave `PARALLEL`. GROUP deprecated in 2.9 |
| `NCCL_SET_STACK_SIZE` | Set CUDA kernel stack to max across NCCL kernels. May avoid memory reconfig hangs. | `0` | 2.9 | `src/init.cc` | 🔧 Set `1` if experiencing hangs from CUDA memory reconfiguration |
| `NCCL_SET_CPU_STACK_SIZE` | Set CPU stack size (for NCCL's CPU threads) | `1` | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_SET_THREAD_NAME` | Give meaningful names to NCCL CPU threads | `0` | 2.12 | `src/debug.cc` | 🔧 Set `1` for debugging/profiling (low overhead) |
| `NCCL_CGA_CLUSTER_SIZE` | CUDA CGA cluster size (sm90+). Overrides `cgaClusterSize` config. | Auto | 2.16 | `src/init.cc` | ⚙️ Leave auto. 0-8 valid |
| `NCCL_GRAPH_MIXING_SUPPORT` | Support multiple outstanding NCCL calls from parallel graphs | `1` | 2.13 | `src/init.cc` | 🔧 Set `0` if experiencing launch hangs with multi-graph usage |
| `NCCL_LAUNCH_ORDER_IMPLICIT` | Implicitly order NCCL ops from different comms on same device by host program order | `0` | 2.26 | `src/enqueue.cc` | 🔧 Set `1` to prevent deadlocks between communicators |
| `NCCL_LAUNCH_RACE_FATAL` | Catch host threads racing to launch on same device → fatal error | `1` | 2.26 | `src/misc/strongstream.cc` | ⚙️ Leave enabled |
| `NCCL_COMM_BLOCKING` | Force blocking/nonblocking communicators. 1=blocking, 0=nonblocking. | Undefined (per config) | 2.14 | `src/init.cc` | ⚙️ Leave undefined; set per-communicator via `ncclConfig_t` |
| `NCCL_CHECKS_DISABLE` | Disable argument checks on collective calls (deprecated 2.2.12) | `0` | 2.0.5 | — | 🚫 Deprecated |
| `NCCL_CHECK_POINTERS` | Enable CUDA memory pointer checking on collective calls | `0` | 2.2.12 | — | 🔧 Set `1` during development |
| `NCCL_L1_SHARED_MEMORY_CARVEOUT` | L1/shared memory carveout for NCCL kernels | `0` | — | `src/enqueue.cc` | ⚙️ Internal |
| `NCCL_GROUP_CUDA_STREAM` | Group CUDA stream setting | `0` (CUDA ≥9.2) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_GRAPH_HELPER_DISABLE` | Disable graph helper | `0` | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_WORK_FIFO_BYTES` | Work FIFO size | `1048576` (1 MiB) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_WORK_ARGS_BYTES` | Work args size | `INT64_MAX` (unlimited) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_ENABLE_VERSION_CHECK` | Enable device runtime version check | `1` | — | `src/dev_runtime.cc` | ⚙️ Internal |
| `NCCL_CREATE_THREAD_CONTEXT` | Create thread context | `0` | — | `src/proxy.cc` | ⚙️ Internal |
| `NCCL_PROXY_CPUSET` | Pin proxy thread to specific CPUs (comma-separated ranges) | — | — | `src/proxy.cc` | 🔧 Set for NUMA-aware proxy pinning |
| `NCCL_PROXY_APPEND_BATCH_SIZE` | Proxy append batch size | `16` | — | `src/proxy.cc` | ⚙️ Internal |
| `NCCL_PROXY_DUMP_SIGNAL` | Dump proxy state on signal. -1=off, otherwise signal number | `-1` (disabled) | — | `src/proxy.cc` | ⚙️ Internal debug |
| `NCCL_PROGRESS_APPENDOP_FREQ` | Progress appendop frequency | `8` | — | `src/proxy.cc` | ⚙️ Internal |
| `NCCL_NVTX_DISABLE` | Disable NVTX annotations | `0` | — | `src/init_nvtx.cc` | ⚙️ Set `1` if NVTX overhead is unwanted |
| `NCCL_NVB_PRECONNECT` | Preconnect NVB peers | `1` | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE` | NVLink utilization-centric scheduling | `0` (effective; UNDEF→`0`) | — | `src/init.cc` | ⚙️ Internal |

---

## 12. CollNet & Plugins

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_COLLNET_ENABLE` | Enable CollNet plugin (in-network reduction) | `0` | 2.6 | `src/init.cc` | 🔧 **IB with SHARP**: Set `1` |
| `NCCL_COLLNET_NODE_THRESHOLD` | Min nodes to enable CollNet | `2` | 2.9.9 | `src/init.cc` | ⚙️ Leave default |
| `NCCL_NET` | Force specific network transport: `IB`, `Socket`, or plugin name | Auto | 2.10 | `src/init.cc` | 🔧 Set `Socket` to force TCP. Otherwise leave auto |
| `NCCL_NET_PLUGIN` | Net plugin library name/suffix. "none" to disable external. | `libnccl-net.so` | 2.11 | `src/plugin/net.cc` | 🔧 Set plugin name or `none` |
| `NCCL_TUNER_PLUGIN` | Tuner plugin library name/suffix. Falls back to net plugin, then internal. | `libnccl-tuner.so` | — | `src/plugin/tuner.cc` | ⚙️ Leave default unless using custom tuner |
| `NCCL_PROFILER_PLUGIN` | Profiler plugin. `STATIC_PLUGIN` searches binary. | `libnccl-profiler.so` | — | `src/plugin/profiler.cc` | 🔧 Set for profiling |
| `NCCL_ENV_PLUGIN` | Environment plugin. "none" to disable. | `libnccl-env.so` | 2.28 | `src/plugin/env.cc` | ⚙️ Leave default |

---

## 13. GIN (GPU-Initiated Networking)

New in v2.29. CUDA device-side networking (GPU kernels directly initiate RDMA).

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_GIN_ENABLE` | Enable GPU-Initiated Networking | `1` | 2.29 | `src/gin/gin_host.cc` | ⚙️ Experimental. Requires DOCA GPUNetIO hardware |
| `NCCL_GIN_TYPE` | GIN backend type | `-1` (auto) | 2.29 | `src/gin/gin_host.cc` | ⚙️ Internal |
| `NCCL_GIN_NCONTEXTS` | Number of GIN contexts | `4` (NCCL_GIN_MAX_CONTEXTS) | 2.29 | `src/gin/gin_host.cc` | ⚙️ Internal |
| `NCCL_GIN_COUNTER_POOL_SIZE` | GIN counter pool size (entries) | `65536` (64K) | 2.29 | `src/gin/gin_host.cc` | ⚙️ Internal |
| `NCCL_GIN_SIGNAL_POOL_SIZE` | GIN signal pool size (entries) | `65536` (64K) | 2.29 | `src/gin/gin_host.cc` | ⚙️ Internal |
| `NCCL_GIN_PROXY_QUEUE_SIZE` | GIN proxy queue size | `-1` (auto) | 2.29 | `src/gin/gin_host_proxy.cc` | ⚙️ Internal |
| `NCCL_GIN_GDAKI_NIC_HANDLER` | GDAKI NIC handler config | `0` | 2.29 | `src/transport/net_ib/gdaki/gin_host_gdaki.cc` | ⚙️ Internal |
| `NCCL_GIN_GDAKI_QP_DEPTH` | GDAKI queue pair depth | `128` | 2.29 | `src/transport/net_ib/gdaki/gin_host_gdaki.cc` | ⚙️ Internal |
| `NCCL_GIN_ERROR_QUERY_SEC` | GIN error query interval (seconds) | `10` | 2.29 | `src/transport/net_ib/gdaki/gin_host_gdaki.cc` | ⚙️ Internal |

---

## 14. RAS (Reliability, Availability, Serviceability)

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_RAS_ENABLE` | Enable RAS subsystem for health querying | `1` | 2.24 | `src/bootstrap.cc` | ⚙️ Leave enabled. Use `ncclras` CLI to query |
| `NCCL_RAS_ADDR` | RAS listener socket address (host:port) | `localhost:28028` | 2.24 | `src/ras/client_support.cc` | 🔧 Set different ports for co-located NCCL jobs |
| `NCCL_RAS_TIMEOUT_FACTOR` | Multiplier for all RAS internal timeouts | `1` | 2.24 | `src/ras/ras.cc` | 🔧 Increase under heavy debugging/tracing overhead |

---

## 15. Miscellaneous Internal

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_SYM_KERNEL` | Symmetric kernel selection | — | — | `src/sym_kernels.cc` | ⚙️ Internal |
| `NCCL_SYM_CTAS` | Symmetric CTAs | `0` | — | `src/sym_kernels.cc` | ⚙️ Internal |
| `NCCL_SYM_NOWIN_ENABLE` | Symmetric scheduling without windows | `0` | — | `src/scheduler/symmetric_sched.cc` | ⚙️ Internal |
| `NCCL_COMM_SPLIT_SHARE_RESOURCES` | Share resources on comm split | `0` (effective; UNDEF→`0`) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_COMM_SHRINK_SHARE_RESOURCES` | Share resources on comm shrink | `0` (effective; UNDEF→`0`) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_NUM_RMA_CTX` | Number of RMA contexts | `1` (effective; UNDEF→`1`) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_RMA_PROXY_QUEUE_SIZE` | RMA proxy queue size | `-1` (auto) | — | `src/rma/rma_proxy.cc` | ⚙️ Internal |
| `NCCL_RMA_PROXY_DUMP_SIGNAL` | Dump RMA proxy state on signal | `-1` (disabled) | — | `src/rma/rma_proxy.cc` | ⚙️ Internal debug |

---

## 16. Profiler Plugin Env Vars (ext-profiler/)

These are read by the example profiler plugin, not by libnccl itself.

| Env Var | Description | Default | Source |
|---------|-------------|---------|--------|
| `NCCL_PROFILE_EVENT_MASK` | Bitmask of which event types to profile | — | `ext-profiler/` |
| `NCCL_PROFILE_DUMP_FILE` | File to dump profiling data | — | `ext-profiler/` |
| `NCCL_PROFILER_CE_TIMING` | Enable CE (Copy Engine) timing | — | `ext-profiler/` |
| `NCCL_PROFILER_CE_POLLER_INTERVAL_MICROSECONDS` | CE poller interval | — | `ext-profiler/` |
| `NCCL_PROFILE_COLL_API_POOL_SIZE` | Pool size for collective API events | — | `ext-profiler/` |
| `NCCL_PROFILE_COLL_POOL_SIZE` | Pool size for collective events | — | `ext-profiler/` |
| `NCCL_PROFILE_P2P_API_POOL_SIZE` | Pool size for P2P API events | — | `ext-profiler/` |
| `NCCL_PROFILE_P2P_POOL_SIZE` | Pool size for P2P events | — | `ext-profiler/` |
| `NCCL_PROFILE_GROUP_API_POOL_SIZE` | Pool size for group API events | — | `ext-profiler/` |
| `NCCL_PROFILE_GROUP_POOL_SIZE` | Pool size for group events | — | `ext-profiler/` |
| `NCCL_PROFILE_KERNEL_LAUNCH_POOL_SIZE` | Pool size for kernel launch events | — | `ext-profiler/` |
| `NCCL_PROFILE_PROXY_CTRL_POOL_SIZE` | Pool size for proxy control events | — | `ext-profiler/` |
| `NCCL_PROFILE_PROXY_DETACH_POOL_SIZE` | Pool size for proxy detach events | — | `ext-profiler/` |
| `NCCL_PROFILE_CE_BATCH_POOL_SIZE` | Pool size for CE batch events | — | `ext-profiler/` |
| `NCCL_PROFILE_CE_COLL_POOL_SIZE` | Pool size for CE collective events | — | `ext-profiler/` |
| `NCCL_PROFILE_CE_SYNC_POOL_SIZE` | Pool size for CE sync events | — | `ext-profiler/` |

---

## 17. Inspector Plugin Env Vars (ext-profiler/inspector/)

| Env Var | Description | Default | Source |
|---------|-------------|---------|--------|
| `NCCL_INSPECTOR_ENABLE` | Enable NCCL inspector | — | `ext-profiler/inspector/` |
| `NCCL_INSPECTOR_DUMP_DIR` | Directory for inspector dumps | — | `ext-profiler/inspector/` |
| `NCCL_INSPECTOR_DUMP_VERBOSE` | Verbose inspector dumps | — | `ext-profiler/inspector/` |
| `NCCL_INSPECTOR_DUMP_THREAD_ENABLE` | Enable background dump thread | — | `ext-profiler/inspector/` |
| `NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS` | Dump thread interval | — | `ext-profiler/inspector/` |
| `NCCL_INSPECTOR_PROM_DUMP` | Prometheus-format dump | — | `ext-profiler/inspector/` |

---

## 18. Tuner Plugin Env Vars (ext-tuner/)

| Env Var | Description | Default | Source |
|---------|-------------|---------|--------|
| `NCCL_TUNER_CONFIG_FILE` | Tuner configuration file path | — | `ext-tuner/` |

---

## 19. Non-NCCL Env Vars Referenced

| Env Var | Description | Where |
|---------|-------------|-------|
| `CUDA_LAUNCH_BLOCKING` | CUDA global synchronous launch (from CUDA runtime) | `src/misc/cudawrap.cc` |
| `CUDA_HOME` | CUDA installation path (build-time) | Build scripts |
| `SLURM_JOB_ID` / `SLURM_JOBID` | Slurm job identification | `src/` (for bootstrap) |
| `SLURM_JOB_NAME` | Slurm job name | `src/` |
| `DOCA_GPUNETIO_LOG` | DOCA GPUNetIO logging level | `src/transport/net_ib/gdaki/doca-gpunetio/` |
| `DOCA_GPUNETIO_DISABLE_GDRCOPY` | Disable GDRCopy in DOCA | `src/transport/net_ib/gdaki/doca-gpunetio/` |

---

## 20. Build-Only Env Vars

| Env Var | Description | Where |
|---------|-------------|-------|
| `NCCL_USE_CMAKE` | When set to `1`, generate.py skips rules.mk generation | `src/device/generate.py`, `src/device/symmetric/generate.py` |
| `NTHREADS` | Build parallelism for Makefile | `Makefile` |

---

## 21. Summary: Quick-Start Recommendations

### Single-Node Multi-GPU (Development/Testing)

```bash
# Minimal — just enable useful debug output
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_IB_DISABLE=1            # No IB needed on single node
export NCCL_SET_THREAD_NAME=1       # Helpful for profiling
```

### Multi-Node InfiniBand/RoCE (Production)

```bash
# Network
export NCCL_SOCKET_IFNAME==ib0      # Bootstrap interface
export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1  # Specific HCAs (adjust per node)
export NCCL_CROSS_NIC=0             # Rail-optimized fabric
export NCCL_IB_TIMEOUT=22           # Conservative for large fabric

# Debugging (prod)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_FILE=/tmp/nccl-%h-%p.log

# Performance
export NCCL_COLLNET_ENABLE=1        # If SHARP available
export NCCL_SET_THREAD_NAME=1
```

### Multi-Node TCP/IP Only (Cloud)

```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME==eth0     # Your data-plane interface
export NCCL_SOCKET_NTHREADS=4       # For 100G networks
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_NET=Socket              # Force socket transport

export NCCL_DEBUG=WARN
export NCCL_DEBUG_FILE=/tmp/nccl-%h-%p.log
```

### Debugging / Troubleshooting

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TOPO_DUMP_FILE=/tmp/nccl-topo.xml
export NCCL_GRAPH_DUMP_FILE=/tmp/nccl-graph.txt
export NCCL_SET_THREAD_NAME=1
export NCCL_CHECK_POINTERS=1        # Dev only, adds latency
```

---

## Appendix: About `NCCL_IB_DISABLE`

> **Fun fact observed by the user**: `NCCL_IB_DISABLE` appears in the NVIDIA docs but seems invisible in the codebase!
>
> **Explanation**: It IS in the code at `src/transport/net_ib/init.cc`:
> ```c
> NCCL_PARAM(IbDisable, "IB_DISABLE", 0);
> ```
> The `NCCL_PARAM` macro constructs the env var name by prepending `NCCL_` to the second argument (`"IB_DISABLE"`),
> yielding `NCCL_IB_DISABLE`. So a grep for the literal string `NCCL_IB_DISABLE` finds nothing — you have to grep for
> `"IB_DISABLE"` or `IbDisable` instead. This is a common gotcha when searching the NCCL codebase.
>
> To find ALL env vars, grep for `NCCL_PARAM(` and `ncclGetEnv(` instead of the full env var names.

---

*Generated: 2026-02-24 | NCCL v2.29.3-1 | eugo-inc/nccl-cmake fork*


# Test either NCCL correctly sees NVLink, NVSwitch, and (in future) EFA or the friends
# Search for the following in the `nccl` sources:
# 1. `nvlink`
#    1. @Important: filter out NVLink **LINKER** matches, as it's irrelvant to the current check
# 2. `nvl` (-> NVLink)
#    1. `nvls` (-> NVLink NVSwitch)
# 3. `nvs`
# 4. `PATH_*` like:
#    1. `PATH_NVB`
#    2. `PATH_NVL`
#
# 1. We're not building this plugin, but we may still some of its code to test that our NCCL works and works properly: https://github.com/eugo-inc/nccl-cmake/ext-tuner/example/test/test_plugin.c#L760-L785
# 2. `CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING` - check in CUDA driver?
# 3. We should see the related output w/ verbose logging turned on: https://github.com/eugo-inc/nccl-cmake/src/init.cc#L803
#   1. `MNNVL_ENABLE`
# 4. `@/src/graph/paths.cc`:
#    1. `ncclTopoPathAllNVLink` -> 1 in Grace and GB10.
#       1. Check whether the system is all GPUs directly or indirectly connected to each other through NVLink and C2C.
#    2. `ncclTopoPathAllDirectNVLink` -> 1 in Grace and GB10.
#       1. Check whether the system is all GPUs connected directly to each other through NVLink/NVSwitch.
#    3. `ncclTopoSplitNvLink` -> 0 in Grace and GB10.
#       1. Check whether we are in a split NVLink situation, with two NVLink domains, not connected through NVLink (e.g. QPI).
# 5. `@/src/graph/search.cc#L818`
#    1. @TODO+: It's where NCCL parses topology file. We need to embed it into our container and AMI for Grace-Blackwell instances (it should be downloadable from NVIDIA or even be installed w/ drivers). GB10 has it's pre-installed on the host side, #    but we'll still need to put that into container to make it visible to the userspace NCCL installation.
#        1. `@/src/graph/xml.cc#L403-406` - autodetection partially exists but because we use not the dynamic DIY build but NVIDIA-prebuilt solutions, I believe it's still safer to use pre-made topology files both on Grace and GB10.
#    2. Looks like, they hardcoded a lot of things per GPU architecture. This means, that we may get lesser performance even on GB300 (sm_103a) compared to GB200 (sm_100a). This is not a big deal for GB10 (sm_121a) as we primarly test NVLink C2C within 1 node and across 2 nodes it relies on ConnectX which we couldn't test w/o `libfabric` and `libibverbs` anyway.
# ```cpp
#  if (system->inter == 0) {
#    nspeeds = ccMin >= 100 ? NSPEEDSINTRA_SM100 : (ccMin >= 90 ? NSPEEDSINTRA_SM90 : NSPEEDSINTRA);
#    speedArray = ccMin >= 100 ? sm100SpeedArrayIntra : (ccMin >= 90 ? sm90SpeedArrayIntra : speedArrayIntra);
#  } else {
#    nspeeds = ccMin >= 100 ? NSPEEDSINTER_SM100 : (ccMin >= 90 ? NSPEEDSINTER_SM90 : NSPEEDSINTER);
#    speedArray = ccMin >= 100 ? sm100SpeedArrayInter : (ccMin >= 90 ? sm90SpeedArrayInter : speedArrayInter);
#  }
#  ```
# 6. `@/src/graph/topo.cc`
# ```cpp
# enum ncclNvLinkDeviceType {
#  ncclNvLinkDeviceUnknown,
#  ncclNvLinkDeviceGpu,
#  ncclNvLinkDeviceSwitch,
#  ncclNvLinkDeviceBridge, // IBM/Power NVLink bridge (Device 04ea)
# };
# ```
# 7. `@/src/graph/topo.h` - a lot of NVLink and friends-related things
#    1. `#define NET_BW 12.0           // 100Gbit` -> BW - Bandwith measured in Gigabytes (100 Gbit/s ~= 12.5 GB/s ~= 12 GB/s - probably w/ overhead subtracted)
# 8. `@/src/graph/tuning.cc` - this file hardcodes latency measurements and other amazing stuff.
#    1. Applies
# `improvements per architecture - the smaller the value, the better as `latency = base_latency * nvlsEfficiency`. The only exception is `0.0`, which doesn't mean infinite performance and 0 latency but rather the absence of `NVLink NVSwitch` support in these architectures, but only the usual `NVLink`.``e
# static const float nvlsEfficiency[NCCL_NUM_COMPCAPS] = {
#  0.0f, // Volta
#  0.0f, // Ampere
#  0.85f, // Hopper
#  0.74f, // Blackwell
# };
# ```
# 9. What we'll see in logs, is something like that:
#    1. ```
#       NCCL INFO NET/IB : GPU/0 1.8TB/s NVLS(SHARP) Tree
#       NCCL INFO Bootstrap : Using [0]eth0:10.0.0.1<0>
#       NCCL INFO NET/Socket : Using [0]eth0:10.0.0.1<0>
#       NCCL INFO NET/NVLS : Using NVLink/SHARP Tree
#       ```
# 10. `nvidia-smi` commands:
#     1. `nvidia-smi nvlink -s` NVLink support test
#     2. `nvidia-smi nvswitch -q sharp` SHARP support test
#     3. `nvidia-smi topo -m` - GPUs topology test (should work on g5g as well)
# 11. `NCCL_NVML_DIRECT` in `@/src/include/nvmlwrap.h`
#     1. @TODO: check if we need to pass this stuff to no rely on copy-pasted variant
#     2. Check how Ray, Torch, and other guys link NCCL - directly or over `dlopen()`
# 12. Partial libfabric tree
# ```bash -> libfabric-devel
# /usr/include/rdma
# /usr/include/rdma/fabric.h
# /usr/include/rdma/fi_atomic.h
# /usr/include/rdma/fi_cm.h
# /usr/include/rdma/fi_collective.h
# /usr/include/rdma/fi_domain.h
# /usr/include/rdma/fi_endpoint.h
# /usr/include/rdma/fi_eq.h
# /usr/include/rdma/fi_errno.h
# /usr/include/rdma/fi_ext.h
# /usr/include/rdma/fi_ext_usnic.h
# /usr/include/rdma/fi_rma.h
# /usr/include/rdma/fi_tagged.h
# /usr/include/rdma/fi_trigger.h
# ```
#
# ```bash -> rdma-core-devel
# usr/include/infiniband
# /usr/include/infiniband/acm.h
# /usr/include/infiniband/acm_prov.h
# /usr/include/infiniband/arch.h
# /usr/include/infiniband/efadv.h
# /usr/include/infiniband/ib.h
# /usr/include/infiniband/ib_user_ioctl_verbs.h
# /usr/include/infiniband/ibnetdisc.h
# /usr/include/infiniband/ibnetdisc_osd.h
# /usr/include/infiniband/mad.h
# /usr/include/infiniband/mad_osd.h
# /usr/include/infiniband/mlx4dv.h
# /usr/include/infiniband/mlx5_api.h
# /usr/include/infiniband/mlx5_user_ioctl_verbs.h
# /usr/include/infiniband/mlx5dv.h
# /usr/include/infiniband/opcode.h
# /usr/include/infiniband/sa-kern-abi.h
# /usr/include/infiniband/sa.h
# /usr/include/infiniband/tm_types.h
# /usr/include/infiniband/umad.h
# /usr/include/infiniband/umad_cm.h
# /usr/include/infiniband/umad_sa.h
# /usr/include/infiniband/umad_sa_mcm.h
# /usr/include/infiniband/umad_sm.h
# /usr/include/infiniband/umad_str.h
# /usr/include/infiniband/umad_types.h
# /usr/include/infiniband/verbs.h
# /usr/include/infiniband/verbs_api.h
# /usr/include/rdma
# /usr/include/rdma/rdma_cma.h
# /usr/include/rdma/rdma_cma_abi.h
# /usr/include/rdma/rdma_verbs.h
# /usr/include/rdma/rsocket.h
```
# 13. RDMA and IBVERBS seem to only be used for GDR and only if it relies on fabrics instead of PCIe
`@/src/transport/net.cc`
```cpp
INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [receive] via NET/%s/%d%s%s%s", channelId, connIndex, peerInfo->rank, peerInfo->nvmlDev, myInfo->rank, myInfo->nvmlDev, comm->ncclNet->name, req.netDev,
   req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "",
   req.shared ? "/Shared" : "");
```
>>> Conclusion `@/src/transport/` includes different transports used by NCCL. `net_ib` seems to be the most complex one but it's only used for GPUDirect and other transports like `NVLink` and all its hundreds variants don't use it directly. Instead, they rely on calls to the `cu*` driver which handles all the heavy lifting. This means, that NCCL works without `libfabric` and `libibverbs` and not only doesn't lose the performance but actually works even better as bypassing such complicated code paths. In future, we may need to add `libfabric` for cross-rack communications, but even in this case, it will be better for us to orchestrate this part of workloads via `ray` directly to not introduce multiple conflicting control / orchestration planes.


```bash
# MARK: - Creating the configuration file
EUGO_NCCL_CONF_PATH="${EUGO_INSTALL_PREFIX_PATH}/share/gdal"

# MARK: - Selecting values
# 1 -> Grace and GB10, 0 -> Others # @TODO: implement that to be dynamic
if [[ -v EUGO_NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE ]]; then
    NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE="${EUGO_NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE}"
else
    NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE=0
fi


# MARK: - Setting values
EUGO_NCCL_CONF_PATH="${EUGO_INSTALL_PREFIX_PATH}/share/gdal"

cat << EOF > "$(eugo_create_package_configuration_script_path "${EUGO_PACKAGE_NAME}" "${EUGO_PACKAGE_KIND}" "runtime")"
# Required to maximize GDAL efficiency by forcing it to use all available CPU cores, wherever it's implemented
# Not everywhere it would improve performance, cause most parts of it are single-threaded
export GDAL_NUM_THREADS="ALL_CPUS"

# Allows reading/writing GeoJSONs of unlimited size
export OGR_GEOJSON_MAX_OBJ_SIZE="0"

# @TODO+:extract automatically. probably via gdal_config? meh.
export GDAL_DATA="${EUGO_GDAL_DATA_PATH}"
export OSM_CONFIG_FILE="${EUGO_GDAL_DATA_PATH}/osmconf.ini"
EOF

```


# @TODO: move to slava's docs:
#

