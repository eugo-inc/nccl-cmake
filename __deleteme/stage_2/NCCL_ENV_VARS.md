# NCCL Environment Variables — Complete Reference

> **Generated from**: NVIDIA docs (v2.29) + exhaustive codebase scan of eugo-inc/nccl-cmake (v2.29.3-1), verified by source grep 2026-04-22.
>
> **Authoritative count**: **222 unique runtime `NCCL_*` env vars** in source (161 `NCCL_PARAM` + 28 `ncclGetEnv` string literals + 26 raw `getenv` + 6 `getEnvFunc` + 3 via `ncclGetLevel` helper). The doc also documents ~8 deprecated/removed vars as anti-guidance, and ~5 doc entries that look like env vars but aren't (`NCCL_PARAM`, `NCCL_CONFIG_DEFAULT`, `NCCL_MAGIC`, `NCCL_SUFFIX`, `NCCL_ALGO_COLLNET_*` — macros/constants/algorithm names).
>
> **Per-scenario verdicts**: the per-row `Recommendation` column below uses the legacy **SN/IB/TCP** triad. For the three production scenarios we actually target (`grace_gb10`, `g5g`, `aws_efa`), use:
> - **[__deleteme/NCCL_ENV_VARS_MATRIX.tsv](NCCL_ENV_VARS_MATRIX.tsv)** — flat matrix, one row per var, columns `{var, sec, grace_gb10, g5g, aws_efa, notes}`. Source of truth.
> - **[__deleteme/NCCL_ENV_VARS_SHORTLIST.md](NCCL_ENV_VARS_SHORTLIST.md)** — must-set / must-not-set / careful, grouped by scenario.
> - **[__deleteme/nccl.conf.grace_gb10](nccl.conf.grace_gb10)**, **[nccl.conf.g5g](nccl.conf.g5g)**, **[nccl.conf.aws_efa](nccl.conf.aws_efa)** — shippable conf files, minimal overrides only.
>
> **How NCCL reads env vars**: Most use the `NCCL_PARAM(Name, "SUFFIX", default)` macro which reads `NCCL_SUFFIX` from environment (macro at [src/include/param.h:20](../src/include/param.h#L20); load in `ncclLoadParam()` at [src/misc/param.cc:135](../src/misc/param.cc#L135)). String-valued vars use `ncclGetEnv("NCCL_FOO")`. A few use raw `getenv()` or the `getEnvFunc` wrapper in [src/debug.cc](../src/debug.cc).
>
> **Conf file loading — `initEnvFunc` / `setEnvFile` ([src/misc/param.cc](../src/misc/param.cc))**: loaded once at startup via `std::call_once`. **Load order (first-match-wins, EUGO patched)**: `$NCCL_CONF_FILE` (if set, non-empty, file exists) → `~/.nccl.conf` → `/usr/local/etc/nccl.conf`. Format: `KEY=VALUE` per line, `#` comments at line start, NO quoting, NO whitespace trimming, 1023-byte key/value truncation, malformed lines silently skipped.
>
> **Shell env vars always win**: `ncclOsSetEnv` calls `setenv(name, value, 0)` ([src/os/linux.cc:~70](../src/os/linux.cc)). The conf file is a **floor**, not a ceiling — any `NCCL_FOO` already exported in the shell is preserved, and conf values apply only for unset names.
>
> **Known upstream bugs fixed by eugo (`@EUGO_CHANGE` in [src/misc/param.cc](../src/misc/param.cc))**:
> 1. **Unconditional `/etc/nccl.conf` override** — upstream always called `setEnvFile("/etc/nccl.conf")` after the user file, silently overriding any values already loaded from `$NCCL_CONF_FILE` or `~/.nccl.conf`. Fixed: first-match-wins, system default moved to `/usr/local/etc/nccl.conf`.
> 2. **1024-byte path truncation** — `confFilePath` is a fixed `char[1024]`; paths longer than 1023 bytes are silently truncated (upstream limitation, not fixed).
>
> **Doc errata resolved**:
> - `NCCL_NVML_DIRECT` — this is a **compile-time `#define`** in [src/include/nvmlwrap.h:12-14](../src/include/nvmlwrap.h#L12), NOT a runtime env var. References to it elsewhere in this file (appendix prose) should not be interpreted as something settable at deployment.
> - `NCCL_IB_GDR_LEVEL` — claimed as legacy alias in §7, but no read of this name exists in current source. Treat as removed.
> - Outstanding @TODO: **topology file embedding for Grace-Blackwell AMIs/containers** ([src/graph/search.cc](../src/graph/search.cc)). Deployment task, not a doc task — see conf file `NCCL_TOPO_FILE` template.

# Official Documentation:
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
```bash
NCCL_CONF_FILE
(since 2.23)

The NCCL_CONF_FILE variable allows the user to specify a file with the static configuration. This does not accept the ~ character as part of the path; please convert to a relative or absolute path first.

Values accepted
If unset or if the version is prior to 2.23, NCCL uses .nccl.conf in the home directory if available.
```

---

## Eugo Legend

- **(+)** — we set this variable in our conf files.
- **(-)** — we explicitly avoid setting this variable (dangerous, deprecated, or redundant).
- **(~)** — 50/50; worth tracking but not setting right now. Default is fine, or data is insufficient, or the cost/benefit is unclear. May become (+) or (-) later.
- Optional comment form: **(+, comment)**, **(-, comment)**, **(~, comment)** — inline justification.

Markers apply at **section level** (in the Table of Contents) and at **per-variable level** (inside each section, next to the variable name). A section marked (-) means *"by default, nothing in this section applies to our deployments; individual (+) entries are exceptions"*. A section marked (+) means *"most entries here are actively tuned; individual (-) entries are exceptions"*.

---

## Notation Used Per Variable (v2 style, matches Ray config doc)

For variables in restyled sections (§1, §4 as of 2026-04-22), each entry uses this block format:

```
(+/-/~[, comment]) NCCL_VAR_NAME
1. Default value: `X`
2. Type: int | float | bool | string | size_t
3. Read / Write: R (user→NCCL) | R/W (per-comm config) | — (compile-time)
4. Recommended value (per scenario): **grace_gb10: X | g5g: Y | aws_efa: Z**
5. Mechanism: **Process: ...** (host-side init, proxy thread, device kernel, etc.)
6. Notes: source file citation + what the var actually does + *why the recommended value* + trade-offs + interactions. This is the load-bearing field — prose, not a one-liner.
7. Eugo conclusion: `(+/-/~)` + one-paragraph justification for the decision.
```

Sections that still use the dense table (older style) will be migrated in subsequent passes. See per-section markers in the Table of Contents.

---

## Legacy Deployment Scenario Key (v1 table rows)

The per-row `Recommendation` column in sections NOT yet restyled uses this legacy triad — kept for transitional reading:

| Symbol | Scenario |
|--------|----------|
| **SN** | Single-node multi-GPU (dev/test, no IB) |
| **IB** | Multi-node with InfiniBand/RoCE (production HPC) |
| **TCP** | Multi-node with TCP/IP only (cloud VMs without RDMA) |
| **ALL** | Applies to all scenarios |

**Legacy recommendation symbols**: ✅ = recommended to set, 🔧 = set if needed, ⚙️ = leave default, 🚫 = do not set in production.

For our actual scenarios (`grace_gb10`, `g5g`, `aws_efa`), see [NCCL_ENV_VARS_MATRIX.tsv](NCCL_ENV_VARS_MATRIX.tsv) and [NCCL_ENV_VARS_SHORTLIST.md](NCCL_ENV_VARS_SHORTLIST.md).

---

## Table of Contents

1. **(+)** [Network & Socket Configuration](#1-network--socket-configuration) — bootstrap interface, socket tuning. Restyled to v2 format.
2. **(-, EFA doesn't speak RC+RDMA_WRITE; not used on any of our 3 scenarios)** [InfiniBand / RoCE Configuration](#2-infiniband--roce-configuration)
3. **(~, tracked; only UID stagger matters at 72-GPU scale)** [Out-of-Band & Bootstrap](#3-out-of-band--bootstrap)
4. **(+)** [Debug & Logging](#4-debug--logging) — `NCCL_DEBUG=WARN` baseline everywhere. Restyled to v2 format.
5. **(+, trust auto-detection on p6e; see §5.0)** [Topology & Graph](#5-topology--graph)
6. **(~, defaults good; few individual overrides for grace_gb10 + aws_efa)** [Transport: P2P / SHM / Net](#6-transport-p2p--shm--net)
7. **(+, DMABUF explicit on aws_efa)** [GPU Direct RDMA (GDR)](#7-gpu-direct-rdma-gdr)
8. **(+, MNNVL strict on aws_efa for NVL72; NVLS auto)** [NVLink / NVLS / MNNVL](#8-nvlink--nvls--mnnvl)
9. **(~, tuning model auto-picks; don't restrict ALGO/PROTO in production)** [Algorithms, Protocols & Tuning](#9-algorithms-protocols--tuning)
10. **(+, CUMEM_ENABLE=1 pinned on aws_efa for MNNVL)** [CUDA Memory & Registration](#10-cuda-memory--registration)
11. **(+, LAUNCH_ORDER_IMPLICIT=1 + MEM_SYNC_DOMAIN pinned on aws_efa)** [Kernel Launch & Threading](#11-kernel-launch--threading)
12. **(-, no SHARP on AWS; no CollNet plugin shipped)** [CollNet & Plugins](#12-collnet--plugins)
13. **(-, GIN_ENABLE=0 on aws_efa per #1913; no DPUs anywhere)** [GIN (GPU-Initiated Networking)](#13-gin-gpu-initiated-networking)
14. **(~, RAS default=1 is fine; disable only at extreme scale)** [RAS (Reliability, Availability, Serviceability)](#14-ras-reliability-availability-serviceability)
15. **(-, all internal; no scenario signal)** [Miscellaneous Internal](#15-miscellaneous-internal)
16. **(-, we don't build/install ext-*/example plugins; see CMakeLists install block)** [Profiler Plugin Env Vars (ext-profiler/)](#16-profiler-plugin-env-vars-ext-profiler)
17. **(-, same as §16)** [Inspector Plugin Env Vars (ext-profiler/inspector/)](#17-inspector-plugin-env-vars-ext-profilerinspector)
18. **(-, same as §16)** [Tuner Plugin Env Vars (ext-tuner/)](#18-tuner-plugin-env-vars-ext-tuner)
19. **(~, FI_* and OFI_NCCL_* go in launcher env, not nccl.conf)** [Non-NCCL Env Vars Referenced](#19-non-nccl-env-vars-referenced)
20. **(~, NCCL_USE_CMAKE handled in CMakeLists; nothing else runtime)** [Build-Only Env Vars](#20-build-only-env-vars)
21. [Summary: Quick-Start Recommendations](#21-summary-quick-start-recommendations)
22. [Interactions & Shadowing (edge-list)](#22-interactions--shadowing)

---

## 1. Network & Socket Configuration

> Section marker: **(+)** — bootstrap interface selection is critical on every scenario; most socket tuning defaults are correct for AWS ENA but we pin a couple explicitly.

These control TCP/IP socket behavior for bootstrap and for the Socket transport (when IB/plugin isn't used).

---

(+) **NCCL_SOCKET_IFNAME**
1. Default value: auto-detected (priority cascade — see Notes)
2. Type: string (comma-separated list of interface prefixes, with optional `^` exclusion or `=` exact-match prefix)
3. Read / Write: R
4. Recommended value (per scenario): **grace_gb10: TEMPLATE (`=ibp1s0` or `^docker0,^lo`) | g5g: TEMPLATE (`=eth0`) | aws_efa: TEMPLATE (`=ens5`)**
5. Mechanism: **Process: host-side init (bootstrap ring, TCP transport setup).** Read via `ncclGetEnv()` in `ncclFindInterfaces()`.
6. Notes: Parsed by `parseStringList()` in [src/misc/utils.cc](../src/misc/utils.cc); matched by `matchIfList()` with `strncmp()` for prefix or full-string compare for exact. **When unset**, a priority cascade tries: (1) `ib*`, (2) interfaces on the same subnet as `NCCL_COMM_ID`, (3) anything except `docker*,lo,virbr*`, (4) `docker*`, (5) `lo`, (6) `virbr*`. This cascade is fine for a predictable host but dangerous inside containers with multiple virtual interfaces — you can silently pick up a docker bridge instead of the real data-plane NIC. Wrong selection means either connection failures at bootstrap or using a slow control-plane network instead of the high-speed data-plane (EFA on p6e has separate interfaces; the ENA is the one to pick for bootstrap). Interacts with `NCCL_SOCKET_FAMILY` (AF_INET vs AF_INET6 filter) and `NCCL_COMM_ID` (used as subnet hint when unset).
7. Eugo conclusion:
   1. (+) on all three scenarios; value is deployment-specific so conf file ships as a commented template. Use `=` prefix for EXACT match — never leave this to auto-cascade inside a container. p6e: use `=ens5` or the ENA's actual name.

(~) **NCCL_SOCKET_FAMILY**
1. Default value: `AF_UNSPEC` (auto — tries both families)
2. Type: string (`AF_INET` | `AF_INET6`)
3. Read / Write: R
4. Recommended value (per scenario): **default (auto) everywhere**
5. Mechanism: **Process: host-side init.** Checked during `ncclFindInterfaces()`.
6. Notes: Forces IPv4 or IPv6 when selecting interfaces. AWS VPC is IPv4-default; IPv6 optional. Not a performance knob — purely a correctness/selection knob for mixed-stack hosts.
7. Eugo conclusion:
   1. (~) not setting; AWS deployments are IPv4 and the default auto-probe finds the right family. Revisit only if we ever add an IPv6-only fleet.

(~) **NCCL_SOCKET_RETRY_CNT**
1. Default value: `34`
2. Type: int
3. Read / Write: R
4. Recommended value (per scenario): **default (34) everywhere**
5. Mechanism: **Process: host-side init (bootstrap ring connection).** Checked in `socketConnectCheck()` ([src/os/linux.cc](../src/os/linux.cc)).
6. Notes: Max retries after `connect()` fails with `ETIMEDOUT`, `ECONNREFUSED`, or `EHOSTUNREACH`. **Backoff is linear**, not exponential: retry *i* sleeps `i × RETRY_SLEEP_MSEC` ms. Total wait = `SLEEP × N(N+1)/2`. With defaults (N=34, S=100 ms): `100 × 34×35/2 = 59,500 ms ≈ 60 s` total retry window. This is the bootstrap stall you observe when one rank starts before another — 60 s is usually enough for SLURM / k8s to converge launches; if your launcher is slower, raise this before raising the timeouts downstream.
7. Eugo conclusion:
   1. (~) default is fine for Slurm/k8s-orchestrated launches where rank start is well-correlated. If we ever move to a pull-based ray launcher with unpredictable rank startup, raise to 60+ to give ~2 min bootstrap window.

(~) **NCCL_SOCKET_RETRY_SLEEP_MSEC**
1. Default value: `100`
2. Type: int (milliseconds)
3. Read / Write: R
4. Recommended value (per scenario): **default (100) everywhere**
5. Mechanism: **Process: host-side init.** Read in `socketConnectCheck()` via `ncclOsSleep(errorRetries * ncclParamRetryTimeOut())`.
6. Notes: Initial per-retry sleep multiplier. Scales linearly with retry index (retry *i* sleeps `i × this_value` ms). Doubling this doubles the total bootstrap retry window.
7. Eugo conclusion:
   1. (~) default is fine. Pair with `SOCKET_RETRY_CNT` when tuning bootstrap window.

(~) **NCCL_SOCKET_POLL_TIMEOUT_MSEC**
1. Default value: `0` (tight-loop — 100% CPU during wait)
2. Type: int (milliseconds; `0` = tight-loop poll)
3. Read / Write: R
4. Recommended value (per scenario): **default (0) on grace_gb10 / aws_efa | consider `100` on g5g**
5. Mechanism: **Process: host-side progress thread.** Poll timeout in `socketWait()`.
6. Notes: When `0` (default), `socketWait()` tight-loops non-blocking `send()`/`recv()` — burns 100% of one CPU core during bootstrap and proxy ops. When non-zero, `poll(fd, 1, timeout_ms)` is called between failed progress attempts, yielding the CPU until the kernel signals data. Adds up to `timeout_ms` latency per socket wait but dramatically reduces CPU burn. On GPU nodes we have spare CPU cores — tight-loop is the right call. On CPU-constrained nodes (g5g has only 64 vCPUs on the largest variant), setting `100` prevents NCCL from starving the actual compute workload.
7. Eugo conclusion:
   1. (~) not setting for now. g5g is dev/test only and CPU pressure hasn't bitten us. If it does, set `100` on g5g; leave tight-loop on grace_gb10 + aws_efa.

(~) **NCCL_SOCKET_NTHREADS**
1. Default value: `-2` (auto — depends on NIC vendor)
2. Type: int
3. Read / Write: R
4. Recommended value (per scenario): **default everywhere (auto picks AWS ENA → 2)**
5. Mechanism: **Process: host-side socket transport.** Read in `ncclNetSocketGetNsockNthread()`.
6. Notes: CPU helper threads per socket transport connection. Auto-detection reads `/sys/class/net/<dev>/device/vendor`: AWS ENA (`0x1d0f`) → 2 threads; GCP gVNIC (`0x1ae0`) → 4; other → 0 (main thread only). Max `MAX_THREADS=16`. When >0, NCCL spawns helper threads, each polling `NSOCKS_PERTHREAD` parallel TCP sockets. Product `nThreads × nSocksPerThread` clamped to `MAX_SOCKETS=64`. Only relevant on the Socket transport path — on `aws_efa` this is entirely bypassed (plugin handles its own threads). On `g5g` the default (2 threads × 8 socks = 16 parallel sockets per connection) is what NVIDIA tested for AWS ENA at 25 Gbps.
7. Eugo conclusion:
   1. (~) default everywhere. The auto-detect logic already knows AWS and GCP NIC vendors; overriding can only break the happy path.

(~) **NCCL_NSOCKS_PERTHREAD**
1. Default value: `-2` (auto — depends on NIC vendor; AWS ENA → 8)
2. Type: int
3. Read / Write: R
4. Recommended value (per scenario): **default everywhere**
5. Mechanism: **Process: host-side socket transport.** Paired with `NCCL_SOCKET_NTHREADS`.
6. Notes: Sockets per helper thread. Auto-detect: AWS ENA → 8; GCP gVNIC → 1; other → 1. Product with `SOCKET_NTHREADS` clamped to 64. If only one of the two is set, the other still auto-detects.
7. Eugo conclusion:
   1. (~) same reasoning as `SOCKET_NTHREADS`. Leave auto.

(-) **NCCL_SOCKET_RCVBUF**
1. Default value: OS default (`net.core.rmem_default`)
2. Type: int (bytes)
3. Read / Write: R
4. Recommended value (per scenario): **do not set on any scenario**
5. Mechanism: **Process: host-side socket transport.** Sets `SO_RCVBUF` on accepted sockets.
6. Notes: Overriding this typically forces `setsockopt(SO_RCVBUF)` which on Linux also disables kernel auto-tuning (`tcp_rmem` autoscaling). Almost always makes things slower unless paired with careful system-level tuning of `/proc/sys/net/core/*`.
7. Eugo conclusion:
   1. (-) do not set; OS default + kernel auto-tuning is strictly better for our workloads.

(-) **NCCL_SOCKET_SNDBUF**
1. Default value: OS default (`net.core.wmem_default`)
2. Type: int (bytes)
3. Read / Write: R
4. Recommended value (per scenario): **do not set on any scenario**
5. Mechanism: **Process: host-side socket transport.** Sets `SO_SNDBUF` on created sockets.
6. Notes: Mirror of `RCVBUF` — same rationale against overriding.
7. Eugo conclusion:
   1. (-) do not set.

(~) **NCCL_SOCKET_INLINE**
1. Default value: `128` (bytes)
2. Type: int (bytes; power of 2)
3. Read / Write: R
4. Recommended value (per scenario): **default everywhere**
5. Mechanism: **Process: host-side socket transport.** Internal threshold in `src/transport/net_socket.cc`.
6. Notes: Below this size, socket sends are packed inline with the control header rather than issued as a separate I/O. Tuning is workload-specific and no public guidance exists for when to change it.
7. Eugo conclusion:
   1. (~) internal, leave default. No evidence overriding helps.

(~) **NCCL_SOCKET_MIN_TASKSIZE**
1. Default value: `65536` (64 KiB)
2. Type: int (bytes)
3. Read / Write: R
4. Recommended value (per scenario): **default everywhere**
5. Mechanism: **Process: host-side socket transport.** Minimum task size for socket transport.
6. Notes: Transfers smaller than this are kept on the main thread rather than dispatched to helper threads. Keeps per-task overhead from dominating small transfers.
7. Eugo conclusion:
   1. (~) internal, leave default.

---

## 2. InfiniBand / RoCE Configuration

Controls for IB Verbs transport. Only relevant with RDMA-capable NICs.

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_IB_DISABLE` | When set to `1`, `ncclParamIbDisable()` returns true at the top of `ncclIbInitDevices()` (`src/transport/net_ib/init.cc:207`), which immediately returns `ncclInternalError` — before any IB device enumeration, QP creation, or memory registration. This causes NCCL's net plugin chain to skip the IB plugin entirely and fall back to the next transport: **TCP socket transport** (`net_socket.cc`). All RDMA codepaths (QP creation, `ibv_reg_mr`, RDMA writes) are never entered. Overrides `NCCL_IB_HCA` (which is never even parsed if IB is disabled). When disabled, `NCCL_SOCKET_IFNAME` becomes critical since sockets are the only inter-node transport. | `0` | — | `src/transport/net_ib/init.cc` | ✅ **TCP/SN**: Set `1`. **IB**: Leave `0` |
| `NCCL_IB_HCA` | Filter IB Verbs devices (read via `ncclGetEnv("NCCL_IB_HCA")` in `ncclIbInitDevices()`). Comma-separated list parsed by `parseStringList()` into a `netIf[]` array (max `MAX_IB_DEVS=32` entries). Per-device/port filtering at line 253: `matchIfList(devices[d]->name, port_num, ...)`. **Prefix match** (default): `mlx5` matches `mlx5_0`, `mlx5_1`, etc. (uses `strncmp`). **`=` exact match**: `=mlx5_0` matches only `mlx5_0`. **`:port` syntax**: `=mlx5_0:1` — port number stored separately by `parseStringList()`, checked by `matchPort()`. **`^` exclusion**: `^mlx5_2` inverts — all devices except `mlx5_2*` pass. | All available | — | `src/transport/net_ib/init.cc` | 🔧 **IB**: Set if you need specific NICs (e.g., `=mlx5_0:1,mlx5_1:1`) |
| `NCCL_IB_TIMEOUT` | IB Verbs QP timeout, set via `qpAttr.timeout` in `ncclIbRtsQp()` when transitioning QP to Ready-To-Send state. The IB spec defines timeout as **4.096µs × 2^value**. Value 14 → ~67ms, 18 → ~1.07s, 20 (default) → **~4.3s**, 22 → ~17.2s. Values 0 or ≥31 → infinite timeout. **Error 12 from `ibv_poll_cq`** = `IBV_WC_RETRY_EXC_ERR` (retry count exceeded): the QP retried `IB_RETRY_CNT` times (default 7), each waiting for the timeout period, without receiving an ACK. Total max wait = retries × timeout. Increase for very large IB fabrics with many switch hops. | `20` (was 18 pre-2.23, 14 pre-2.14) | — | `src/transport/net_ib/connect.cc` | ⚙️ Increase for very large fabrics. `0` or `≥31` = infinite |
| `NCCL_IB_RETRY_CNT` | IB QP retry count for failed transmissions. Set via `qpAttr.retry_cnt` in `ncclIbRtsQp()`. After this many retries (each waiting `IB_TIMEOUT` duration) with no ACK, the QP enters error state and `ibv_poll_cq` returns error 12 (`IBV_WC_RETRY_EXC_ERR`). Total max wait before failure = `IB_RETRY_CNT × (4.096µs × 2^IB_TIMEOUT)`. | `7` | 2.1.15 | `src/transport/net_ib/connect.cc` | ⚙️ Leave default |
| `NCCL_IB_GID_INDEX` | GID table index for RoCE. Read by `ncclIbGetGidIndex()`. **When `-1` (auto)**: On **IB link layer**, uses `NCCL_IB_ROUTABLE_FLID_GID_INDEX` (default 1) for routable FLID, falls back to index 0. On **RoCE (Ethernet) link layer**, iterates all GID entries scoring candidates by: (1) address family match (`IB_ADDR_FAMILY`), (2) subnet match (`IB_ADDR_RANGE`), (3) RoCE version match (`IB_ROCE_VERSION_NUM`, default 2 = RoCEv2). Wrong GID index on RoCE → connection failures. Use `show_gids` to verify auto-detected index. | `-1` (auto) | 2.1.4 | `src/transport/net_ib/connect.cc` | 🔧 **IB/RoCE**: Set per `show_gids` output if auto-detect fails |
| `NCCL_IB_ADDR_FAMILY` | IP address family filter for GID auto-selection on RoCE. Used by `ncclUpdateGidIndex()` to prefer GIDs matching this family (AF_INET or AF_INET6). Only relevant when `IB_GID_INDEX=-1`. | `AF_INET` | 2.21 | `src/transport/net_ib/connect.cc` | ⚙️ Leave default |
| `NCCL_IB_ADDR_RANGE` | CIDR range (e.g., `10.0.0.0/24`) to filter GID candidates during auto-selection. `ncclUpdateGidIndex()` prefers GIDs whose IP falls within this range. Only relevant when `IB_GID_INDEX=-1`. | Unset (ignored) | 2.21 | `src/transport/net_ib/connect.cc` | 🔧 **IB**: Set if you need to restrict GID selection to a subnet |
| `NCCL_IB_ROCE_VERSION_NUM` | RoCE version preference for GID auto-selection. 1 = RoCEv1, 2 = RoCEv2. `ncclUpdateGidIndex()` prefers GID entries matching this version. Only relevant when `IB_GID_INDEX=-1` on RoCE. | `2` | 2.21 | `src/transport/net_ib/connect.cc` | ⚙️ Leave default (RoCEv2) |
| `NCCL_IB_SL` | InfiniBand Service Level, set on QP creation. Controls routing priority and path selection in the IB fabric. | `0` | 2.1.4 | `src/transport/net_ib/connect.cc` | 🔧 **IB**: Set per fabric admin instructions |
| `NCCL_IB_TC` | InfiniBand Traffic Class, set on QP creation. Used for QoS traffic classification — the IB switch uses this to map packets to virtual lanes. | `0` | 2.1.15 | `src/transport/net_ib/connect.cc` | 🔧 **IB**: Set per fabric QoS policy |
| `NCCL_IB_FIFO_TC` | Separate IB Traffic Class for control messages (credit return FIFO). Allows the credit-return path to use a high-priority traffic class while bulk data uses the default TC, preventing head-of-line blocking on congested data lanes. Falls back to `NCCL_IB_TC` when unset. | Same as `NCCL_IB_TC` | 2.22.3 | `src/transport/net_ib/connect.cc` | 🔧 **IB**: Set to high-priority TC if available |
| `NCCL_IB_PKEY` | InfiniBand Partition Key, set on QP creation. Restricts communication to peers sharing the same PKey — fabric-level access control. | `0` | — | `src/transport/net_ib/connect.cc` | ⚙️ Leave default unless fabric uses partitioning |
| `NCCL_IB_RETURN_ASYNC_EVENTS` | Report fatal IB async events and stop IB comms | `1` | 2.23 | `src/transport/net_ib/common.cc` | ⚙️ Leave enabled |
| `NCCL_IB_QPS_PER_CONNECTION` | Queue Pairs per connection. Read in `ncclIbSetupConn()` — total QPs = `QPS_PER_CONNECTION × ndevs` (physical devices per merged NIC). CQ is sized proportionally (`2 × MAX_REQUESTS × nqps`). In `ncclIbMultiSend()`, data is **striped** across QPs in 128B-aligned chunks. QPs are created round-robin across devices (`devIndex = qpIndex % ndevs`). **Benefit**: each QP gets a different path through the IB switch fabric (different SL/source port hash). On multi-tier fat-tree fabrics, more QPs = better utilization of available paths = higher aggregate bandwidth. Interacts with `IB_SPLIT_DATA_ON_QPS` (controls striping mode). | `1` | 2.10 | `src/transport/net_ib/connect.cc` | 🔧 **IB multi-tier**: Set `2-4` for multi-rail fabrics |
| `NCCL_IB_SPLIT_DATA_ON_QPS` | Controls data distribution across multiple QPs. `0` (round-robin): uses `nDataQps = max(local_ndevs, remote_ndevs)`. `1`: uses all `nqps` for striping. | `0` (since 2.20) | 2.18 | `src/transport/net_ib/connect.cc` | ⚙️ Leave `0` (round-robin) unless testing |
| `NCCL_IB_AR_THRESHOLD` | Message size threshold for adaptive-routing-friendly message split. In `ncclIbMultiSend()`, when AR is active AND message > threshold: data is sent as a plain `RDMA_WRITE` (no immediate data), plus a separate **0-byte `RDMA_WRITE_WITH_IMM`** to signal completion. The IB switch can re-route each message independently. When AR is off or message ≤ threshold: data and completion are combined into a single `RDMA_WRITE_WITH_IMM` (one fewer verb post). Setting > `NCCL_BUFFSIZE` effectively disables the split for all messages. | `8192` | 2.6 | `src/transport/net_ib/p2p.cc` | ⚙️ Setting > BUFFSIZE disables AR entirely |
| `NCCL_IB_ADAPTIVE_ROUTING` | Per-device adaptive routing flag. Default `-2` = **auto**: enabled on IB link layer, disabled on RoCE. Set at device enumeration time in `ncclIbInitDevices()`: `ar = (link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0`. User override replaces per-device value. During connection setup, AR is AND'd across all merged devices. Controls the message split behavior in `ncclIbMultiSend()` (see `IB_AR_THRESHOLD`). On AR-capable fabrics, the default 8KB threshold means most data transfers use the split path, improving cross-switch load balancing. | `-2` (auto: `1` IB, `0` RoCE) | 2.16 | `src/transport/net_ib/init.cc` | ⚙️ Enable on AR-capable fabrics |
| `NCCL_IB_PCI_RELAXED_ORDERING` | PCIe Relaxed Ordering for IB memory registration. 0=off, 1=force enable, 2=auto. Auto-detect queries `IBV_ACCESS_RELAXED_ORDERING` via `wrap_ibv_reg_mr_iova2()`. When enabled, allows PCIe reordering of IB DMA transactions, improving throughput especially in VMs. | `2` (auto) | 2.12 | `src/transport/net_ib/init.cc` | ⚙️ Leave `2` for auto-detect. Helps in VMs |
| `NCCL_IB_ECE_ENABLE` | Enhanced Connection Establishment. Enables advanced features (congestion control, adaptive routing, software retransmit) via `ibv_set_ece()`. | `1` (since 2.19) | 2.23 | `src/transport/net_ib/connect.cc` | ⚙️ Leave enabled. Disable if ECE misconfigured at system level |
| `NCCL_IB_MERGE_NICS` | Combine dual-port NICs into a single logical device for bandwidth aggregation. Checked in `ncclIbMakeVDeviceInternal()` — if `0` and `props->ndevs > 1`, returns `ncclInvalidUsage`. When enabled, multiple physical ports are merged into one virtual device with aggregated speed. | `1` | 2.20 | `src/transport/net_ib/init.cc` | ⚙️ Leave enabled |
| `NCCL_IB_MERGE_VFS` | Merge Virtual Functions for SR-IOV. When enabled, `ncclIbMatchVfPath()` compares PCI paths with 4-char suffix tolerance (VF numbering) instead of 1-char (port numbering). | `1` | — | `src/transport/net_ib/init.cc` | ⚙️ Internal, for VM/SR-IOV setups |
| `NCCL_IB_DATA_DIRECT` | Enable direct data path for IB | `1` | — | `src/transport/net_ib/init.cc` | ⚙️ Internal |
| `NCCL_IB_USE_INLINE` | Use inline data for small IB messages | `0` | — | `src/transport/net_ib/connect.cc` | ⚙️ Internal |
| `NCCL_IB_WARN_RAIL_LOCAL` | Warn about rail-local suboptimal NIC assignment | `0` | — | `src/transport/net_ib/connect.cc` | ⚙️ Internal debug |
| `NCCL_IB_ROUTABLE_FLID_GID_INDEX` | Routable FLID GID index. Used on IB link layer in `ncclIbGetGidIndex()` to check for routable FLIDs at this GID table index. | `1` | — | `src/transport/net_ib/connect.cc` | ⚙️ Internal |
| `NCCL_IB_CUDA_SUPPORT` | Force/disable GPU Direct RDMA (removed in 2.4, see `NCCL_NET_GDR_LEVEL`) | — | removed 2.4.0 | — | 🚫 Deprecated, use `NCCL_NET_GDR_LEVEL` |
| `NCCL_GDR_FLUSH_DISABLE` | Disable GDR flush | `0` | — | `src/transport/net_ib/connect.cc` | ⚙️ Internal |
| `NCCL_IB_MQP_RETRY_ALL` | Multi-QP retry for all operations | `0` | — | `src/misc/ibvwrap.cc` | ⚙️ Internal |
| `NCCL_IB_MQP_RETRY_CNT` | Multi-QP retry count | `34` | — | `src/misc/ibvwrap.cc` | ⚙️ Internal |
| `NCCL_IB_MQP_RETRY_SLEEP_MSEC` | Multi-QP retry sleep (ms) | `100` | — | `src/misc/ibvwrap.cc` | ⚙️ Internal |

---

## 3. Out-of-Band & Bootstrap

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_OOB_NET_ENABLE` | Controls whether the bootstrap ring uses NCCL's net plugin (IB/Socket) or plain TCP sockets for out-of-band init operations. Checked at 6 critical points in `bootstrapInit()`, `bootstrapSplit()`, `bootstrapAllGather()`, and `bootstrapClose()`. **When `0` (default)**: the bootstrap ring uses TCP sockets (`ncclSocket*` calls — `socketRingConnect`, `socketRingAllGather`). **When `1`**: uses the NCCL net plugin (`state->net->listen()`, `state->net->isend()`, `state->net->irecv()`), so all OOB control traffic (rank discovery, barrier, AllGather of connect info) goes over the high-speed network (e.g., IB) instead of TCP. Note: P2P bootstrap messages (`bootstrapSend`/`bootstrapRecv`) always use TCP sockets regardless of this flag — only the ring AllGather path is switched. | `0` | 2.23 | `src/bootstrap.cc` | 🔧 **IB**: Set `1` to use IB for OOB comms |
| `NCCL_OOB_NET_IFNAME` | Filter interfaces for OOB net (when `OOB_NET_ENABLE=1`). Parsed like `NCCL_SOCKET_IFNAME`. | — | 2.23 | `src/bootstrap.cc` | 🔧 Set if using OOB over specific IB device |
| `NCCL_UID_STAGGER_THRESHOLD` | Number of ranks above which bootstrap connections are staggered to avoid overwhelming the root listener. | `256` | 2.23 | `src/bootstrap.cc` | ⚙️ Reduce for smaller UniqueId sockets |
| `NCCL_UID_STAGGER_RATE` | Target message rate (msgs/sec) when staggering bootstrap connections. | `7000` | 2.23 | `src/bootstrap.cc` | ⚙️ Leave default |
| `NCCL_COMM_ID` | Manual communicator ID in `host:port` format (read via `ncclGetEnv("NCCL_COMM_ID")`). Parsed by `ncclSocketGetAddrFromString()` supporting IPv4 (`10.0.0.1:12345`), IPv6 (`[::1]:12345`), and hostname (`gpu-node:12345`). **When set**: (1) in `bootstrapGetUniqueId()`, the handle's address is set directly to the parsed address (skipping normal `bootstrapCreateRoot()` listen), magic set to `NCCL_MAGIC`; (2) in `ncclCommInitRankDev()`, rank 0 starts listening on this address; (3) in `ncclFindInterfaces()`, used as subnet hint for interface auto-detection. Eliminates the need for programmatic `ncclGetUniqueId()` + broadcast — all ranks just read the address from env. Cannot be used with `ncclCommGrow()` or multi-root init. | Auto | — | `src/bootstrap.cc`, `src/init.cc`, `src/misc/socket.cc` | 🔧 Set for manual bootstrap (no MPI/NCCL unique ID) |
| `NCCL_HOSTID` | Override host identification | Auto | — | `src/misc/utils.cc` | ⚙️ Internal, for environments where hostname detection fails |

---

## 4. Debug & Logging

> Section marker: **(+)** — we pin `NCCL_DEBUG=WARN` everywhere as a production floor (operators override to INFO/TRACE via shell env when debugging). Other knobs here are left default.

---

(+) **NCCL_DEBUG**
1. Default value: unset (no debug output except some internal VERSION prints)
2. Type: string (`VERSION` | `WARN` | `INFO` | `ABORT` | `TRACE`)
3. Read / Write: R
4. Recommended value (per scenario): **grace_gb10: WARN | g5g: WARN | aws_efa: WARN**
5. Mechanism: **Process: all NCCL threads (host + proxy).** Read via `getEnvFunc("NCCL_DEBUG")` in `ncclDebugInit()` ([src/debug.cc](../src/debug.cc)), stored in the global `ncclDebugLevel`.
6. Notes: Five levels. `VERSION` (1) — only the NCCL version string at init, no hostname/pid prefix. `WARN` (2) — VERSION + warnings, formatted as `\n[timestamp] host:pid:tid [cudaDev] file:line NCCL WARN message\n`; warnings are also saved to `ncclLastError[]` for `ncclGetLastError()` (so even if the operator never looks at stderr, the API call gives them the last warning). `INFO` (3) — WARN + informational messages filtered by subsystem mask (default mask: `INIT|BOOTSTRAP|ENV`). `ABORT` (4) — like WARN but also calls abort() on the warning path. `TRACE` (5) — all messages; **requires compile-time `ENABLE_TRACE` define** in [src/include/debug.h](../src/include/debug.h) — without it, `TRACE()` is a no-op macro. **TRACE performance cost**: every call acquires `ncclDebugMutex` (a `std::mutex`), calls `clock_gettime(CLOCK_REALTIME)`, formats a string, then `vfprintf()`. The mutex serializes every log statement across all threads — devastating for latency-sensitive collectives. On a 72-GPU AllReduce this alone can add milliseconds. `INFO()` short-circuits with an atomic load before calling the log function, so the cost when the level is below INFO is a single atomic compare — essentially free. **Why `WARN` is the right floor**: anything quieter hides issues the operator genuinely needs to see (IB QP errors, FM misconfig, OOM); anything louder is expensive and noisy across 72-rank logs.
7. Eugo conclusion:
   1. (+) `WARN` on all three scenarios. Shell env overrides to INFO/TRACE when debugging; conf floor ensures we never silently run with zero logging.

(~) **NCCL_DEBUG_FILE**
1. Default value: unset (writes to stdout/stderr)
2. Type: string (path; supports `%h` = hostname, `%p` = PID, `%%` = literal `%`)
3. Read / Write: R
4. Recommended value (per scenario): **unset in conf; operator sets when needed** (e.g., `/tmp/nccl-%h-%p.log`)
5. Mechanism: **Process: all NCCL threads.** Read in `ncclDebugInit()` via `getEnvFunc("NCCL_DEBUG_FILE")`.
6. Notes: Path is scanned char-by-char in `ncclDebugInit()`: `%h` → hostname with `%` and `/` replaced by dashes; `%p` → PID in decimal; `%%` → literal `%`. Opened with `fopen(path, "w")` (truncating) and `setlinebuf()` for line-buffered real-time output. Only processed if `ncclDebugLevel > NCCL_LOG_VERSION` — i.e., if `NCCL_DEBUG` wasn't set or is `VERSION`, this env var is ignored. If file can't be opened (permissions, missing dir), NCCL stays on stdout silently. Baking `/tmp/nccl-%h-%p.log` into the conf would fragment logs across nodes and make log shipping annoying; better to let the launcher decide.
7. Eugo conclusion:
   1. (~) do not set in conf; leave to operators / launcher. Keep template in the docs.

(~) **NCCL_DEBUG_SUBSYS**
1. Default value: `INIT,BOOTSTRAP,ENV` (mask `0x1081`)
2. Type: string (CSV of subsystem names; `^` prefix inverts)
3. Read / Write: R
4. Recommended value (per scenario): **unset in conf; operator sets `ALL` or `INIT,NET` when debugging**
5. Mechanism: **Process: all NCCL threads.** Read via `getEnvFunc("NCCL_DEBUG_SUBSYS")` in `ncclDebugInit()`, parsed with `strtok(copy, ",")` + `strcasecmp()`.
6. Notes: Filters INFO-level output by subsystem bitmask. Only active when `NCCL_DEBUG=INFO` or higher. Valid names: `INIT` (0x1), `COLL` (0x2), `P2P` (0x4), `SHM` (0x8), `NET` (0x10), `GRAPH` (0x20), `TUNING` (0x40), `ENV` (0x80), `ALLOC` (0x100), `CALL` (0x200), `PROXY` (0x400), `NVLS` (0x800), `BOOTSTRAP` (0x1000), `REG` (0x2000), `PROFILE` (0x4000), `RAS` (0x8000), `ALL` (~0). `^TUNING` excludes TUNING. When our baseline is `WARN`, this has no effect — included only for when an operator flips to INFO.
7. Eugo conclusion:
   1. (~) do not set in conf. Operator convention: `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL` for full debugging; `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` for network issues.

(~) **NCCL_DEBUG_TIMESTAMP_FORMAT**
1. Default value: `[%F %T]` (e.g., `[2026-04-22 14:03:22]`)
2. Type: string (strftime format)
3. Read / Write: R
4. Recommended value (per scenario): **default everywhere**
5. Mechanism: **Process: all NCCL threads.** Applied in the log formatter.
6. Notes: Standard strftime. `%Xf` gives fractional seconds. Underscores render as spaces. `[%F %T.%3f]` would add milliseconds — useful for correlating NCCL events with app-side timestamps, but aggregated log tools (CloudWatch, Splunk) re-add their own timestamps anyway.
7. Eugo conclusion:
   1. (~) default is fine; no value in overriding.

(~) **NCCL_DEBUG_TIMESTAMP_LEVELS**
1. Default value: `WARN` (timestamp only on WARN-level messages)
2. Type: string (CSV: `VERSION`, `WARN`, `INFO`, `ABORT`, `TRACE`, `ALL`; `^` inverts)
3. Read / Write: R
4. Recommended value (per scenario): **default everywhere**
5. Mechanism: **Process: all NCCL threads.** Applied in the log formatter.
6. Notes: Which log levels get timestamps prepended. Default `WARN` means our baseline prod logs (WARN-only) always carry timestamps.
7. Eugo conclusion:
   1. (~) default is fine.

(-) **NCCL_WARN_ENABLE_DEBUG_INFO**
1. Default value: unset (not auto-promoting)
2. Type: bool (`1` enables)
3. Read / Write: R
4. Recommended value (per scenario): **do not set**
5. Mechanism: **Process: all NCCL threads.**
6. Notes: When `=1`, after the first WARN fires NCCL auto-promotes the global debug level to INFO. Intended as "turn on verbose debug after first sign of trouble" — but the behavior is racy with the reproduction conditions (by the time INFO kicks in, the earlier init events are gone) and log volume explodes unpredictably. Better to set INFO pre-emptively when chasing a bug.
7. Eugo conclusion:
   1. (-) do not set. Racy and produces less-useful logs than a pre-emptive INFO.

(+) **NCCL_CONF_FILE**
1. Default value: unset (cascade falls through to `~/.nccl.conf` then `/usr/local/etc/nccl.conf`)
2. Type: string (absolute or relative path)
3. Read / Write: R
4. Recommended value (per scenario): **set in launcher/image to the shipped conf path** (e.g., `/etc/nccl/nccl.conf.aws_efa`)
5. Mechanism: **Process: host-side init, once per process via `std::call_once`.** Read by `initEnvFunc()` → `setEnvFile()` in [src/misc/param.cc](../src/misc/param.cc).
6. Notes: **Loading priority (first-match-wins, EUGO patched)**: (1) `$NCCL_CONF_FILE` if set, non-empty, file exists; (2) `~/.nccl.conf` if exists; (3) `/usr/local/etc/nccl.conf` if exists. Format: plain text, `KEY=VALUE` per line, `#` comments at line start, NO quoting, whitespace NOT trimmed, 1023-byte cap on key and value (silent truncation above that — verify with our `validate_nccl_conf.sh`). `setEnvFile()` reads line-by-line and calls `ncclOsSetEnv(key, value)` for each; `ncclOsSetEnv` is `setenv(name, value, 0)` on Linux — meaning **shell env vars always win over conf values**. **@EUGO_CHANGE**: upstream merges all three sources and unconditionally loads `/etc/nccl.conf` last (silently overriding user files); we changed to first-match-wins and moved the system default to `/usr/local/etc/nccl.conf` to avoid conflicting with distro packages that may drop files into `/etc/nccl.conf`. This variable itself isn't something we set *in* the conf (chicken-and-egg); operators set it at launcher level to point at the right per-scenario conf.
7. Eugo conclusion:
   1. (+) set by the launcher/AMI at process start to select the right `nccl.conf.{grace_gb10,g5g,aws_efa}`. Document in deployment runbooks.

---

## 5. Topology & Graph

### 5.0. Topology reliability on p6e-gb200 — should we ship a hand-curated XML? (2026-04 research)

**Short answer**: **No — trust NCCL 2.29 auto-detection on p6e-gb200. Do not ship a blind XML.** Capture a dump on first boot, diff across NCCL upgrades, override with `NCCL_TOPO_FILE` only if a reproducible wrong-graph case is traced to an upstream issue.

Detail below.

#### Q1. How reliable is NCCL 2.29 auto-detection on p6e-gb200 (9- and 18-tray UltraServers)?

**Bare-metal: reliable.** NVIDIA [explicitly states](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html) that on MNNVL systems like GB200 NVL72, NCCL auto-detects NVLink domains and selects optimal algorithms (requires NCCL ≥ 2.25.2; we're on 2.29.3). NCCL 2.29 specifically shipped updated tuning data for GB200-NV72.

**Container edge case: fixed in 2.29.** One of the 2.29.x changelog entries reads: "NCCL topology generation for GB200 in Docker containers where NUMA nodes disconnected from Package nodes caused incorrect topology generation." So Docker-in-container was broken before 2.29, fixed now. If we ship inside a container, this is the version floor to respect.

**Known open issues we should track (as of 2026-04)**:
- [NVIDIA/nccl #1810](https://github.com/NVIDIA/nccl/issues/1810) — tuning opportunity for broadcast on GB200 NVL72 (open)
- [NVIDIA/nccl #1801](https://github.com/NVIDIA/nccl/issues/1801) — higher AllReduce bandwidth with tree protocol between three NVL partitions on GB200 when `NCCL_NVLS_ENABLE=0` (open; suggests the auto-tuner picks suboptimal algo for some partition layouts)
- [NVIDIA/nccl #1913](https://github.com/NVIDIA/nccl/issues/1913) — **NCCL 2.28.9 GIN incompatible with external network plugins on multi-rail EFA** (open; directly relevant → we pin `NCCL_GIN_ENABLE=0` on aws_efa)
- [pytorch/pytorch #161116](https://github.com/pytorch/pytorch/issues/161116) — NCCL segfault when initializing a process group on an NVL72 (>10 trays / 40 GPUs, earlier NCCL)

None of these are wrong-XML issues — they're tuning heuristics or the GIN/EFA incompatibility. Auto-detection itself produces the right graph.

#### Q2. How stable is the XML schema across NCCL versions?

**Tags are stable, attributes are not, and there is no formal contract.** The core tag set (`system` / `cpu` / `pci` / `gpu` / `nic` / `nvlink`) has been present since NCCL 2.6 (when `NCCL_TOPO_FILE` was introduced). But [src/graph/xml.h](../src/graph/xml.h) has grown ~50% from 9,437 bytes in v2.20.3 to 14,194 bytes in v2.29.7 — additions of attributes and constraints that can change parse behavior. NVIDIA does **not** publish a stability contract. There's at least one historical break: [#326](https://github.com/NVIDIA/nccl/issues/326) reported a crash moving from 2.5 → 2.6.

**Implication**: an XML captured on NCCL 2.29 is not guaranteed to parse cleanly on 2.30+. If we ship a topology file with an AMI, we have to re-validate it on every NCCL upgrade (or ship a different file per NCCL version).

#### Q3. Can the p6e-gb200 topology XML be downloaded without spawning an instance?

**No. Not currently.** Searched (2026-04):
- [`aws/aws-ofi-nccl/tree/master/topology`](https://github.com/aws/aws-ofi-nccl/tree/master/topology) — only ships `g5.48xl-topo.xml`, `p4d-24xl-topo.xml`, `p4de-24xl-topo.xml`. **No p5 / p5e / p6 / p6e files.**
- [`aws/aws-ofi-nccl/include/platform-aws.h`](https://github.com/aws/aws-ofi-nccl/blob/master/include/platform-aws.h) — defines the `PLATFORM_NAME_P6E_GB200` string but configures the plugin at runtime rather than shipping an XML.
- [`awslabs/awsome-distributed-training`](https://github.com/awslabs/awsome-distributed-training) — the cluster-topology tool emits Mermaid diagrams, not NCCL XML. No p6e XML.
- NGC / AWS docs / AWS DLAMI release notes — no published topology XML.

**Only way to obtain it today**: spawn a p6e-gb200 instance, then either (a) copy `/var/run/nvidia-topologyd/virtualTopology.xml` if the NVIDIA topologyd daemon is running, or (b) set `NCCL_TOPO_DUMP_FILE=/tmp/nccl-topo.xml` during a first-boot NCCL init and grab the dumped auto-detected XML.

#### Q4. Do UltraClusters need a different topology file than a single UltraServer?

**No.** aws-ofi-nccl handles inter-UltraServer EFA traffic as an opaque libfabric SRD transport. NCCL sees EFA NICs as `<nic>` endpoints and computes rings/trees across them; the EFA fabric's *structure* (which top-of-rack switch, which spine) is abstracted away by SRD multipathing. **The same per-host XML works whether the UltraServer is standalone or part of an UltraCluster.** What NCCL needs is per-host NIC↔GPU↔CPU affinity, which is identical in both deployments.

The knobs that actually matter for UltraCluster performance are runtime settings, not XML:
- `NCCL_CROSS_NIC=1` (we set this) — allows topology search to use alternating NICs across channels, exploiting the fat-tree.
- `NCCL_PXN_C2C=1` (we set this) — routes remote-GPU→local-NIC traffic through Grace C2C rather than PCIe.
- `NCCL_MNNVL_ENABLE=1` (we set this, strict) — keeps intra-UltraServer 72-GPU collectives on NVLink instead of EFA.

#### Q5. What to ship?

1. **Don't set `NCCL_TOPO_FILE`** on the default image. Let auto-detection run.
2. **On canary / first boot** of a new AMI: set `NCCL_TOPO_DUMP_FILE=/tmp/nccl-topo-p6e-gb200-<nccl-version>.xml` once, capture the file, commit to the repo as a reference (for diffing).
3. **On each NCCL upgrade**: re-capture on a throwaway instance, diff against the previous version's dump. If the graph is materially different, investigate. If the graph is identical, no action.
4. **Only if** a reproducible wrong-graph case is observed (visible as unexpected NCCL INFO channel output or a nccl-tests bandwidth regression), file upstream and pin `NCCL_TOPO_FILE` to a hand-edited XML as a workaround. This is a last resort, not a routine.

#### Q6. p6e-gb200 hardware specs (from AWS docs)

| Item | Per tray (`p6e-gb200.36xlarge`) | UltraServer 9x (`u-p6e-gb200x36`) | UltraServer 18x (`u-p6e-gb200x72`) |
|---|---|---|---|
| GB200 Superchips | 2 | 18 | 36 |
| Blackwell GPUs | 4 | 36 | 72 |
| Grace CPUs | 2 | 18 | 36 |
| EFAv4 NICs | up to 17 @ 400 Gbps | — | — |
| Aggregate EFA BW | — | up to 28.8 Tbps | up to 28.8 Tbps |
| NVLink domain | intra-tray only | single domain across all 36 GPUs | single domain across all 72 GPUs |
| HBM3e total | — | 6.7 TB | 13.4 TB |

[AWS p6e-gb200 product page](https://aws.amazon.com/ec2/instance-types/p6/) • [AWS p6e-gb200 launch blog](https://aws.amazon.com/blogs/aws/new-amazon-ec2-p6e-gb200-ultraservers-powered-by-nvidia-grace-blackwell-gpus-for-the-highest-ai-performance/) • [ParallelCluster p6e IMEX docs](https://docs.aws.amazon.com/parallelcluster/latest/ug/support-nvidia-imex-p6e-gb200-instance.html)

---

### 5.1. Variable reference

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_TOPO_FILE` | Load XML topology file before auto-detection. Read via `ncclGetEnv("NCCL_TOPO_FILE")` in `ncclTopoGetXmlFromFile()`. The XML is parsed by `xmlLoadSub()` with a `"system"` root handler. **If unset**: NCCL tries `/var/run/nvidia-topologyd/virtualTopology.xml` first (non-fatal if missing), then auto-detects topology via sysfs/NVML. **Malformed file**: if the file opens but XML parsing fails (malformed tags, missing attributes), `xmlLoadSub` propagates an error that fails `ncclTopoGetSystem()` → aborts communicator creation. **No graceful fallback from parse errors** — only from missing-file. After loading, host_hash attributes on `<cpu>` nodes are forcibly overwritten with the local host hash (the XML file's values are not trusted). | `/var/run/nvidia-topologyd/virtualTopology.xml` if exists | 2.6 | `src/graph/topo.cc` | 🔧 **IB**: Set if auto-detect is wrong for your fabric |
| `NCCL_TOPO_DUMP_FILE` | Dump detected XML topology to file | — | 2.6 | `src/init.cc` | 🔧 Set for debugging topology issues |
| `NCCL_TOPO_DUMP_FILE_RANK` | Which rank dumps topology | `0` | — | `src/graph/topo.cc` | ⚙️ Set `0` to only dump from rank 0 |
| `NCCL_GRAPH_FILE` | Load communication graph from file | — | — | `src/graph/search.cc` | ⚙️ For custom graph injection |
| `NCCL_GRAPH_DUMP_FILE` | Dump searched communication graph to file | — | — | `src/graph/search.cc` | 🔧 For debugging graph search |
| `NCCL_GRAPH_DUMP_FILE_RANK` | Which rank dumps graph | `0` | — | `src/init.cc` | ⚙️ Set `0` |
| `NCCL_CROSS_NIC` | Cross-NIC ring/tree policy. Checked in `ncclTopoSearchCheckNet()` and `ncclTopoCompute()`. **`0` (disable cross-NIC)**: when returning to the NIC at end of a ring/tree, the NIC must have the same ASIC and port as the start NIC — forces traffic to stay on the same physical NIC (rail-local). XML graph paths with `crossnic=1` are rejected. **`1` (force cross-NIC)**: `graph->crossNic=1`, the search does NOT constrain exit NIC to match entry NIC — any reachable NIC can be used. **`2` (auto, default)**: search starts with `crossNic=0`; if pass 1 fails for RING/BALANCED_TREE, retries with `crossNic=2` where alternating channels use different NICs (even channels: NIC A→...→NIC A; odd channels: NIC A→...→NIC B), creating a "figure-8" pattern in `ncclTopoConnectRings()`. Cross-NIC doubles effective inter-node bandwidth when there are multiple NICs per node. | `2` | — | `src/graph/search.cc` | ⚙️ **IB rail-optimized**: `0`. **IB fat-tree**: `1`. Default `2` is safe |
| `NCCL_IGNORE_CPU_AFFINITY` | Controls whether the process's CPU affinity mask is used in NIC selection. Read in `ncclTopoGetCpuAffinity()`. NCCL assigns threads to CPU cores near the GPU to minimize memory access latency. The final CPU affinity is normally the **intersection** of: (a) the process's current CPU affinity (from `sched_getaffinity`), and (b) the NUMA node affinity closest to the GPU. **When `1`**: the process's CPU affinity mask is **ignored** — the full GPU-local NUMA node affinity is used (`finalMask = cpuMask`). **When `0` (default)**: `finalMask = ncclOsCpuAnd(mask, cpuMask)` (intersection). Set this when your job launcher (Slurm, MPI) pins processes to CPU cores that don't overlap with the GPU's NUMA node — without it, NCCL may get an empty affinity mask and fall back to suboptimal NIC selection routing memory traffic across NUMA boundaries. | `0` | 2.4.6 | `src/graph/topo.cc` | 🔧 Set `1` if CPU affinity causes suboptimal NIC selection |
| `NCCL_NET_FORCE_MERGE` | Force-merge network devices by name pattern | — | — | `src/graph/topo.cc` | ⚙️ Internal |
| `NCCL_NET_MERGE_LEVEL` | Network device merge level | — | — | `src/graph/topo.cc` | ⚙️ Internal |
| `NCCL_NETDEVS_POLICY` | Network device assignment policy. Read via `ncclGetEnv("NCCL_NETDEVS_POLICY")` in `getNetDevsPolicyOnce()` (parsed once via `std::call_once`). **`AUTO` (default)**: computes `netsPerGpu = DIVUP(localNetCount, localGpuCount)` in `ncclTopoGetLocalNet()` — distributes NICs evenly across GPUs (e.g., 8 NICs / 8 GPUs = 1 NIC/GPU). **`ALL`**: `netsPerGpu = localNetCount` — every GPU can use ALL local NICs, channels cycle through them. **`MAX:N`**: caps NICs per GPU to N — in `ncclTopoSelectNets()`, the sorted NIC list is truncated. The policy controls how many NICs are considered during topology graph search. AUTO prevents wasting search time on far NICs; ALL maximizes bandwidth utilization; MAX:N provides hard caps. | `AUTO` | 2.28 | `src/graph/topo.cc` | 🔧 **IB**: Set `MAX:N` if AUTO causes sharing conflicts |
| `NCCL_NET_DISABLE_INTRA` | Disable intra-node net usage | `0` | — | `src/graph/paths.cc` | ⚙️ Internal |

---

## 6. Transport: P2P / SHM / Net

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_P2P_DISABLE` | Disable P2P (NVLink/PCIe direct GPU access) entirely | `0` | — | implied | 🚫 Only for debugging. Use `P2P_LEVEL` for fine control |
| `NCCL_P2P_LEVEL` | Max allowed topological distance for CUDA P2P (GPUDirect). Read via `ncclGetLevel()` supporting both string and legacy int values. **All levels and physical meaning**: `LOC`(0) = same GPU (self) — **disables P2P entirely**; `NVL`(1) = direct NVLink; `NVB`(2) = 2-hop NVLink via intermediate GPU; `C2C`(3) = Chip-to-Chip (Grace Hopper); `PIX`(4) = single PCIe bridge (GPUs under same PCIe switch); `PXB`(5) = multiple PCIe bridges, not crossing CPU host bridge; `PHB`(8) = crosses PCIe Host Bridge (through CPU); `SYS`(9) = crosses SMP interconnect (QPI/UPI between NUMA nodes). **Legacy int** 0-5 mapped via `levelsOldToNew[]`. In `ncclTopoCheckP2p()`: default is `PATH_PXB` (exception: AMD dual-GPU → `PATH_SYS`). P2P enabled only if `path->type <= p2pLevel`. Also reads legacy `NCCL_P2P_DISABLE` — if `1`, level becomes `PATH_LOC` (P2P disabled). | Auto (`PXB`) | 2.3.4 | `src/graph/paths.cc` | ⚙️ Leave auto. 🔧 Set `NVL` to restrict P2P to NVLink only |
| `NCCL_P2P_DIRECT_DISABLE` | Controls P2P direct pointer mode. When two ranks share the same process (`P2P_SAME_PID` — same host+pid hash) AND are on different GPUs, NCCL can use **direct device pointers** (`P2P_DIRECT`) instead of CUDA IPC handles or cuMem API. Direct pointers avoid IPC handle creation/mapping overhead — one rank directly reads/writes the other's GPU memory within the same address space. **When `0` (default)**: `resources->type = P2P_DIRECT` for same-process peers. **When `1`**: forces IPC or cuMem fallback (`P2P_CUMEM` or `P2P_IPC`), adding setup latency but same steady-state bandwidth. Logged as `P2P/direct pointer` vs `P2P/IPC`/`P2P/CUMEM` in channel setup INFO messages. Set `1` if you encounter stability issues with direct device pointers in multi-GPU-per-process configurations. | `0` | — | `src/transport/p2p.cc` | 🔧 Set `1` if buffers aren't P2P-accessible (custom allocators) |
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
| `NCCL_NET_SHARED_BUFFERS` | Controls whether net transport connections share a buffer pool or get dedicated buffers. In both `sendSetup` and `recvSetup`: shared buffers are **never** used for graph-captured operations or the first connection index (connIndex=0). For secondary connections, default is `1` (shared). **When shared (`1`)**: `sharedNetBuffersInit()` creates a **single pool** per local peer: `nChannels × NCCL_SHARED_STEPS(16) × p2pChunkSize` bytes. Multiple connections to the same peer share this pool with slot offsets by channel and step index. **When not shared (`0`)**: each connection gets dedicated `buffSizes[p]` for each protocol. Memory savings: shared scales as O(peers × channels × steps × chunkSize) — constant per peer regardless of connection count. | `-2` (auto→`1`) | 2.8 | `src/transport/net.cc` | ⚙️ Leave enabled |
| `NCCL_NET_SHARED_COMMS` | Reuse connections in PXN context for message aggregation | `1` | 2.12 | `src/transport/net.cc` | ⚙️ Leave enabled |
| `NCCL_NET_OPTIONAL_RECV_COMPLETION` | Optional receive completion | `1` | — | `src/transport/net.cc` | ⚙️ Internal |
| `NCCL_NET_OVERHEAD` | Net overhead estimation for tuning | `-2` (auto) | — | `src/graph/tuning.cc` | ⚙️ Internal |
| `NCCL_NET_FORCE_FLUSH` | Force net flush | `0` | — | `src/graph/paths.cc` | ⚙️ Internal |
| `NCCL_NET_PLUGIN_REF_COUNT` | Net plugin reference counting | `0` | — | `src/plugin/net.cc` | ⚙️ Internal |
| `NCCL_RUNTIME_CONNECT` | Controls lazy vs eager transport connection establishment. **Prerequisite**: requires `cuMemSupport` (CUDA VMM API); if cuMem unavailable, `runtimeConn` is always false regardless of env var. **When `1` (lazy, default)**: during `ncclCommInit`, only channels are set up and NVLS is attempted. Transport connections (Ring, Tree, PAT, CollNet) are NOT established. In `ncclEnqueueCheck()`, when a collective is first enqueued with a specific algorithm, connections for that algorithm are established on-demand (`comm->initAlgoChannels[algo]`). Init is faster, but **first collective of each algorithm type pays a setup penalty**. **When `0` (eager)**: all transport connections (Ring, Tree, PAT, NVLS, CollNet) established during `ncclCommInit`. First collective runs immediately — no on-demand stall. Better for benchmarking or latency-sensitive applications. | `1` | 2.22 | `src/init.cc` | ⚙️ Leave `1`. Set `0` for deterministic init-time connection |
| `NCCL_CONNECT_ROUND_MAX_PEERS` | Max peers per connection round | `128` | — | `src/transport.cc` | ⚙️ Internal |
| `NCCL_REPORT_CONNECT_PROGRESS` | Report connection progress during init | `0` | — | `src/transport.cc` | ⚙️ Internal, useful for debugging slow init |
| `NCCL_IGNORE_DISABLED_P2P` | Ignore disabled P2P settings | `0` | — | `src/graph/paths.cc` | ⚙️ Internal |
| `NCCL_P2P_PER_CHANNEL_NET_BW` | Per-channel net bandwidth for P2P paths (GB/s) | `14` | — | `src/graph/paths.cc` | ⚙️ Internal |
| `NCCL_MAX_P2P_NCHANNELS` | Max P2P channels | `64` (MAXCHANNELS) | — | `src/graph/paths.cc` | ⚙️ Internal |
| `NCCL_MIN_P2P_NCHANNELS` | Min P2P channels | `1` | — | `src/graph/paths.cc` | ⚙️ Internal |
| `NCCL_MULTI_SEGMENT_REGISTER` | Multi-segment registration for generic transport | `1` | — | `src/transport/generic.cc` | ⚙️ Internal |
| `NCCL_SINGLE_PROC_MEM_REG_ENABLE` | Single-process memory registration | `0` | — | `src/group.cc` | ⚙️ Internal |
| `NCCL_IPC_USE_ABSTRACT_SOCKET` | Controls IPC socket type for exchanging CUDA IPC handles (`SCM_RIGHTS` file descriptors) between same-node ranks. Socket name format: `/tmp/nccl-socket-<rank>-<hash>`. **When `1` (abstract, default)**: `sun_path[0] = '\0'` — Linux abstract namespace. Auto-cleaned by kernel when all FDs close; not visible in filesystem (`ls /tmp` won't show them); works if abstract socket namespace is shared (default Docker behavior). **When `0` (filesystem)**: normal bind to path. Requires `unlink()` before bind and on close (stale socket files left after crashes); visible in filesystem; requires `/tmp` shared between containers. Set `0` in environments where abstract Unix sockets don't work (very old kernels, security policies, or when debugging — filesystem sockets visible with `ls` and `ss -x`). | `1` | 2.29 | `src/misc/ipcsocket.cc` | 🔧 Set `0` if containers have different network namespaces |
| `NCCL_P2P_EPOCH_ENABLE` | Enable P2P epoch mode | `1` | — | `src/enqueue.cc` | ⚙️ Internal |

---

## 7. GPU Direct RDMA (GDR)

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_NET_GDR_LEVEL` | Max GPU-NIC topological distance for GPU Direct RDMA (GDR). Parsed via same `ncclGetLevel()` as `P2P_LEVEL` — same string/int levels: `LOC`(0), `NVL`(1), `NVB`(2), `C2C`(3), `PIX`(4), `PXB`(5), `PHB`(8), `SYS`(9). ~~Also reads legacy `NCCL_IB_GDR_LEVEL`.~~ (Removed — no source reader in v2.29.) In `ncclTopoCheckGdr()`: GDR enabled only if `path->type <= gdrLevel`. **Auto default**: `PATH_P2C`(6) — enabled through NVLink/C2C/single-PCIe-switch but NOT across host bridge. Exception: on Amere+NVSwitch or C2C architectures, default is `PATH_SYS`. GDR allows the NIC to DMA directly to/from GPU memory (bypassing CPU bounce buffers). When GDR is disabled for a particular GPU-NIC pair, NCCL stages data through host memory — adding a `cudaMemcpy` on each transfer. Setting too permissive (e.g., `SYS`) can hurt performance by routing RDMA across NUMA boundaries; too restrictive (e.g., `PIX`) may disable GDR entirely on topologies where it would help. | Auto (`PATH_P2C`) | 2.3.4 | `src/graph/paths.cc` | ⚙️ Leave auto. **IB**: 🔧 Set `PHB` if GDR causes issues across NUMA |
| `NCCL_NET_GDR_READ` | Enable GPU Direct RDMA for send path (read from GPU) | `1` (NVLink), `0` (PCIe) since 2.4.2 | — | `src/graph/paths.cc` | ⚙️ Leave auto |
| `NCCL_NET_GDR_C2C` | Enable GDR via NIC on C2C-connected CPU (distance PHB) | `1` (since 2.27; was 0 in 2.26) | 2.26 | `src/graph/paths.cc` | ⚙️ Leave default |
| `NCCL_GDRCOPY_ENABLE` | Enable GDRCopy for small-message GPU Direct RDMA optimization. GDRCopy uses a CPU-mapped BAR1 aperture for sub-microsecond GPU memory access without launching kernels. Read in `initGdrCopy()` — if enabled and `gdr_open()` fails (library not installed or `/dev/gdrdrv` missing), silently falls to disabled. When active: the receive side uses `gdr_copy_from_mapping()` to read small data directly from GPU FIFO instead of waiting for a DMA completion callback — eliminates PCIe round-trip latency for ≤8KB messages. Also used for FIFO writes (`GDRCOPY_FIFO_ENABLE`) and memory synchronization (`GDRCOPY_SYNC_ENABLE`). Requires the GDRCopy kernel module (`gdrdrv`) and userspace library (`libgdrapi.so`). | `0` | — | `src/init.cc` | 🔧 **IB**: Enable if GDRCopy is installed |
| `NCCL_GDRCOPY_FIFO_ENABLE` | Enable GDRCopy for FIFO | `1` | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_GDRCOPY_FLUSH_ENABLE` | Enable GDRCopy flush | `0` | — | `src/transport/net.cc` | ⚙️ Internal |
| `NCCL_GDRCOPY_SYNC_ENABLE` | Enable GDRCopy sync | `1` | — | `src/transport/net.cc` | ⚙️ Internal |
| `NCCL_DMABUF_ENABLE` | Enable GPU Direct RDMA via Linux dma-buf subsystem. The dma-buf path is the modern alternative to `nvidia-peermem` for GPU Direct RDMA. Read in `initNet()` — NCCL probes: (1) `ibv_reg_dmabuf_mr` exists in libibverbs, (2) `cuMemGetHandleForAddressRange()` available in CUDA driver (≥12.0), (3) kernel support (`/proc/driver/nvidia/params` for peermem). If any check fails, dma-buf is silently disabled and NCCL falls back to `nvidia-peermem.ko` (legacy GDR path). When dma-buf works: `ibv_reg_dmabuf_mr()` registers GPU memory with the NIC in a single call, versus the legacy `ibv_reg_mr()` + peermem callback chain. dma-buf provides better error handling and supports heterogeneous memory types. | `1` (auto-disabled without support) | 2.13 | `src/init.cc` | ⚙️ Leave enabled |

---

## 8. NVLink / NVLS / MNNVL

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_NVLS_ENABLE` | NVLink SHARP (NVSwitch multicast). **`0`**: NVLS transport disabled — AllReduce, ReduceScatter, AllGather use Ring/Tree algorithms only. **`1`**: NVLS required — if multicast group allocation via `cuMulticastCreate()` fails (wrong hardware, insufficient resources), communicator creation fails with an error. **`2` (default, try)**: attempt NVLS setup; on failure, silently fall back to Ring/Tree. In `nvlsInit()`: checks `cuMulticastSupported` (NVSwitch v3 + Hopper+), calls `cuMulticastCreate()` to create a multicast object spanning all local GPUs, then `cuMulticastAddDevice()` for each. The NVSwitch hardware performs in-switch reduction — AllReduce data travels GPU→NVSwitch→GPU in one hop (vs Ring's O(n) hops). NVLS typically provides 2-4× bandwidth improvement for AllReduce on 8-GPU DGX H100 nodes. Requires CUDA 12.1+ driver with multicast support. | `2` | 2.17 | `src/transport/nvls.cc` | ⚙️ Leave `2`. Only on NVSwitch v3 (NVLink4) + Hopper+ |
| `NCCL_NVLS_NCHANNELS` | Number of NVLS channels | auto (UNDEF) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_NVLS_CHUNKSIZE` | NVLS chunk size | `131072` (128 KiB) | — | `src/transport/nvls.cc` | ⚙️ Internal |
| `NCCL_MNNVL_ENABLE` | Multi-Node NVLink — extends NVLS across nodes via NVSwitch fabric. **`0`**: MNNVL disabled, inter-node uses IB/RoCE/TCP. **`1`**: MNNVL required — if prerequisites missing, communicator creation fails. **`2` (auto)**: attempt MNNVL; fall back if unsupported. **Prerequisites** (ALL must be true): (a) `cuMemSupport` enabled (`CUMEM_ENABLE` not `0`), (b) CUDA driver has IMEX (Inter-process Memory Exchange) channel support (`cuMulticastGetGranularity` with `CU_MEM_ALLOC_GRANULARITY_RECOMMENDED`), (c) NVIDIA Fabric Manager running and reporting `MULTINODE` clique type, (d) GPUs report `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` with fabric support. In `mnnvlInit()`: queries `fabricInfo` from driver, checks `cliqueId` assignment, then calls `cuMulticastCreate()` with `numDevices` spanning all nodes. Data path: GPU→NVSwitch→NVSwitch (inter-node fabric)→GPU, entirely bypassing PCIe/IB. On DGX SuperPOD with NVLink4 fabric, provides ~900 GB/s bisection bandwidth per node. | `2` | 2.21 | `src/init.cc` | ⚙️ Leave `2`. Requires IMEX domain + `CUMEM_ENABLE` |
| `NCCL_MNNVL_UUID` | Override MNNVL UUID (normally from Fabric Manager) | Auto | 2.25 | `src/init.cc` | ⚙️ Do not set unless debugging FM |
| `NCCL_MNNVL_CLIQUE_ID` | Override the MNNVL clique identifier for soft partitioning of NVSwitch fabric. Normally the clique ID comes from NVIDIA Fabric Manager (via `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_CLUSTER_ID)` or fabricInfo). **When set**: overrides the FM-assigned clique, allowing multiple NCCL communicators on the same physical fabric to be logically isolated. GPUs with the same clique ID form an MNNVL group; different clique IDs cannot share multicast objects. Use case: running multiple training jobs on the same DGX SuperPOD — each job sets a unique `MNNVL_CLIQUE_ID` so their NVLS multicast groups don't interfere. The clique ID is an integer. If two ranks have different clique IDs, MNNVL is disabled between them (they fall back to IB/RoCE). | Auto (from FM) | 2.25 | `src/init.cc` | 🔧 Set to partition MNNVL jobs |
| `NCCL_MNNVL_RAIL_PER_HOST` | Rails per host for MNNVL | `0` | — | `src/graph/search.cc` | ⚙️ Internal |
| `NCCL_MNNVL_SCATTER_NETS_ENABLE` | Scatter nets for MNNVL | `1` | — | `src/graph/search.cc` | ⚙️ Internal |

---

## 9. Algorithms, Protocols & Tuning

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_ALGO` | Algorithm allow/deny list. Parsed in `parseList()` with algorithms: `Ring`, `Tree`, `CollnetChain`, `CollnetDirect`, `NVLS`, `NVLSTree`, `PAT`. **Syntax (since 2.24)**: semicolon-separated per-function rules: `"AllReduce:Ring,Tree;AllGather:NVLS"`. `^` prefix excludes: `"^Tree"` disables Tree for all functions. Without per-function prefix, applies globally. In `ncclTopoTuneModel()`: for each `(collective, messageSize)` pair, NCCL evaluates all algorithm×protocol combos and picks the one with lowest estimated time. When you restrict algorithms, you exclude candidates from this selection — NCCL picks the best *remaining* option, never an excluded one. **Ring**: O(n) bandwidth-optimal for large messages, O(n) latency; **Tree**: O(log n) latency, lower bandwidth; **NVLS/NVLSTree**: NVSwitch in-switch reduction (if hardware available); **PAT**: Pairwise All-to-all Tree; **CollnetChain/CollnetDirect**: SHARP in-network reduction (requires CollNet plugin). | All available (auto) | 2.5 | `src/enqueue.cc`, `src/graph/tuning.cc` | 🚫 Don't set in production. 🔧 Debug: `^Tree` to disable Tree |
| `NCCL_PROTO` | Protocol allow/deny list. Same syntax as `NCCL_ALGO`. Protocols: **`Simple`**: standard DMA-based transfer — proxy thread on CPU side issues RDMA ops, GPU kernel writes/reads from shared buffers. Highest bandwidth (saturates NVLink/IB), highest latency per-op (~5-10µs overhead). **`LL` (Low-Latency)**: 8-byte data + 4-byte flag packed into one 16-byte unit (50% bandwidth efficiency). No CPU proxy involvement for intra-node — GPU kernel directly polls flag words. Sub-microsecond latency for small messages (≤16KB). **`LL128`**: 120 bytes data + 8 bytes flags per 128-byte cacheline (93.75% efficiency). Uses CUDA `__ldg` for reading and relies on 128-byte atomic visibility. Better bandwidth than LL for medium messages (16KB-512KB), still lower latency than Simple. `LL128` requires NVLink or specific IB configurations (not all PCIe topologies support it). Per-function rules (since 2.24): `"AllReduce:Simple;Broadcast:LL"`. `^LL128` disables LL128 globally. | All supported | 2.5 | `src/enqueue.cc`, `src/graph/tuning.cc` | 🚫 Don't set in production. 🔧 Debug: `^LL128` if data corruption suspected |
| `NCCL_BUFFSIZE` | Per-channel communication buffer size (bytes). Set in `initTransportsRank()` on `comm->buffSizes[NCCL_PROTO_SIMPLE]`. This is the **Simple protocol** ring buffer between each GPU pair per channel. **Memory cost**: `buffSize × nChannels × nPeers × 2 (send+recv)` of **GPU memory** per rank. Default 4 MiB × 32 channels × 2 peers × 2 = 512 MiB on an 8-GPU node. Larger buffers allow more pipelining (overlap of send/recv/compute) but consume proportionally more GPU memory. For memory-constrained training (large models), reducing to 1-2 MiB can free significant GPU memory at a 5-15% bandwidth cost. The LL and LL128 buffers have separate size params (`LL_BUFFSIZE`, `LL128_BUFFSIZE`). Minimum: `2 × NCCL_STEPS(8) × sliceSize`; values below this are silently clamped up. | `4194304` (4 MiB) | — | `src/init.cc` | 🚫 Only for memory-constrained or perf experiments |
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
| `NCCL_MAX_CTAS` | Maximum CTAs (Cooperative Thread Arrays = CUDA thread blocks) for NCCL kernels. Each CTA corresponds to one channel, so `MAX_CTAS` effectively caps the number of communication channels. Read in `initTransportsRank()` — overrides the topology-computed `maxChannels`. Also reads legacy `MAX_NCHANNELS` as fallback. **Impact**: more CTAs = more parallelism = higher bandwidth for large messages, but more SM occupancy stolen from compute kernels. On H100 with 132 SMs, 32 CTAs use ~24% of GPU. Reducing to 8 CTAs frees SMs for overlapped compute but reduces peak inter-GPU bandwidth by ~2×. For inference workloads with small messages, 4-8 CTAs may be optimal. NCCL auto-sizes based on topology (NVLink nodes default to ~32, PCIe-only to ~8). Valid range: 1-64 (clamped in `ncclTopoSetMaxChannels`). | Auto | 2.17 | `src/init.cc` | 🔧 Set to limit GPU resource usage (1-64) |
| `NCCL_MIN_CTAS` | Minimum CTAs to guarantee parallelism. Read alongside `MAX_CTAS` in `initTransportsRank()`. Forces NCCL to use at least N channels even if the topology search found fewer (e.g., PCIe-only systems default to few channels). The topology search in `ncclTopoCompute()` finds rings/trees respecting `minChannels` — if it can't build enough distinct paths, some channels may reuse the same physical path (reducing effective bandwidth). Useful when NCCL underestimates available parallelism. Also reads legacy `MIN_NCHANNELS`. Valid range: 1-`MAX_CTAS`. Setting `MIN_CTAS > MAX_CTAS` results in `MAX_CTAS` winning. | Auto | 2.17 | `src/init.cc` | 🔧 Set to guarantee parallelism (1-64) |
| `NCCL_CTA_POLICY` | CTA policy: `DEFAULT`/`0`, `EFFICIENCY`/`1`, `ZERO`/`2`. Pipe with `\|`. | `DEFAULT` | 2.29 (legacy 2.27) | `src/init.cc` | ⚙️ Leave default |
| `NCCL_UNPACK_DOUBLE_NCHANNELS` | Double channels for unpacking | `1` | — | `src/graph/connect.cc` | ⚙️ Internal |
| `NCCL_NCHANNELS_PER_NET_PEER` | Channels per network peer | auto (UNDEF) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_P2P_NET_CHUNKSIZE` | Chunk size for network-transported `ncclSend()`/`ncclRecv()` operations. Read in `initTransportsRank()` and stored in `comm->p2pNetChunkSize`. Each P2P transfer is split into chunks of this size — smaller chunks reduce latency-to-first-byte (pipelining starts sooner) but increase per-chunk overhead (proxy wakeups, RDMA post count). **Bandwidth-optimal**: large chunks (≥512KB) amortize per-chunk overhead; a single 1MB transfer with 128K chunks = 8 RDMA posts vs 2 posts with 512K. **Latency-optimal**: smaller chunks overlap computation earlier. For MoE all-to-all with many small expert-parallel messages (4-64KB), the default 128K is already larger than the message, so each transfer is exactly one chunk. Must be a power of 2. | `131072` (128K) | 2.14 | `src/init.cc` | ⚙️ Power of 2, next up: 262144 |
| `NCCL_P2P_NVL_CHUNKSIZE` | NVLink chunk size for P2P | `524288` (512 KiB) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_P2P_PCI_CHUNKSIZE` | PCI chunk size for P2P | `131072` (128 KiB) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_P2P_LL_THRESHOLD` | Max message size for LL protocol in P2P | `16384` | 2.14 | `src/enqueue.cc` | ⚙️ Leave default |
| `NCCL_ALLOC_P2P_NET_LL_BUFFERS` | Pre-allocate LL (Low-Latency) protocol buffers for ALL point-to-point net connections, not just those that auto-select LL. Normally, LL buffers are only allocated for connections where the tuning model predicts LL will be used. **When `1`**: every net transport connection gets LL buffers at init time. This eliminates the latency penalty when a small `ncclSend`/`ncclRecv` triggers LL protocol on a connection that didn't pre-allocate LL buffers (which would cause a synchronous buffer allocation mid-operation). **Memory cost**: `llBuffSize × nNetPeers × nChannels × 2` additional GPU memory. On large-scale all-to-all workloads (e.g., expert parallelism in MoE models) where every rank sends small messages to every other rank, this pre-allocation pays off by avoiding thousands of lazy buffer allocations. | `0` | 2.14 | `src/init.cc` | 🔧 Set `1` for latency-sensitive all-to-all at scale (high memory) |
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
| `NCCL_CUMEM_ENABLE` | Controls whether NCCL uses the CUDA Virtual Memory Management API (`cuMemCreate`/`cuMemMap`/`cuMemSetAccess`) for device memory allocation. **Auto-detection** (default since 2.19): `ncclCuMemEnable()` in `cudawrap.cc` runs a 4-check chain: (1) `cuMemCreate` symbol exists in driver, (2) driver version ≥ 12.0, (3) `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` is true, (4) a test `cuMemCreate()` + `cuMemRelease()` succeeds. If all pass, cuMem is enabled. **When `0`**: forces legacy `cudaMalloc`/`cudaFree` path. Disables MNNVL (which requires cuMem for multi-node multicast handles), buffer registration optimizations, and `P2P_CUMEM` transport variant. **When `1`**: forces cuMem usage; if the 4-check fails, communicator creation errors out. cuMem provides: (a) explicit physical/virtual separation for fine-grained sharing, (b) `cuMemExportToShareableHandle` for IPC without `cudaIpcGetMemHandle` limitations, (c) `cuMemSetAccess` for multi-GPU access grants without `cudaEnablePeerAccess`. | Auto-detect (since 2.19) | 2.18 | `src/misc/cudawrap.cc` | ⚙️ Leave auto. Required for MNNVL |
| `NCCL_CUMEM_HOST_ENABLE` | Use cuMem* for host memory | `1` if CUDA driver ≥ 12.6, runtime ≥ 12.2 (since 2.24) | 2.23 | `src/misc/cudawrap.cc` | ⚙️ Leave auto |
| `NCCL_GRAPH_REGISTER` | Auto-register buffers in CUDA Graphs (CollNet, P2P, single GPU/proc) | `1` | 2.11 | `src/enqueue.cc` | ⚙️ Leave enabled |
| `NCCL_LOCAL_REGISTER` | Enable explicit buffer registration via ncclCommRegister | `1` | 2.19 | `src/register/register.cc` | ⚙️ Leave enabled |
| `NCCL_LEGACY_CUDA_REGISTER` | Allow registration of legacy cudaMalloc buffers (can cause sync/hang!) | `0` | 2.24 | `src/transport/p2p.cc` | 🚫 Leave `0`. Use cuMem-based allocators instead |
| `NCCL_WIN_ENABLE` | Enable window memory registration | `1` | 2.27 | `src/init.cc` | ⚙️ Leave enabled |
| `NCCL_WIN_STRIDE` | Window stride | `-1` (auto) | — | `src/dev_runtime.cc` | ⚙️ Internal |
| `NCCL_MEM_SYNC_DOMAIN` | CUDA Memory Synchronization Domain for NCCL kernels (Hopper/sm90+ only). Controls the `syncDomain` field in `cudaLaunchAttribute` when launching NCCL kernels. **`0` (Default domain)**: NCCL kernels share the default memory ordering domain with compute kernels — `__threadfence_system()` and memory fences in NCCL can stall or be stalled by unrelated compute kernel memory operations. **`1` (Remote domain, default)**: NCCL kernels run in a separate synchronization domain — `__threadfence_system()` in NCCL only synchronizes with other Remote-domain operations, NOT with compute kernels in the Default domain. This eliminates interference: compute kernels' memory fences don't block NCCL's progress, and vice versa. On H100, switching from domain 0 to 1 can improve AllReduce latency by 10-30% during overlapped compute+communication. The domain isolation is a Hopper hardware feature; on pre-sm90 GPUs, this setting is silently ignored. | `1` (Remote) | 2.16 | `src/enqueue.cc` | ⚙️ Leave `1`. Eliminates compute/NCCL interference |

---

## 11. Kernel Launch & Threading

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_LAUNCH_MODE` | CUDA launch mode: `PARALLEL` or `GROUP` (deprecated) | `PARALLEL` | 2.1.0 | `src/init.cc` | ⚙️ Leave `PARALLEL`. GROUP deprecated in 2.9 |
| `NCCL_SET_STACK_SIZE` | Pre-configure CUDA kernel stack size to the maximum required across all NCCL kernels. Read in `ncclInitKernelsForDevice()`. **When `0` (default)**: CUDA runtime may dynamically reconfigure stack size between kernel launches. On some driver versions, this reconfiguration can trigger a device-wide memory reallocation that **blocks all GPU streams** — causing multi-second hangs visible as `cudaLaunchKernel` stalls. **When `1`**: at communicator init, NCCL calls `cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` for all kernels, then `cudaDeviceSetLimit(cudaLimitStackSize, maxStackSize)` to the maximum stack across all NCCL kernel variants. This one-time cost at init prevents runtime reconfigurations. The stack size is computed from `ncclShmem` struct size + per-kernel requirements. Set `1` if you observe periodic multi-second latency spikes during NCCL operations, especially when mixing different collective types. | `0` | 2.9 | `src/init.cc` | 🔧 Set `1` if experiencing hangs from CUDA memory reconfiguration |
| `NCCL_SET_CPU_STACK_SIZE` | Set CPU stack size (for NCCL's CPU threads) | `1` | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_SET_THREAD_NAME` | Assign human-readable names to NCCL's internal CPU threads. Read in `ncclSetThreadName()` via `ncclDebugSetThreadName()`. When `1`: uses `pthread_setname_np()` to set names like `NCCL <rank> proxy`, `NCCL <rank> bootstrap`, `NCCL <rank> progress`, `NCCL <rank> RAS`. Thread names appear in `ps -T`, `htop`, `perf record`, NVIDIA Nsight Systems timelines, and `gdb info threads`. **Overhead**: single `pthread_setname_np` call per thread creation — negligible. **Debugging value**: when NCCL hangs, `gdb -p <pid>` + `info threads` immediately shows which NCCL component is stuck (proxy? bootstrap? RAS?) vs generic `pthread` names. In NSight Systems, named threads make the NCCL communication timeline immediately identifiable among application threads. | `0` | 2.12 | `src/debug.cc` | 🔧 Set `1` for debugging/profiling (low overhead) |
| `NCCL_CGA_CLUSTER_SIZE` | CUDA CGA cluster size (sm90+). Overrides `cgaClusterSize` config. | Auto | 2.16 | `src/init.cc` | ⚙️ Leave auto. 0-8 valid |
| `NCCL_GRAPH_MIXING_SUPPORT` | Support multiple outstanding NCCL operations launched from parallel CUDA graphs. Read via `ncclParamGraphMixingSupport()` in `initTransportsRank()`, stored in `comm->graphUsageMode`. **When `1` (default)**: `graphUsageMode = ncclGraphUsageMix` — NCCL supports being called from different graph-captured streams simultaneously. This requires additional synchronization in the launch path: device-side mutexes (`graphHead`/`graphTail` counters in `ncclShmem`) serialize access to shared resources across graph instances. **When `0`**: `graphUsageMode = ncclGraphUsageStreamOrdered` — NCCL assumes all operations are stream-ordered (no parallel graph launches). Removes the graph-mixing synchronization overhead. Set `0` if (a) you never use CUDA graphs with NCCL, or (b) all NCCL calls are in a single graph. Set `1` (keep default) if multiple graphs containing NCCL ops may execute concurrently (e.g., pipeline parallelism with overlapping forward/backward graph instances). | `1` | 2.13 | `src/init.cc` | 🔧 Set `0` if experiencing launch hangs with multi-graph usage |
| `NCCL_LAUNCH_ORDER_IMPLICIT` | Implicitly serialize NCCL operations from different communicators on the same device by host-program order. Read in `ncclEnqueueCheck()`. **When `0` (default)**: operations from different communicators launched on the same GPU execute in GPU stream order only — if two communicators' operations are on the same stream, CUDA serializes them; if on different streams, they may execute concurrently. This can cause **deadlocks** when two communicators (e.g., data-parallel + tensor-parallel) both need the same GPU but wait for each other's operations. **When `1`**: NCCL inserts a `cudaStreamWaitEvent` between operations from different communicators, enforcing host-program launch order on the device. Operation A launched before operation B (in wall-clock host time) will complete before B starts on-device, even across communicators and streams. This prevents deadlocks at the cost of serializing overlappable operations. The ordering is enforced via `comm->lastOpStream` / `comm->lastOpEvent` tracking in `ncclEnqueueCheck()`. | `0` | 2.26 | `src/enqueue.cc` | 🔧 Set `1` to prevent deadlocks between communicators |
| `NCCL_LAUNCH_RACE_FATAL` | Catch host threads racing to launch on same device → fatal error | `1` | 2.26 | `src/misc/strongstream.cc` | ⚙️ Leave enabled |
| `NCCL_COMM_BLOCKING` | Override per-communicator blocking mode. Normally set via `ncclConfig_t::blocking` in `ncclCommInitRankConfig()`. **When `1` (blocking)**: all NCCL collective calls block until the operation completes on all ranks. `ncclGroupEnd()` waits for all operations in the group. Errors are returned synchronously from the call that triggered them. **When `0` (nonblocking)**: collectives return immediately after enqueuing. Caller must poll `ncclCommGetAsyncError()` or use `ncclCommFinalize()` + `ncclCommDestroy()` to detect errors. Operations are only guaranteed complete after a successful `cudaStreamSynchronize` on the NCCL stream. **When unset**: per-communicator config controls behavior (default is blocking). This env var overrides ALL communicators' settings — useful for forcing nonblocking mode in frameworks that don't expose the config API. Nonblocking mode enables true overlap of NCCL init/collectives with compute but requires careful error handling. | Undefined (per config) | 2.14 | `src/init.cc` | ⚙️ Leave undefined; set per-communicator via `ncclConfig_t` |
| `NCCL_CHECKS_DISABLE` | Disable argument checks on collective calls (deprecated 2.2.12) | `0` | 2.0.5 | — | 🚫 Deprecated |
| `NCCL_CHECK_POINTERS` | Enable CUDA memory pointer validation on collective call arguments. **When `1`**: before each collective, NCCL calls `cuPointerGetAttribute()` on the send/recv buffer pointers to verify they are valid CUDA device pointers. Catches common errors: host pointers passed where device pointers expected, freed memory, wrong-device pointers. Adds ~2-5µs per collective call (one `cuPointerGetAttribute` per buffer). **When `0` (default)**: no validation — invalid pointers cause silent corruption or `cudaErrorIllegalAddress` later. Production code should leave disabled (frameworks validate at higher levels); enable during development and debugging. | `0` | 2.2.12 | — | 🔧 Set `1` during development |
| `NCCL_L1_SHARED_MEMORY_CARVEOUT` | L1/shared memory carveout for NCCL kernels | `0` | — | `src/enqueue.cc` | ⚙️ Internal |
| `NCCL_GROUP_CUDA_STREAM` | Group CUDA stream setting | `0` (CUDA ≥9.2) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_GRAPH_HELPER_DISABLE` | Disable graph helper | `0` | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_WORK_FIFO_BYTES` | Work FIFO size | `1048576` (1 MiB) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_WORK_ARGS_BYTES` | Work args size | `INT64_MAX` (unlimited) | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_ENABLE_VERSION_CHECK` | Enable device runtime version check | `1` | — | `src/dev_runtime.cc` | ⚙️ Internal |
| `NCCL_CREATE_THREAD_CONTEXT` | Create thread context | `0` | — | `src/proxy.cc` | ⚙️ Internal |
| `NCCL_PROXY_CPUSET` | Pin NCCL proxy threads to specific CPU cores. Format: comma-separated ranges, e.g., `"0-3,8-11"`. Read via `ncclGetEnv("NCCL_PROXY_CPUSET")` in `ncclProxyCreate()`. The proxy thread handles all CPU-side network operations (IB verbs posts, socket I/O, buffer management) for a communicator. **When unset**: proxy threads inherit the process's affinity mask; on NUMA systems, they may land on cores far from the GPU or NIC, adding memory access latency to every RDMA operation. **When set**: `sched_setaffinity()` pins proxy threads to the specified cores. Best practice: pin to cores on the same NUMA node as the NIC (check `numactl -H` and `cat /sys/class/infiniband/mlx5_*/device/numa_node`). With 8 GPUs and 8 NICs, pin each proxy to 1-2 cores near its NIC. Format parsed by `ncclCpusetParse()`: ranges (`0-3`), singles (`5`), comma-separated. Invalid format silently falls back to no pinning. | — | — | `src/proxy.cc` | 🔧 Set for NUMA-aware proxy pinning |
| `NCCL_PROXY_APPEND_BATCH_SIZE` | Proxy append batch size | `16` | — | `src/proxy.cc` | ⚙️ Internal |
| `NCCL_PROXY_DUMP_SIGNAL` | Dump proxy state on signal. -1=off, otherwise signal number | `-1` (disabled) | — | `src/proxy.cc` | ⚙️ Internal debug |
| `NCCL_PROGRESS_APPENDOP_FREQ` | Progress appendop frequency | `8` | — | `src/proxy.cc` | ⚙️ Internal |
| `NCCL_NVTX_DISABLE` | Disable NVTX (NVIDIA Tools Extension) range annotations. Read in `initNvtxRegisteredEnums()`. **When `0` (default)**: NCCL pushes NVTX range markers around collective operations (`ncclAllReduce`, `ncclSend`, etc.) with metadata (message size, data type, algorithm, protocol). These appear in NSight Systems / NSight Compute timelines as named colored bars on the NCCL thread rows. **When `1`**: NVTX registration is skipped; no markers are emitted. NVTX overhead when no profiler is attached is near-zero (the NVTX API checks an injection flag before doing any work). When a profiler IS attached, each `nvtxRangePush`/`nvtxRangePop` adds ~100-500ns. The only reason to disable is if (a) you're profiling with a tool that doesn't handle NVTX well, or (b) you're measuring sub-microsecond overheads and want to eliminate any noise. | `0` | — | `src/init_nvtx.cc` | ⚙️ Set `1` if NVTX overhead is unwanted |
| `NCCL_NVB_PRECONNECT` | Preconnect NVB peers | `1` | — | `src/init.cc` | ⚙️ Internal |
| `NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE` | NVLink utilization-centric scheduling | `0` (effective; UNDEF→`0`) | — | `src/init.cc` | ⚙️ Internal |

---

## 12. CollNet & Plugins

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_COLLNET_ENABLE` | Enable in-network collective reduction via CollNet plugin (NVIDIA SHARP or similar). Read in `initTransportsRank()`. **When `0` (default)**: CollNet transport is not initialized — all collectives use Ring/Tree/NVLS algorithms. **When `1`**: NCCL attempts to initialize the CollNet transport by calling `ncclCollnetInit()`, which loads the CollNet plugin (part of `NET_PLUGIN`). If the plugin reports `ncclCollNetSupport()` for the topology, NCCL builds CollnetChain and CollnetDirect channels in addition to Ring/Tree. The tuning model then considers CollNet algorithms (`NCCL_ALGO_COLLNET_CHAIN`, `NCCL_ALGO_COLLNET_DIRECT`) in the cost model. **SHARP in-network reduction**: the network switch hardware performs the AllReduce reduction — data goes GPU→NIC→switch(reduce)→NIC→GPU in one round-trip, vs Ring's per-rank latency. Provides 2-4× latency improvement for large AllReduce. Requires: (a) Mellanox SHARP-capable switches (QM87xx+), (b) `sharp_coll` library, (c) SHARP aggregation manager running. `COLLNET_NODE_THRESHOLD` controls the minimum node count to activate. | `0` | 2.6 | `src/init.cc` | 🔧 **IB with SHARP**: Set `1` |
| `NCCL_COLLNET_NODE_THRESHOLD` | Minimum number of nodes to activate CollNet. Read in `initTransportsRank()` after CollNet init. If `comm->nNodes < threshold`, CollNet channels are not built even if `COLLNET_ENABLE=1` and the plugin supports it. **Rationale**: SHARP in-network reduction has a fixed setup cost (aggregation tree construction on switches) that only pays off for multi-node jobs. Single-node AllReduce via NVLink/NVLS is already faster than SHARP's switch-level reduction. Default `2` means CollNet activates for any multi-node job. Set higher (e.g., `4`, `8`) if SHARP aggregation manager struggles with frequent small-job setups. | `2` | 2.9.9 | `src/init.cc` | ⚙️ Leave default |
| `NCCL_NET` | Force a specific network transport. Read in `ncclNetInit()`. **When unset (auto)**: NCCL tries transports in order: (1) external NET_PLUGIN if loaded, (2) built-in IB/RoCE (`ncclNetIb`), (3) built-in Socket (`ncclNetSocket`). The first transport whose `init()` succeeds is used. **When set to `"IB"`**: forces the built-in IB transport; if `ibv_get_device_list()` returns no devices, communicator creation fails. **When set to `"Socket"`**: forces TCP sockets; skips IB entirely. **When set to a plugin name** (e.g., `"EFA"`, `"GDRCopy"`): forces that specific plugin (must match the name reported by the plugin's `ncclNet_t::name` field). Useful for debugging: `Socket` isolates whether a problem is network-related vs IB-specific. Also useful in cloud environments (AWS EFA) where the plugin name may differ from generic transport names. | Auto | 2.10 | `src/init.cc` | 🔧 Set `Socket` to force TCP. Otherwise leave auto |

(-; discarded as we don't build plugins)
@Note(for LLM): `NCCL_NET_PLUGIN=aws-ofi-nccl` will be set but disabled for initial testing
@begin:
| `NCCL_NET_PLUGIN` | Network plugin discovery. Read in `ncclPluginLoad()`. **Discovery chain**: (1) if set to `"none"`: no external plugin loaded, only built-in IB/Socket. (2) if set to a path (contains `/`): `dlopen()` that exact path. (3) if set to a name: try `dlopen("libnccl-net-<name>.so")`, then `dlopen("libnccl-net.so")`, then `dlopen("<name>")`. (4) if unset: try `dlopen("libnccl-net.so")` from `LD_LIBRARY_PATH`. The plugin must export `ncclNet_v9` or earlier versioned symbol (NCCL tries v9, v8, v7, v6, v5 in order). On AWS: `NCCL_NET_PLUGIN=aws-ofi-nccl` loads `libnccl-net-aws-ofi-nccl.so` which provides EFA (Elastic Fabric Adapter) transport. On Azure: `NCCL_NET_PLUGIN=msccl-net` loads the MSCCL network plugin. `STATIC_PLUGIN` checks a built-in symbol (`ncclNetPlugin_v9`) first, used when the plugin is statically linked into libnccl. | `libnccl-net.so` | 2.11 | `src/plugin/net.cc` | 🔧 Set plugin name or `none` |
| `NCCL_TUNER_PLUGIN` | Tuner plugin library name/suffix. Falls back to net plugin, then internal. | `libnccl-tuner.so` | — | `src/plugin/tuner.cc` | ⚙️ Leave default unless using custom tuner |
| `NCCL_PROFILER_PLUGIN` | Profiler plugin. `STATIC_PLUGIN` searches binary. | `libnccl-profiler.so` | — | `src/plugin/profiler.cc` | 🔧 Set for profiling |
| `NCCL_ENV_PLUGIN` | Environment plugin. "none" to disable. | `libnccl-env.so` | 2.28 | `src/plugin/env.cc` | ⚙️ Leave default |
@end

---

## 13. GIN (GPU-Initiated Networking)

New in v2.29. CUDA device-side networking (GPU kernels directly initiate RDMA).

| Env Var | Description | Default | Since | Source | Recommendation |
|---------|-------------|---------|-------|--------|----------------|
| `NCCL_GIN_ENABLE` | Enable GPU-Initiated Networking (GIN). New in v2.29. When enabled, NCCL can use a **device-side networking path** where GPU kernels directly initiate RDMA operations without CPU proxy involvement. Read in `ginInit()` (`src/gin/gin_host.cc`). **When `1` (default)**: GIN initialization attempts to detect compatible hardware. Currently requires DOCA GPUNetIO hardware (NVIDIA BlueField-3 DPU with GPU-addressable NIC queues). `ginDetectType()` probes available backends: `GIN_GDAKI` (GPU Direct Async Kernel-Initiated via DOCA). If no compatible hardware found, GIN silently falls back to standard CPU-proxy networking. **When `0`**: GIN is entirely disabled; all network operations go through the CPU proxy thread. GIN eliminates proxy thread latency (~1-3µs per operation) and CPU overhead, but requires specific hardware. The GIN kernel code runs in the `nccl_colldevice` target (device compilation) and links against DOCA GPUNetIO device headers (`.cuh` files). | `1` | 2.29 | `src/gin/gin_host.cc` | ⚙️ Experimental. Requires DOCA GPUNetIO hardware |
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
| `NCCL_RAS_ENABLE` | Enable the RAS (Reliability, Availability, Serviceability) subsystem. Read in `bootstrapCreateRoot()`. **When `1` (default)**: during bootstrap, NCCL creates a RAS thread via `ncclRasCreate()`. This thread: (a) establishes a **full-mesh TCP overlay** among all ranks (each rank connects to all others via RAS-specific sockets), (b) opens a **client listener** on the address specified by `RAS_ADDR` (default `localhost:28028`), (c) monitors peer liveness via periodic heartbeat messages, (d) collects topology, configuration, and error state from all ranks. The `ncclras` CLI tool connects to this listener to query cluster health. **When `0`**: no RAS thread, no mesh, no listener. Saves ~1 TCP connection per rank² (significant at scale: 1024 ranks = ~1M connections) and eliminates RAS heartbeat traffic. Disable for ephemeral jobs or environments where the overhead matters. RAS is purely diagnostic — disabling has **zero effect** on collective performance or correctness. | `1` | 2.24 | `src/bootstrap.cc` | ⚙️ Leave enabled. Use `ncclras` CLI to query |
| `NCCL_RAS_ADDR` | RAS client listener socket address. Format: `host:port` or just `port`. Read in `rasClientSocketInit()` (`src/ras/client_support.cc`) using `ncclSocketInit()`. **Default `localhost:28028`**: the RAS listener binds to loopback on port 28028. The `ncclras` CLI connects to this address. **When set to `0.0.0.0:PORT`**: binds on all interfaces, allowing remote `ncclras` queries across the network (security consideration: no authentication). **When set to just a port number** (e.g., `29029`): binds to loopback on that port. **Multi-job on same host**: if two NCCL jobs run on the same node with the same RAS port, the second job's `bind()` fails and RAS is disabled for that job (logged as WARN). Set different ports for co-located jobs: `NCCL_RAS_ADDR=28029` for job 2. The listener socket is non-blocking; the RAS thread uses `poll()` to multiplex client connections with the rank-mesh heartbeat loop. | `localhost:28028` | 2.24 | `src/ras/client_support.cc` | 🔧 Set different ports for co-located NCCL jobs |
| `NCCL_RAS_TIMEOUT_FACTOR` | Multiplier for all RAS internal timeouts (heartbeat, peer detection, client query response). Read in `rasInit()`. **When `1` (default)**: default timeouts apply (e.g., heartbeat interval ~5s, peer timeout ~30s, client query timeout ~10s). **When `N`**: all timeouts multiply by N. Set `2` or `3` when running under heavy debugging overhead (GDB, Valgrind, sanitizers) where normal timeouts would trigger false-positive peer-death detections. The heartbeat loop in `rasMainLoop()` checks `lastHeartbeat + (heartbeatTimeout * timeoutFactor)` before declaring a peer dead. Setting too high (e.g., `10`) means genuine failures take 10× longer to detect. | `1` | 2.24 | `src/ras/ras.cc` | 🔧 Increase under heavy debugging/tracing overhead |

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
(see `native/cuda_toolkit`)  | `CUDA_LAUNCH_BLOCKING` | CUDA global synchronous launch (from CUDA runtime) | `src/misc/cudawrap.cc` |
(see `native/cuda_toolkit`) | `CUDA_HOME` | CUDA installation path (build-time - primarly in Makefile and for Python bindings) | Build scripts |

(-; discarded as not built / used for now)
@begin:
| `SLURM_JOB_ID` / `SLURM_JOBID` | Slurm job identification | `src/` (for bootstrap) |
| `SLURM_JOB_NAME` | Slurm job name | `src/` |
| `DOCA_GPUNETIO_LOG` | DOCA GPUNetIO logging level | `src/transport/net_ib/gdaki/doca-gpunetio/` |
| `DOCA_GPUNETIO_DISABLE_GDRCOPY` | Disable GDRCopy in DOCA | `src/transport/net_ib/gdaki/doca-gpunetio/` |
@end:

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

## 22. Interactions & Shadowing

Variable interactions bind harder than any individual default. `NCCL_IB_DISABLE=1` makes 20 other `NCCL_IB_*` vars no-ops. `NCCL_P2P_DISABLE=1` shadows the entire `NCCL_P2P_*` and NVLS subsystems. These edges are load-bearing when writing conf files: setting `IB_DISABLE=1` on g5g means we should NOT add any `NCCL_IB_*` lines to that conf (they'd be dead weight and confuse future readers).

The full edge-list lives in [__deleteme/NCCL_ENV_VARS_INTERACTIONS.tsv](NCCL_ENV_VARS_INTERACTIONS.tsv) (columns: `source_var, source_value, relation, target_var, target_effect, scenario_filter`). Relations:
- **shadows** — setting source makes target's value ignored/unread
- **requires** — setting source to the value only works if target is also set a specific way
- **conflicts** — source + target combination is invalid or silently degrades
- **enables** — source being true is a precondition for target's path to run

Highest-impact edges (summarized):

| Source = Value | Relation | Targets affected | Effect |
|---|---|---|---|
| `NCCL_IB_DISABLE=1` | shadows | all 20+ `NCCL_IB_*` vars | IB device enum/QP/MR skipped; those vars are not parsed |
| `NCCL_NET=Socket` | shadows | `NCCL_NET_PLUGIN`, `NCCL_DMABUF_ENABLE`, all GDR vars | plugin chain bypassed; no RDMA path exists |
| `NCCL_P2P_DISABLE=1` | shadows | all `NCCL_P2P_*`, NVLS subsystem | `P2P_LEVEL` forced to `PATH_LOC` |
| `NCCL_NVLS_ENABLE=0` | conflicts | `NCCL_ALGO=NVLS*` entries | tuning model silently picks next algo |
| `NCCL_MNNVL_ENABLE=1` | requires | `NCCL_CUMEM_ENABLE`≠0, Fabric Manager | init fails without both |
| `NCCL_CUMEM_ENABLE=0` | conflicts | `NCCL_MNNVL_ENABLE`, `NCCL_RUNTIME_CONNECT` | MNNVL cannot init; runtimeConn forced false |
| `NCCL_GIN_ENABLE=0` | shadows | all `NCCL_GIN_*` vars | GIN init skipped |
| `NCCL_RAS_ENABLE=0` | shadows | `NCCL_RAS_ADDR`, `NCCL_RAS_TIMEOUT_FACTOR` | RAS thread never spawned |
| `NCCL_DEBUG` < INFO | shadows | `NCCL_DEBUG_SUBSYS`, `NCCL_DEBUG_FILE` | subsystem filter is INFO-gated; file only opened if level > VERSION |
| `$NCCL_CONF_FILE` set | shadows | `~/.nccl.conf`, `/usr/local/etc/nccl.conf` | first-match-wins (EUGO patch) |
| shell env var (any) | shadows | any conf-file same-name var | `setenv(name,value,0)` — shell always wins |

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
# 1 -> Grace and GB10, 0 -> Others # @TODO: implement that to be dynamic
if [[ -v EUGO_NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE ]]; then
    export NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE="${EUGO_NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE}"
else
    export NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE=0
fi
```


# @TODO: move to slava's docs:
#

