# NCCL Env-Var Shortlist — per scenario

> **Pivot of** [NCCL_ENV_VARS_MATRIX.tsv](NCCL_ENV_VARS_MATRIX.tsv). Rationale is one line per row; full descriptions live in [NCCL_ENV_VARS.md](NCCL_ENV_VARS.md).
>
> **Columns**:
> - **Must set** — conf file must carry this line for correctness or guaranteed behavior.
> - **Must NOT set** — override to the listed value is dangerous or deprecated; keep unset or leave at default.
> - **Careful** — default surprises in some way; know what it does before overriding.

---

## Cross-scenario (applies to all three)

### Must NOT set
- `NCCL_LEGACY_CUDA_REGISTER=1` — can cause sync/hang; use cuMem-based allocators instead. §10.
- `NCCL_CHECKS_DISABLE=1` — deprecated in 2.2.12, ignored. §11.
- `NCCL_LAUNCH_MODE=GROUP` — deprecated in 2.9, kernel-launch semantics removed. §11.
- `NCCL_NTHREADS` outside {64, 128, 256, 512} — only those values are valid thread counts. §9.
- Any `NCCL_IB_CUDA_SUPPORT` — removed in 2.4, use `NCCL_NET_GDR_LEVEL`. §2.
- `NCCL_IB_GDR_LEVEL` — no source reader in v2.29; legacy alias is gone. §7.

### Careful (defaults that surprise)
- `NCCL_SOCKET_POLL_TIMEOUT_MSEC=0` (default) — burns 100% CPU in tight send/recv loop. Set `100` on CPU-constrained hosts. §1.
- `NCCL_COMM_ID` — if set, cannot be used with `ncclCommGrow()` or multi-root init. §3.
- `NCCL_CONF_FILE` — path truncated at 1023 bytes; verify full path fits. [src/misc/param.cc:40](../src/misc/param.cc#L40).
- Shell env vars always override conf file (`setenv(name, value, 0)`). If ops set `NCCL_DEBUG=INFO` in a systemd unit, the conf's `WARN` is invisible. §preamble.
- `NCCL_DEBUG=TRACE` acquires a global mutex per log call — devastates latency-sensitive collectives. §4.
- `NCCL_SET_STACK_SIZE=0` (default) — periodic CUDA stack reconfig can cause multi-second stream stalls; set `1` if you see this. §11.
- `NCCL_GRAPH_MIXING_SUPPORT=1` (default) — adds device-side mutex overhead even for single-graph users. Set `0` if not using parallel CUDA graphs. §11.
- `NCCL_LAUNCH_ORDER_IMPLICIT=0` (default) — can deadlock when multiple communicators share a device (e.g., data-parallel + tensor-parallel). Set `1` to insert cross-communicator stream waits. §11.

---

## `grace_gb10`

### Must set (in conf file)
- `NCCL_DEBUG=WARN` — baseline production logging.
- `NCCL_SET_THREAD_NAME=1` — profiling hygiene, near-zero overhead.

### Templated (populate at deployment — DO set, but value is site-specific)
- `NCCL_SOCKET_IFNAME` — bootstrap interface (e.g., `=ibp1s0` or `^docker0,^lo`).
- `NCCL_IB_HCA` — HCA filter for 2-node variant (e.g., `=mlx5_0:1,mlx5_1:1`).
- `NCCL_TOPO_FILE` — hand-curated XML if auto-detect is wrong for the rack. (Outstanding @TODO — embed virtualTopology.xml in AMI/container.)

### Must NOT set
- `NCCL_IB_DISABLE=1` — only if 1-node variant. For the 2-node default, leave at `0` (default) so ConnectX IB engages.
- `NCCL_NET=Socket` — would bypass IB on 2-node and tank inter-node bandwidth.
- `NCCL_NVLS_ENABLE=0` — Hopper+NVSwitch is available; disabling NVLS loses 2-4× AllReduce bandwidth.
- `NCCL_MNNVL_ENABLE=0` — leave at `2` (auto) so MNNVL engages if Fabric Manager reports MULTINODE clique.
- `NCCL_CUMEM_ENABLE=0` — would disable MNNVL path and `RUNTIME_CONNECT` optimizations.

### Careful
- `NCCL_CROSS_NIC` — default `2` (auto). On rail-optimized fabrics, `0` may improve locality; on fat-tree, `1` may improve aggregate bandwidth. Profile before changing.
- `NCCL_PXN_C2C=1` (default in 2.28+) — keep the default; earlier versions had `0` which loses C2C-PXN on Grace.
- `NCCL_IB_TIMEOUT=20` (default, ~4.3s) — increase to `22` (~17s) for very large fabrics with many switch hops.
- `NCCL_IB_ADAPTIVE_ROUTING=-2` (auto: on IB, off on RoCE) — correct default; don't override unless fabric doesn't support AR.

---

## `g5g`

### Must set (in conf file)
- `NCCL_DEBUG=WARN`
- `NCCL_IB_DISABLE=1` — no IB hardware; built-in IB transport would fail at enum. Shadows all `NCCL_IB_*`.
- `NCCL_NET=Socket` — force TCP; redundant safety with IB_DISABLE.
- `NCCL_NVLS_ENABLE=0` — T4G is Turing (compute 7.5), no NVSwitch, so multicast probe is guaranteed to fail; skip it.
- `NCCL_MNNVL_ENABLE=0` — single-node, no NVLink fabric.
- `NCCL_SET_THREAD_NAME=1`

### Templated
- `NCCL_SOCKET_IFNAME` — data-plane interface (e.g., `=eth0`).

### Must NOT set
- Any `NCCL_IB_*` — shadowed by `IB_DISABLE=1`; adding them creates confusing dead weight in the conf.
- Any `NCCL_NVLS_*` — shadowed by `NVLS_ENABLE=0`.
- Any `NCCL_MNNVL_*` — shadowed by `MNNVL_ENABLE=0`.
- Any `NCCL_COLLNET_*` — no SHARP hardware.
- Any `NCCL_GIN_*` — no DOCA GPUNetIO / BlueField-3.
- `NCCL_NET_GDR_LEVEL` / `NCCL_DMABUF_ENABLE` — no RDMA to benefit from; NET=Socket shadows these.
- `NCCL_P2P_DISABLE=1` — would disable intra-node P2P which T4G can still do via PCIe; defeats the whole point.

### Careful
- `NCCL_SOCKET_NTHREADS` / `NCCL_NSOCKS_PERTHREAD` — auto-detect recognizes AWS ENA vendor and sets 2×8. Override only if profiling shows TCP is the bottleneck. Product must stay ≤ 64.
- `NCCL_P2P_LEVEL` — default auto (PXB). T4G GPUs on g5g share a PCIe complex; auto should work.
- `NCCL_BUFFSIZE=4MiB` (default) × 32 channels × 2 peers × 2 = 512 MiB per rank. On 16GB T4G, this is 3% — not alarming but can trim to 2 MiB if memory-pressured.

---

## `aws_efa`

### Must set (in conf file)
- `NCCL_DEBUG=WARN`
- `NCCL_IB_DISABLE=1` — EFA *is* visible via libibverbs (rdma-core `efa` provider), but only supports SRD QPs; NCCL's net_ib requires RC + RDMA_WRITE. Skip net_ib so the aws-ofi-nccl plugin (libfabric fi_efa) takes over.
- `NCCL_SET_THREAD_NAME=1`

### Templated
- `NCCL_SOCKET_IFNAME` — bootstrap interface (e.g., `=ens5`).
- `NCCL_NET_PLUGIN` — leave unset if aws-ofi-nccl ships as `libnccl-net.so` (NCCL's default search finds it). Set explicitly if coexisting plugins require disambiguation.

### Must NOT set
- Any `NCCL_IB_*` — shadowed by `IB_DISABLE=1`.
- Any `NCCL_COLLNET_*` — no in-network reduction on EFA.
- Any `NCCL_GIN_*` — no DOCA GPUNetIO.
- `FI_*` / `OFI_NCCL_*` in nccl.conf — these are **plugin-side** env vars, not NCCL-side. Conf loader parses all lines via `setenv`, but the plugin's own init reads them from the environment, so technically setting them in nccl.conf would work. DO NOT mix them in to keep plugin config separate from libnccl config.
- `NCCL_NVLS_ENABLE=0` on Hopper+ — leave at default `2` (auto) so intra-node NVLink SHARP still engages.
- `NCCL_MNNVL_ENABLE=0` — **INCORRECT for NVL72 racks**. See "Careful" below; leave at default=2 (auto) so it engages on p6e-gb200 and silently disables on p4d/p5.

### Careful
- `NCCL_MNNVL_ENABLE` — leave at default `2` (auto). On **NVL72** instances (EC2 p6e-gb200, GB200 Blackwell), 72 GPUs × 9 NVSwitches form a rack-scale NVLink fabric; Fabric Manager reports `MULTINODE` clique → auto-probe engages MNNVL and intra-rack collectives route at ~1.8 TB/s/GPU over NVLink instead of EFA. On **non-NVL72** (p4d/p5/p5e), FM doesn't report MULTINODE → auto-probe silently disables MNNVL; aws-ofi-nccl handles everything. Pinning `=0` breaks the NVL72 case. If you want fail-loud for a known-NVL72 deployment, set `=1` (strict) instead.
- `NCCL_DMABUF_ENABLE=1` (default) — correct for EFA+H100/B100 GDR. Don't disable unless plugin probe fails.
- `NCCL_CROSS_NIC` — EFA nodes have multiple EFA devices per node (e.g., 4 on p4d, 32 on p5). Auto default `2` handles rail routing; override only if profiling shows imbalance.
- `NCCL_OOB_NET_ENABLE=0` (default) — keeps bootstrap on TCP sockets. Safe for EFA deployments; the aws-ofi-nccl plugin isn't optimized as an OOB transport.
- `NCCL_ALGO` / `NCCL_PROTO` — do not restrict in production. Tuning model already knows EFA's latency/bandwidth characteristics.

---

## Deprecated / removed (ignored by v2.29 — never set)

| Var | Since | Replacement / notes |
|---|---|---|
| `NCCL_IB_CUDA_SUPPORT` | removed 2.4 | `NCCL_NET_GDR_LEVEL` |
| `NCCL_IB_GDR_LEVEL` | no reader found | use `NCCL_NET_GDR_LEVEL` |
| `NCCL_CHECKS_DISABLE` | deprecated 2.2.12 | no longer respected |
| `NCCL_LAUNCH_MODE=GROUP` | deprecated 2.9 | PARALLEL is the only supported mode |
| `NCCL_LL_THRESHOLD` | removed 2.5 | controlled by tuning model |
| `NCCL_TREE_THRESHOLD` | removed 2.5 | controlled by tuning model |
| `NCCL_SINGLE_RING_THRESHOLD` | removed 2.3 | controlled by tuning model |
| `NCCL_MAX_NRINGS` / `NCCL_MIN_NRINGS` | superseded 2.5 | use `NCCL_MAX_CTAS` / `NCCL_MIN_CTAS` |
| `NCCL_MAX_NCHANNELS` / `NCCL_MIN_NCHANNELS` | superseded 2.17 | use `NCCL_MAX_CTAS` / `NCCL_MIN_CTAS` |
| `NCCL_NVML_DIRECT` | compile-time only | `#define` in nvmlwrap.h, not a runtime var |
| `NCCL_USE_CMAKE` | build-time only | honored by `src/device/generate.py`, not libnccl |
