1. Narrow down and optimize @__deleteme/stage_2/nccl.conf.aws_efa to for AWS p6e UltraServers (9x and 18x nodes / trays in rack) and UltraClusters (communicating w/ each other via EFA). Don't focus on other instance types in this configuration so we can squeeze everything from that setup.
2. Adjust the style of @__deleteme/stage_2/NCCL_ENV_VARS.md  to match these examples:
    1. Add to our configuration: ```
        (+) RAY_object_manager_max_bytes_in_flight
        1. Default value: `2 GB`
        2. Type: uint64
        3. Read / Write: R
        4. Recommended value (for HPC): **`8–16 GB`**
        5. Mechanism: **Process: Raylet.**
        6. Notes: Configured in `raylet/main.cc` as `object_manager_config.max_bytes_in_flight`. This is the flow-control cap on total bytes currently being transferred across *all* active object-transfer operations on this node. Once in-flight bytes hit this ceiling, new transfers queue until existing ones complete. On InfiniBand networks capable of 100+ Gbps, the default 2 GB limits throughput to a fraction of the available bandwidth. Raise to 8–16 GB to saturate high-speed interconnects. | With InfiniBand (100+ Gbps), the default bottlenecks transfers. |
        7. Eugo conclusion:
        1. +; we set it but to the same value across both CPU and GPU instances - most of the GPU-related traffic is controlled by the driver anyway, not by Ray + in future, Ray is going to use NCCL for GPU data transfers and orchestration so these GPU-allocated objects should not go via Ray anyway - if they do, it defeats the NVLink purpose and creates redundant pressure on CPU and its RAM
        ```
    2. Watch but too brittle for us to change for now:
        ```
        (~) RAY_scheduler_spread_threshold
        1. Default value: `0.5`
        2. Type: float
        3. Read / Write: R
        4. Recommended value (for HPC): **`0.0–0.2`**
        5. Mechanism: **Process: Raylet.**
        6. Notes: Used in `SchedulingOptions::Hybrid()` (`scheduling_options.h`), passed to `HybridSchedulingPolicy::ComputeNodeScore()`. When a node's resource utilization is below this fraction, the scheduler prefers packing (placing tasks on already-busy nodes). Above this fraction, it switches to spreading (distributing to less-loaded nodes). Default 0.5 means spreading only starts when a node is >50% utilized. Set 0.0–0.2 to spread earlier, improving GPU utilization across nodes by preventing hot spots. | Lower = earlier spreading = better GPU utilization across nodes. `0.0` = always spread. |
        7. Eugo conclusion:
        1. ~; we've decided not to touch this value - allocating tasks closer to each other also improves data locality and reduces NUMA effects. On the other hand, in future we may redecide on that so let's keep this flag in mind.
    ```
   3. Discard:
        ```
        (-) 1. RAY_worker_oom_score_adjustment
        1. Default value: `1000`
        2. Type: int
        3. Read / Write: R
        4. Recommended value (for HPC): **`0`**
        5. Mechanism: **Process: Raylet (acts on workers).**
        6. Notes: After spawning each worker, the raylet's `WorkerPool::AdjustWorkerOomScore()` (`worker_pool.cc`) writes this value to `/proc/<pid>/oom_score_adj`. This is the Linux OOM killer's per-process priority knob (range 0–1000): higher = killed first when system memory is exhausted. Default 1000 makes workers *maximally killable* — the OS will kill every worker before touching the raylet. On HPC nodes where the job IS the workers, losing workers to OOM (while preserving the raylet) just kills your training run anyway. Set `0` so workers have equal OOM priority with the raylet. IO workers are exempted (not adjusted). Linux-only (`#ifdef __linux__`). | Default makes workers first to be OOM-killed. On HPC nodes the job IS the workers. |
        7. Eugo conclusion:
        1. -; it's safer to keep the defaults as we want workers to be killed BEFORE raylet to know what caused the issue - OOM in workers - i.e., user code - or in Raylet - our & ray code
    ```

   1. Alongside it, increase the verbosity of every variable description and try giving us the justification why do you believe we should set the certain value for it.
   2. Reapply our decisions we've made for section (in Table of Contents) or per group (enclosed into `@begin:` and `@end:`): we use the following signs for that `(-)`, `(~)`, `(+)` w/ optional comment-including form like `(~, comment)`.
3. See `## 5. Topology & Graph` in `@__deleteme/stage_2/NCCL_ENV_VARS.md`. How good, reliable, and performant the NCCL topology autodetection? That is - what if we won't supply that file - will it make our p6e instances less efficient or could it miss our configurations?
   1. How stable its format across versions - if we'd extract it from AWS DL AMI - would it work w/o changes for a time?
   2. Could you search in AWS GitHub or other sources if this file can downloaded w/o spawning the instance (we talk about these p6e UltraServers)?
   3. Do we need a separate file for p6e UltraClusters (multiple physically close UltraServers connected via AWS EFA)? Does AWS provides that file as well?