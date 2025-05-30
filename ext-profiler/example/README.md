# NCCL Example Profiler Plugin Usage

This page describes how to use the NCCL example profiler plugin

# Overview

The example profiler plugin implements the NCCL profiler plugin API introduced in NCCL v2.23. The API
defines a set of events and data structures that NCCL uses to share event information with profiler
plugins. The user can control what events are instrumented by NCCL and when traces collected by the
profiler should be dumped through environment variables, as described in the rest of the document.
The user can also control other profiler parameters that alter its behavior. For example, users can
change the size of the event window the profiler keeps track of.

## Building the profiler plugin

To use the example plugin, just type `make`. You will need a NCCL build's include directory present.
You can override `NCCL_HOME` to where the NCCL installation is on your system.

## Using the profiler plugin

1. Add the directory of this profiler plugin to your `LD_LIBRARY_PATH` or set the `NCCL_PROFILER_PLUGIN`,
   as documented in `ext-profiler/README.md`.

2. Set `NCCL_PROFILE_EVENT_MASK` bitmask to specify the NCCL events you want to instrument. By
   default, all collectives and send/recv operations will be traced. For more details about the event
   representation used by the profiler refer to `ext-profiler/README.md`.

   As an example, setting:

   `NCCL_PROFILE_EVENT_MASK` to 1 (`ncclProfileGroup`) | 2 (`ncclProfileColl`) | 8 (`ncclProfileProxyOp`)

   enables the profiling of the group, the collective and the proxy op events. The same events can be
   expressed more concisely by setting `NCCL_PROFILE_EVENT_MASK` to 8 (`ncclProfileProxyOp`). Indeed,
   in NCCL all the events above (in the event hierarchy) the one requested are also captured. The advantage
   is that the profiler can easily correlate events that belong to the same NCCL operation and present
   them accordingly.

3. Set `NCCL_PROFILE_DUMP_FILE` to the name of the dump file for the collected traces. A file named
   ${NCCL_PROFILE_DUMP_FILE}-hostname-tid.txt is created. Profiler traces are saved using the chrome
   event format (more precisely, using asynchronous events).

4. If you set the dump file variable, type chrome://tracing on your chromium browser search bar and
   open the created dump file to visualize the traces.

# Changing the profiler memory pool sizes

The example profiler uses separate memory pools for different types of events. The size of these memory
pools (i.e., the # events) determines the number of events that the profiler can keep track of at the
same time. When NCCL requests a new event (e.g., collective event) to profile a `ncclAllReduce`
operation, by calling `startEvent`, the profiler searches in the collective pool for a free event. If it
finds one, it marks it as in use and returns the handle to NCCL. If the pool is completely used the
profiler returns `NULL` to NCCL and ignores all the following NCCL profiler calls for the `NULL` event
handle. When the `ncclAllReduce` has been processed, NCCL calls `stopEvent` with the previosly returned
event handle. The profiler has a total of 5 memory pools.

The group, collective and p2p pools contain objects for the corresponding events. The `ProxyCtrl` pool
contains objects for `ProxyCtrl` events and the `ProxyDetach` pool contains objects for `ProxyOp` events
generated by remote proxies. A list of pools and their size is reported below:

- `NCCL_PROFILE_GROUP_POOL_SIZE` (16)
- `NCCL_PROFILE_COLL_POOL_SIZE` (16)
- `NCCL_PROFILE_P2P_POOL_SIZE` (1024)
- `NCCL_PROFILE_PROXY_CTRL_POOL_SIZE` (16)
- `NCCL_PROFILE_PROXY_DETACH_POOL_SIZE` (128)

Remote proxy operations are generated when PXN is in use. Refer to this article for more information
about PXN and how it works:
https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/

# Reported events

The example profiler generates traces using the json format. An example of trace is reported below:

```
[
{"name": "Group", "cat": "GROUP", "ph": "b", "id": 0, "pid": 4157654, "tid": 1, "ts": 764234.611328, "args": {"groupId": 0}},
{"name": "AllReduce", "cat": "COLL", "ph": "b", "id": 0, "pid": 4157654, "tid": 1, "ts": 764237.294922, "args": {"SeqNum": 0, "CommHash": 673864846479792718, "Rank": 1, "Count": 32768, "Datatype": "ncclFloat32", "Algorithm": "RING", "Protocol": "LL", "nMaxChannels": 2}},
{"name": "Recv", "cat": "PROXY", "ph": "b", "id": 0, "pid": 4157654, "tid": 1, "ts": 768464.936523, "args": {"Channel": 0, "Peer": 0, "Steps": 14, "ChunkSize": 32768, "transSize": 229376, "POSTED": {"step": 14, "ts": 772020.300781}, "RECEIVED": {"step": 14, "ts": 772196.049805}, "TRANSMITTED": {"step": 14, "ts": 772197.326172}, "DONE": {"step": 14, "ts": 772201.538086}}},
{"name": "RecvBufferWait", "cat": "NET", "ph": "b", "id": 0, "pid": 4157654, "tid": 1, "ts": 768465.158203, "args": {"Step": 0}},
{"name": "RecvBufferWait", "cat": "NET", "ph": "e", "id": 0, "pid": 4157654, "tid": 1, "ts": 768477.924805},
{"name": "RecvWait", "cat": "NET", "ph": "b", "id": 0, "pid": 4157654, "tid": 1, "ts": 768477.924805, "args": {"Step": 0}},
{"name": "RecvWait", "cat": "NET", "ph": "e", "id": 0, "pid": 4157654, "tid": 1, "ts": 768547.197266},
{"name": "RecvFlushWait", "cat": "NET", "ph": "b", "id": 0, "pid": 4157654, "tid": 1, "ts": 768547.197266, "args": {"Step": 0}},
{"name": "RecvFlushWait", "cat": "NET", "ph": "e", "id": 0, "pid": 4157654, "tid": 1, "ts": 768564.174805},
{"name": "RecvGpuWait", "cat": "NET", "ph": "b", "id": 0, "pid": 4157654, "tid": 1, "ts": 768564.174805, "args": {"Step": 0}},
{"name": "RecvGpuWait", "cat": "NET", "ph": "e", "id": 0, "pid": 4157654, "tid": 1, "ts": 768568.276367},
{"name": "RecvBufferWait", "cat": "NET", "ph": "b", "id": 1, "pid": 4157654, "tid": 1, "ts": 768503.604492, "args": {"Step": 1}},
{"name": "RecvBufferWait", "cat": "NET", "ph": "e", "id": 1, "pid": 4157654, "tid": 1, "ts": 768504.549805},
{"name": "RecvWait", "cat": "NET", "ph": "b", "id": 1, "pid": 4157654, "tid": 1, "ts": 768504.549805, "args": {"Step": 1}},
{"name": "RecvWait", "cat": "NET", "ph": "e", "id": 1, "pid": 4157654, "tid": 1, "ts": 769994.490234},
{"name": "RecvFlushWait", "cat": "NET", "ph": "b", "id": 1, "pid": 4157654, "tid": 1, "ts": 769994.490234, "args": {"Step": 1}},
{"name": "RecvFlushWait", "cat": "NET", "ph": "e", "id": 1, "pid": 4157654, "tid": 1, "ts": 769995.012695},
{"name": "RecvGpuWait", "cat": "NET", "ph": "b", "id": 1, "pid": 4157654, "tid": 1, "ts": 769995.012695, "args": {"Step": 1}},
{"name": "RecvGpuWait", "cat": "NET", "ph": "e", "id": 1, "pid": 4157654, "tid": 1, "ts": 770006.914062},
{"name": "RecvBufferWait", "cat": "NET", "ph": "b", "id": 2, "pid": 4157654, "tid": 1, "ts": 768506.941406, "args": {"Step": 2}},
{"name": "RecvBufferWait", "cat": "NET", "ph": "e", "id": 2, "pid": 4157654, "tid": 1, "ts": 768507.435547},
{"name": "RecvWait", "cat": "NET", "ph": "b", "id": 2, "pid": 4157654, "tid": 1, "ts": 768507.435547, "args": {"Step": 2}},
{"name": "RecvWait", "cat": "NET", "ph": "e", "id": 2, "pid": 4157654, "tid": 1, "ts": 771452.536133},
{"name": "RecvFlushWait", "cat": "NET", "ph": "b", "id": 2, "pid": 4157654, "tid": 1, "ts": 771452.536133, "args": {"Step": 2}},
{"name": "RecvFlushWait", "cat": "NET", "ph": "e", "id": 2, "pid": 4157654, "tid": 1, "ts": 771453.060547},
{"name": "RecvGpuWait", "cat": "NET", "ph": "b", "id": 2, "pid": 4157654, "tid": 1, "ts": 771453.060547, "args": {"Step": 2}},
{"name": "RecvGpuWait", "cat": "NET", "ph": "e", "id": 2, "pid": 4157654, "tid": 1, "ts": 771468.458008},
{"name": "RecvBufferWait", "cat": "NET", "ph": "b", "id": 3, "pid": 4157654, "tid": 1, "ts": 768509.484375, "args": {"Step": 3}},
{"name": "RecvBufferWait", "cat": "NET", "ph": "e", "id": 3, "pid": 4157654, "tid": 1, "ts": 768510.250000},
{"name": "RecvWait", "cat": "NET", "ph": "b", "id": 3, "pid": 4157654, "tid": 1, "ts": 768510.250000, "args": {"Step": 3}},
{"name": "RecvWait", "cat": "NET", "ph": "e", "id": 3, "pid": 4157654, "tid": 1, "ts": 771904.499023},
{"name": "RecvFlushWait", "cat": "NET", "ph": "b", "id": 3, "pid": 4157654, "tid": 1, "ts": 771904.499023, "args": {"Step": 3}},
{"name": "RecvFlushWait", "cat": "NET", "ph": "e", "id": 3, "pid": 4157654, "tid": 1, "ts": 771904.991211},
{"name": "RecvGpuWait", "cat": "NET", "ph": "b", "id": 3, "pid": 4157654, "tid": 1, "ts": 771904.991211, "args": {"Step": 3}},
{"name": "RecvGpuWait", "cat": "NET", "ph": "e", "id": 3, "pid": 4157654, "tid": 1, "ts": 771910.500000},
{"name": "Send", "cat": "PROXY", "ph": "b", "id": 1, "pid": 4157654, "tid": 1, "ts": 768482.878906, "args": {"Channel": 0, "Peer": 2, "Steps": 14, "ChunkSize": 32768, "transSize": 229376, "POSTED": {"step": 14, "ts": 771995.675781}, "REM_FIFO_WAIT": {"step": 14, "ts": 772190.692383}, "TRANSMITTED": {"step": 14, "ts": 772191.516602}, "DONE": {"step": 14, "ts": 772208.473633}}},
{"name": "SendBufferWait", "cat": "NET", "ph": "b", "id": 14, "pid": 4157654, "tid": 1, "ts": 768483.019531, "args": {"Step": 0}},
{"name": "SendBufferWait", "cat": "NET", "ph": "e", "id": 14, "pid": 4157654, "tid": 1, "ts": 768483.300781},
{"name": "SendGpuWait", "cat": "NET", "ph": "b", "id": 14, "pid": 4157654, "tid": 1, "ts": 768483.300781, "args": {"Step": 0}},
{"name": "SendGpuWait", "cat": "NET", "ph": "e", "id": 14, "pid": 4157654, "tid": 1, "ts": 769594.615234},
{"name": "SendWait", "cat": "NET", "ph": "b", "id": 14, "pid": 4157654, "tid": 1, "ts": 769594.615234, "args": {"Step": 0}},
{"name": "SendWait", "cat": "NET", "ph": "e", "id": 14, "pid": 4157654, "tid": 1, "ts": 769618.889648},
{"name": "SendBufferWait", "cat": "NET", "ph": "b", "id": 15, "pid": 4157654, "tid": 1, "ts": 768505.083008, "args": {"Step": 1}},
{"name": "SendBufferWait", "cat": "NET", "ph": "e", "id": 15, "pid": 4157654, "tid": 1, "ts": 768505.163086},
{"name": "SendGpuWait", "cat": "NET", "ph": "b", "id": 15, "pid": 4157654, "tid": 1, "ts": 768505.163086, "args": {"Step": 1}},
{"name": "SendGpuWait", "cat": "NET", "ph": "e", "id": 15, "pid": 4157654, "tid": 1, "ts": 769610.555664},
{"name": "SendWait", "cat": "NET", "ph": "b", "id": 15, "pid": 4157654, "tid": 1, "ts": 769610.555664, "args": {"Step": 1}},
{"name": "SendWait", "cat": "NET", "ph": "e", "id": 15, "pid": 4157654, "tid": 1, "ts": 769622.517578},
{"name": "SendBufferWait", "cat": "NET", "ph": "b", "id": 16, "pid": 4157654, "tid": 1, "ts": 768507.937500, "args": {"Step": 2}},
{"name": "SendBufferWait", "cat": "NET", "ph": "e", "id": 16, "pid": 4157654, "tid": 1, "ts": 768508.017578},
{"name": "SendGpuWait", "cat": "NET", "ph": "b", "id": 16, "pid": 4157654, "tid": 1, "ts": 768508.017578, "args": {"Step": 2}},
{"name": "SendGpuWait", "cat": "NET", "ph": "e", "id": 16, "pid": 4157654, "tid": 1, "ts": 770002.129883},
{"name": "SendWait", "cat": "NET", "ph": "b", "id": 16, "pid": 4157654, "tid": 1, "ts": 770002.129883, "args": {"Step": 2}},
{"name": "SendWait", "cat": "NET", "ph": "e", "id": 16, "pid": 4157654, "tid": 1, "ts": 770013.848633},
{"name": "SendBufferWait", "cat": "NET", "ph": "b", "id": 17, "pid": 4157654, "tid": 1, "ts": 768510.742188, "args": {"Step": 3}},
{"name": "SendBufferWait", "cat": "NET", "ph": "e", "id": 17, "pid": 4157654, "tid": 1, "ts": 768510.822266},
{"name": "SendGpuWait", "cat": "NET", "ph": "b", "id": 17, "pid": 4157654, "tid": 1, "ts": 768510.822266, "args": {"Step": 3}},
{"name": "SendGpuWait", "cat": "NET", "ph": "e", "id": 17, "pid": 4157654, "tid": 1, "ts": 771461.563477},
{"name": "SendWait", "cat": "NET", "ph": "b", "id": 17, "pid": 4157654, "tid": 1, "ts": 771461.563477, "args": {"Step": 3}},
{"name": "SendWait", "cat": "NET", "ph": "e", "id": 17, "pid": 4157654, "tid": 1, "ts": 771469.171875},
 ... [ trace truncated for brevity ]
{"name": "AllReduce", "cat": "COLL", "ph": "e", "id": 0, "pid": 4157654, "tid": 1, "ts": 772209.317383},
{"name": "Group", "cat": "GROUP", "ph": "e", "id": 0, "pid": 4157654, "tid": 1, "ts": 772209.418945},
{}]
```

Details about the fields used in the trace can be found at this link:
https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0#heading=h.yr4qxyxotyw

The trace above is obtained by running a `ncclAllReduce` operation on 8 GPUs, communicating with each other through
the network interface. The `Group` event encloses all traces that are related to the single `ncclAllReduce` call.
(Note that for single collective invocations, where there are no explicit group calls, NCCL creates a group with only
one collective and this is what is presented in the traces above).


The `AllReduce` event encloses traces for the proxy operation associated to the `ncclAllReduce` operation. The `args`
field in the traces contains NCCL specific information (aside from the chrome trace event format).

## AllReduce trace

The `AllReduce` entry presents information about the `ncclAllReduce` operation. It contains the following info in the args field:

- seqNum      : sequential number of the collective in the communicator (every collective type has its own sequence number in the communicator)
- commHash    : communicator unique identifier
- rank        : NCCL rank for the ncclAllReduce
- datatype    : NCCL datatype
- algorithm   : algorithm used to process the ncclAllReduce
- protocol    : protocol used to process the ncclAllReduce
- nMaxChannels: max number of channels used to process the ncclAllReduce

If the proxy events are not active (e.g., the `ncclAllReduce` is intranode) the end timestamp will match the time
consumed by the CPU to launch the collective. For more details refer to `ext-profiler/README.md`, section `Profiling
of collective and p2p operations`.

### Proxy Send
The `Send` entry presents information about the `ProxyOp` processing in the progress thread. It contains the following
info in the args field:

- Channel      : id of the channel used by this proxy operation to send data to the peer
- Peer         : peer rank
- Steps        : number of network steps required to transfer transSize bytes to the peer
- ChunkSize    : chunk size used by NCCL to pipeline data through the proxy thread
- transSize    : bytes transferred across the channel by this proxy operation
- POSTED       : struct containing the number of buffer posts to the GPU and the time stamp for the last post
- REM_FIFO_WAIT: struct containing the number of remote buffer waits and the time stamp for the last wait
- TRANSMITTED  : struct containing the number of network sends and the time stamp of the last send
- DONE         : struct containing the number of network sends completed and the time stamp of the last send completed

In case of a network problem the POSTED, REM_FIFO_WAIT, TRANSMITTED and DONE might all have partially updated steps,
which could help identify at which point the network problem occurred.

The Proxy send trace gives a summary of the proxy progress thread activity for the channel. If more details are
needed, these can be obtained by enabling the proxy step event (`ncclProfileProxyStep`). In which case the trace
entries below are also reported by the profiler.

#### Proxy SendBufferWait

Presents, for every network step, the time the CPU proxy spends waiting for the channel staging buffer to become available.

#### Proxy SendGPUWait

Presents, for every network step, the time the CPU proxy spends waiting for the GPU to provide the data in the staging
buffer.

#### Proxy SendWait

Presents, for every network step, the time the CPU proxy spends waiting for the `isend` to complete

### Proxy Recv

The `Recv` entry presents information about the `ProxyOp` processing in the progress thread. It contains the following
info in the args field:

- Channel    : id of the channel used by this proxy operation to recv data from the peer
- Peer       : peer rank
- Steps      : number of network steps required to transfer transSize bytes from the peer
- ChunkSize  : chunk size used by NCCL to pipeline data through the proxy thread
- transSize  : bytes transferred across the channel by this proxy operation
- POSTED     : struct containing the number of recvs posted and the time stamp for the last recv posted
- RECEIVED   : struct containing the number of recvs completed and the time stamp for the last recv completed
- TRANSMITTED: struct containing the number of recvs flushed to the GPU memory and the time stamp for the last recv flushed
- DONE       : struct containing the number of flush completed and the time stamp for the last flush completed

The Proxy Recv trace gives a summary of the proxy progress thread activity for the channel. If more details are
needed, these can be obtained by enabling the proxy step event (`ncclProfileProxyStep`). In which case the trace
entries below are also reported by the profiler.


#### Proxy RecvBufferWait

Presents, for every network step, the time the CPU proxy spends waiting for the staging buffer for the channel to
become available.

#### Proxy RecvWait

Presents, for every network step, the time the CPU proxy spends waiting for a posted `irecv` to complete

#### Proxy RecvFlushWait

Presents, for every network step, the time the CPU proxy spends waitng for the recv data to be flushed to the GPU

#### Proxy RecvGPUWait

Presents, for every network step, the time the CPU proxy spends waiting for the GPU to consume the recv data
