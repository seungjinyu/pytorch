
# SplitMagic Documentation

## Overview 

SplitMagic is a framework that seperates forward and backward execution across two nodes while preserving PyTorch autograd semantics.

Node A performs forward execution and captures tensors required for backward computation. 

Node B reconstructs the backward pass using the captured tensors and executes gradient computation without access to the original input data.

## High-Level flow 

### Node A 
|- forward 
|- Capture Saved Tensors 
|- Build Payload
|- Send Payload
    |
    |
    V
### Node B 
|- Load Payload 
|- Overwrite Saved Tensors 
|- Execute Backward 
|- Compute Gradient 

## Main Components 

### Tensor Capture 
- Collects tensors required by backward operators

### Payload Serialization
- Serializes captured tensors

### Dry-Run Plannings
- Generates execution plans 

### Runtime State
- Loads payload tensors into C++ runtimes 

### Tensor Overwrite
- Replaces saved tensors during backward 

### Execution Plan 
- Provides a stable operator ordering mechanism independent of operator-local counters. 


## Core Functions and Modules
| Module / Function Group | Main Files | Role |
|---|---|---|
| Node A Runtime | `node_a_runtime.py`, `test_node_a_resnet.py` | Runs forward on Node A, captures tensors, builds payload, sends data to Node B. |
| Node B Runtime | `node_b_runtime.py`, `test_node_b_resnet.py` | Receives payload, reconstructs backward, runs loss/backward, compares gradients. |
| Shared Runtime | `runtime.py` | Common runtime logic used by both Node A and Node B. |
| Payload Handling | `payload.py`, `data.py`, `comm.py` | Defines payload structure, tensor serialization, communication utilities. |
| Dry-Run / Plan | `plan.py`, `op_indexer.py`, `dryrun_resnet18_saved_tensors.py` | Builds and manages execution plans used to match saved tensors with backward operators. |
| Graph / FX Utilities | `graph.py`, `fx_trace.py`, `fx_recompute.py`, `debug_backward_graph.py` | Inspects model graphs and supports recomputation or graph-level debugging. |
| Tensor Matching / Key Mapping | `keymap.py`, `matcher.py`, `resolver.py`, `inspector.py` | Maps captured tensors to execution-plan keys and resolves tensor/operator matching. |
| Hooks | `hooks.py` | Registers hooks or capture logic around model execution. |
| Recompute / Replay | `recompute.py`, `replay.py` | Provides recomputation or replay-based reconstruction utilities. |
| Models | `models.py` | Defines model architectures used in experiments. |
| Verification / Tests | `test_compare_resnet18.py`, `test_single_resnet.py`, `eval_resnet18_split_final.py` | Compares baseline gradients with SplitMagic gradients and evaluates correctness. |
| Logs / Artifacts | `log/`, `*.csv`, `*.pt`, `resnet_a.txt`, `resnet_b.txt` | Stores experiment outputs, gradients, model states, and debugging logs. |

## Repository Structure

```text
splitmagic/
├── node_a_runtime.py              # Node A forward/capture runtime
├── node_b_runtime.py              # Node B backward/reconstruction runtime
├── runtime.py                     # Shared SplitMagic runtime logic
├── payload.py                     # Payload data structure and serialization
├── comm.py                        # Communication utilities
├── data.py                        # Dataset/data loading utilities
├── plan.py                        # Execution plan utilities
├── op_indexer.py                  # Operator indexing / ordering utilities
├── hooks.py                       # Hook registration and tensor capture helpers
├── keymap.py                      # Tensor key mapping utilities
├── matcher.py                     # Tensor/operator matching utilities
├── resolver.py                    # Resolves keys, tensors, or plan entries
├── inspector.py                   # Debugging and inspection utilities
├── graph.py                       # Graph inspection utilities
├── fx_trace.py                    # FX tracing utilities
├── fx_recompute.py                # FX-based recomputation utilities
├── recompute.py                   # Recompute utilities
├── replay.py                      # Replay utilities
├── models.py                      # Model definitions
├── data/                          # Dataset directory
├── log/                           # Runtime logs
├── test_codes/                    # Additional test scripts
├── utils/                         # Utility functions
├── Doc/                           # Documentation directory
└── README.md

SplitMagic

├── Runtime Layer
│   ├── node_a_runtime.py
│   ├── node_b_runtime.py
│   └── runtime.py
│
├── Capture Layer
│   └── hooks.py
│
├── Payload Layer
│   ├── payload.py
│   └── data.py
│
├── Planning Layer
│   ├── plan.py
│   └── op_indexer.py
│
├── Matching Layer
│   ├── keymap.py
│   ├── matcher.py
│   ├── resolver.py
│   └── inspector.py
│
├── Graph Analysis Layer
│   ├── graph.py
│   ├── fx_trace.py
│   └── debug_backward_graph.py
│
├── Recomputation Layer
│   ├── recompute.py
│   ├── fx_recompute.py
│   └── replay.py
│
└── Verification Layer
    ├── test_compare_resnet18.py
    ├── test_single_resnet.py
    └── eval_resnet18_split_final.py