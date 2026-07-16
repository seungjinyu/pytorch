┌──────────────────────┐
│        Node A        │
│ Forward execution    │
└──────────┬───────────┘
           │
           │ Saved tensors
           │ Model output
           │ Labels
           │ Execution plan
           ▼
┌──────────────────────┐
│        Node B        │
│ Receive payload      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Build backward graph │
└──────────┬───────────┘
           │
           ▼
┌────────────────────────────┐
│ Check required tensors     │
│ for backward execution     │
└──────────┬─────────────────┘
           │
      Missing tensors?
        ┌──┴──┐
        │     │
       No    Yes
        │     │
        │     ▼
        │  ┌──────────────────────┐
        │  │ Recompute missing    │
        │  │ tensors on Node B    │
        │  └──────────┬───────────┘
        │             │
        └──────┬──────┘
               ▼
┌──────────────────────┐
│ Execute backward     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Optimizer update     │
└──────────┬───────────┘
           │
           │ Updated model state
           │ Loss / gradients
           ▼
┌──────────────────────┐
│        Node A        │
│ Receive result       │
└──────────────────────┘

Figure 1

Figure 1 shows the overall execution flow between Node A and Node B. Node A performs the forward pass and transfers the model output, labels, saved tensors, and execution plan to Node B. Node B constructs the backward graph and checks whether all tensors required for backward execution are available. Missing tensors are reconstructed through recomputation before the backward pass is executed. After gradient computation and the optimizer update, the updated model state and execution results are returned to Node A.

======================================================================

┌─────────────────────────────────────┐
│ tests/test_node_b_resnet18.py       │
│ main()                              │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ run_node_b()                        │
│ splitmagic/node_b_runtime.py        │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Receive request from Node A         │
│ - payload                           │
│ - labels                            │
│ - model output                      │
│ - execution plan                    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ backward_jin()                      │
│ splitmagic/runtime.py               │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Identify tensors required           │
│ for backward execution              │
└──────────────────┬──────────────────┘
                   │
                   ▼
          ┌───────────────────┐
          │ Missing tensors?  │
          └─────────┬─────────┘
                No  │  Yes
            ┌───────┘   └──────────────┐
            │                          ▼
            │             ┌─────────────────────────┐
            │             │ recompute_missing_keys()│
            │             └────────────┬────────────┘
            │                          │
            │                          ▼
            │             ┌─────────────────────────┐
            │             │ recompute_path()        │
            │             │ Execute required nodes  │
            │             └────────────┬────────────┘
            │                          │
            └──────────────┬───────────┘
                           ▼
┌─────────────────────────────────────┐
│ Inject available/recomputed tensors │
│ into backward execution             │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ loss.backward()                     │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ optimizer.step()                    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Send results back to Node A         │
│ - loss                              │
│ - gradients                         │
│ - updated model state               │
└─────────────────────────────────────┘

Figure 2 

Figure 2 illustrates the internal execution flow of Node B. The process begins in tests/test_node_b_resnet18.py, which invokes run_node_b() in node_b_runtime.py. Node B receives the forward-pass results and saved-tensor payload from Node A, then calls backward_jin() in runtime.py.

Before executing the backward pass, Node B identifies the tensors required by the backward graph and checks whether they are available in the received payload. If any tensors are missing, recompute_missing_keys() determines how they can be reconstructed, and recompute_path() executes the necessary forward operators. The received and recomputed tensors are then supplied to the backward execution. Finally, Node B performs loss.backward(), updates the model parameters, and returns the results to Node A.