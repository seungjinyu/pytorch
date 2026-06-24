# Architecture

```
+----------------+
|     Node A     |
|  Forward Side  |
+----------------+
            |
            | Payload
            v
+----------------+
|     Node B     |
| Backward Side  |
+----------------+
```

# Node A Responsibilities 

1. Receive template plan 
2. Execute forward pass 
3. Capture required tensors 
4. Build payload 
5. Send payload to Node B 
6. Receive updated model state 

# Node B Responsibilities

1. Generate template plan 
2. Wait for payload 
3. Execute dummy forward 
4. Overwrite saved tensors 
5. Execute backward 
6. Update parameters 
7. Return state_dict

# Execution Flow

Node B starts -> Template Plan Generation -> Node A starts -> Forward Pass -> Payload Capture -> Payload Transfer -> Dummy Forward on B -> Saved Tensor Overwrite -> Backward Execution -> Optimizer Step -> State Synchronization