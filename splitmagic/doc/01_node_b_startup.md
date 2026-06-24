# 01. Node B Startup

## Purpose 

Node B is the backward-side runtime.


## Usage Example 
```
from splitmagic.node_b_runtime import run_node_b


...

def main():

...

    run_node_b(
        model=model,
        endpoint="tcp://*:5555",
        csv_path="node_b_vgg11bn_grad.csv",
        lr=0.1,
    )

```