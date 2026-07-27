import time
import torch


# CONCURRENCY = 128
# MATRIX_SIZE = 2048


CONCURRENCY = 32
MATRIX_SIZE = 2048

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")

    streams = [
        torch.cuda.Stream(device=device)
        for _ in range(CONCURRENCY)
    ]

    inputs = [
        torch.randn(
            MATRIX_SIZE,
            MATRIX_SIZE,
            device=device,
        )
        for _ in range(CONCURRENCY)
    ]

    weights = [
        torch.randn(
            MATRIX_SIZE,
            MATRIX_SIZE,
            device=device,
        )
        for _ in range(CONCURRENCY)
    ]

    torch.cuda.synchronize()

    print(
        f"[GPU_CONTENTION] "
        f"streams={CONCURRENCY} "
        f"matrix_size={MATRIX_SIZE}",
        flush=True,
    )

    try:
        while True:
            outputs = []

            for stream, x, weight in zip(
                streams,
                inputs,
                weights,
            ):
                with torch.cuda.stream(stream):
                    output = torch.mm(x, weight)
                    outputs.append(output)

            torch.cuda.synchronize()

    except KeyboardInterrupt:
        print("[GPU_CONTENTION] stopped", flush=True)


if __name__ == "__main__":
    main()
