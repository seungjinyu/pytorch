python3 tests/test_operator_concurrency_resnet18.py \
    --device cuda \
    --dataset cifar10 \
    --batch-size 32 \
    --all

python3 tests/test_operator_concurrency_resnet18.py \
    --device cuda \
    --dataset cifar10 \
    --batch-size 32 \
    --all \
    --output ./operator_profiles/resnet18_cifar10_cuda.csv