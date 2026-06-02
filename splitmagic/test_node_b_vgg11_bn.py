from splitmagic.node_b_runtime import run_node_b
from splitmagic.models import make_vgg11_bn_cifar10


def main():
    model = make_vgg11_bn_cifar10()
    run_node_b(model=model)


if __name__ == "__main__":
    main()