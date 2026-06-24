RECOMPUTE_POLICIES = {
    "resnet18_exact": {
        "keep": {
            "model.output",
            "graph:conv:19:input",
            "graph:relu:16:result",
            "graph:relu:14:result",
            "graph:relu:12:result",
        },
        "drop": {
            # bn input
            *{f"graph:bn:{i}:input" for i in range(20)},

            # non-critical relu
            "graph:relu:15:result",
            "graph:relu:13:result",
            "graph:relu:11:result",
            "graph:relu:9:result",
            "graph:relu:7:result",
            "graph:relu:5:result",
            "graph:relu:3:result",
            "graph:relu:1:result",
            "graph:relu:0:result",

            "graph:addmm:0:mat1",
        },
    },

    "vgg_exact": {
        "keep": {
            "model.output",
            # 여기에 VGG seed relu/checkpoint 넣기
            "graph:relu:8:result",
            "graph:relu:6:result",
            "graph:relu:4:result",
            "graph:relu:2:result",
            "graph:addmm:1:mat1",
        },
        "drop": {
            *{f"graph:bn:{i}:input" for i in range(8)},
            "graph:relu:7:result",
            "graph:relu:5:result",
            "graph:relu:3:result",
            "graph:relu:1:result",
            "graph:relu:0:result",
        },
    },
}