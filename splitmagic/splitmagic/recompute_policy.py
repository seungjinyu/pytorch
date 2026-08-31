RECOMPUTE_POLICIES = {
    "resnet18_exact": {
        "keep": {
            "model.output",
            "graph:conv:19:input",

        },
        "drop": {
            # bn input
            *{f"graph:bn:{i}:input" for i in range(20)},
            "graph:conv:18:input",
            "graph:conv:17:input",
            "graph:conv:16:input",
            "graph:conv:15:input",
            "graph:conv:14:input",
            "graph:conv:13:input",
            "graph:conv:12:input",
            "graph:conv:11:input",
            "graph:conv:10:input",
            "graph:conv:9:input",
            "graph:conv:8:input",
            "graph:conv:7:input",
            "graph:conv:6:input",
            "graph:conv:5:input",
            "graph:conv:4:input",
            "graph:conv:3:input",
            "graph:conv:2:input",
            "graph:conv:1:input",
            "graph:conv:0:input",

            "graph:relu:16:result",
            "graph:relu:14:result",
            "graph:relu:12:result",
            "graph:relu:10:result",
            "graph:relu:8:result",
            "graph:relu:6:result",
            "graph:relu:4:result",
            "graph:relu:2:result",
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

            "graph:maxpool2d:0:input",
            "graph:maxpool2d:0:indices",
        },
    },

    # "vgg_exact": {
    #     "keep": {
    #         "model.output",
    #         # 여기에 VGG seed relu/checkpoint 넣기
    #         "graph:relu:8:result",
    #         "graph:relu:6:result",
    #         "graph:relu:4:result",
    #         "graph:relu:2:result",
    #         "graph:addmm:1:mat1",
    #     },
    #     "drop": {
    #         *{f"graph:bn:{i}:input" for i in range(8)},
    #         "graph:relu:7:result",
    #         "graph:relu:5:result",
    #         "graph:relu:3:result",
    #         "graph:relu:1:result",
    #         "graph:relu:0:result",
    #     },
    # },
    "vgg_exact": {
        "keep": {
            "model.output",
            # First real model input checkpoint
            "graph:conv:0:input",

            # Intermediate feature checkpoints
            # "graph:relu:2:result",
            # "graph:relu:4:result",
            # "graph:relu:6:result",

            # Classifier checkpoint
            
        },

        "drop": {
            *{f"graph:conv:{i}:input" for i in range(1, 7)},
            *{f"graph:bn:{i}:input" for i in range(8)},
            *{f"graph:relu:{i}:result" for i in range(1, 9)},

            *{
                f"graph:maxpool2d:{i}:input"
                for i in range(5)
            },

            "graph:addmm:0:mat1",
            "graph:addmm:1:mat1",
        },
    },
}