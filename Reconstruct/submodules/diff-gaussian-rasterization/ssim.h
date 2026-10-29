#pragma once
#include <torch/torch.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
);

torch::Tensor
fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
);

static std::vector<std::string> allowed_padding = {"same", "valid"};

class FusedSSIMMap : public torch::autograd::Function<FusedSSIMMap> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 double C1, double C2, torch::Tensor img1,
                                 torch::Tensor img2, std::string padding = "same", bool train = true) {
        torch::Tensor ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12;
        
        // Call the fusedssim function (assumed to be defined)
        std::tie(ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12) = fusedssim(C1, C2, img1, img2, train);
        
        if (padding == "valid") {
            ssim_map = ssim_map.index({torch::indexing::Slice(), torch::indexing::Slice(), 
                                       torch::indexing::Slice(5, -5), torch::indexing::Slice(5, -5)});
        }

        ctx->save_for_backward({img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12});
        ctx->saved_data["C1"] = C1;
        ctx->saved_data["C2"] = C2;
        ctx->saved_data["padding"] = padding;

        return ssim_map;
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list opt_grad) {
        auto saved = ctx->get_saved_variables();
        auto img1 = saved[0];
        auto img2 = saved[1];
        auto dm_dmu1 = saved[2];
        auto dm_dsigma1_sq = saved[3];
        auto dm_dsigma12 = saved[4];

        double C1 = ctx->saved_data["C1"].toDouble();
        double C2 = ctx->saved_data["C2"].toDouble();
        std::string padding = ctx->saved_data["padding"].toStringRef();

        auto dL_dmap = opt_grad[0];
        if (padding == "valid") {
            dL_dmap = torch::zeros_like(img1);
            dL_dmap.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 
                                torch::indexing::Slice(5, -5), torch::indexing::Slice(5, -5)}, opt_grad[0]);
        }

        // Call the fusedssim_backward function (assumed to be defined)
        auto grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
        return {torch::Tensor(), torch::Tensor(), grad, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor fused_ssim(torch::Tensor img1, torch::Tensor img2, std::string padding = "same", bool train = true);
