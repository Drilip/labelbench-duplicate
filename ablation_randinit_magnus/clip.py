# Code related to zeroshot classifier is mainly ported from
# https://github.com/mlfoundations/wise-ft/blob/master/src/models/zeroshot.py
import clip
import torch
import torch.nn as nn

from LabelBench.skeleton.model_skeleton import register_model
from LabelBench.model.model_impl.zero_shot_head import get_zeroshot_classifier


class CLIPVisionOnly(nn.Module):
    @staticmethod
    def _random_init_visual(visual_module):
        before = {name: tensor.detach().clone() for name, tensor in visual_module.state_dict().items()}
        checksum_before = sum(tensor.detach().float().abs().sum().item() for tensor in before.values())
        print(f"[random_init_visual] checksum before: {checksum_before:.6f}")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            for module in visual_module.modules():
                if isinstance(module, nn.MultiheadAttention):
                    module._reset_parameters()
                elif hasattr(module, "reset_parameters"):
                    module.reset_parameters()

            width = visual_module.class_embedding.shape[-1]
            if hasattr(visual_module, "class_embedding"):
                visual_module.class_embedding.data = torch.randn(
                    visual_module.class_embedding.shape,
                    device=visual_module.class_embedding.device,
                    dtype=visual_module.class_embedding.dtype,
                ) * (width ** -0.5)
            if hasattr(visual_module, "positional_embedding"):
                visual_module.positional_embedding.data = torch.randn(
                    visual_module.positional_embedding.shape,
                    device=visual_module.positional_embedding.device,
                    dtype=visual_module.positional_embedding.dtype,
                ) * (width ** -0.5)
            if hasattr(visual_module, "proj"):
                visual_module.proj.data = torch.randn(
                    visual_module.proj.shape,
                    device=visual_module.proj.device,
                    dtype=visual_module.proj.dtype,
                ) * (width ** -0.5)

        checksum_after = sum(
            tensor.detach().float().abs().sum().item() for tensor in visual_module.state_dict().values()
        )
        print(f"[random_init_visual] checksum after: {checksum_after:.6f}")

        for name, tensor in visual_module.state_dict().items():
            if name in before and torch.equal(tensor, before[name]):
                raise AssertionError(f"[random_init_visual] visual tensor {name} did not change")

    def __init__(self, num_output, ret_emb, pretrain, model_name, random_init_visual=False):

        super(CLIPVisionOnly, self).__init__()
        assert pretrain, "CLIPVisionOnly only support pretrain model"

        model, preprocess = clip.load(model_name)
        self.image_encoder_model = model.float()  # Convert to float to avoid NAN loss when using AdamW.
        if random_init_visual:
            self._random_init_visual(self.image_encoder_model.visual)
        self.embed_dim = model.state_dict()["text_projection"].shape[1]
        self.preprocess_transform = preprocess
        self.ret_emb = ret_emb
        self.num_output = num_output

        # Set num_output to 0 to return the embedding.
        if num_output != 0:
            self.classifier = nn.Linear(self.embed_dim, num_output)
        else:
            self.classifier = nn.Identity()

    def forward(self, imgs, ret_features=False, freeze=False):
        if freeze:
            with torch.no_grad():
                features = self.image_encoder_model.encode_image(imgs)
        else:
            features = self.image_encoder_model.encode_image(imgs)

        if ret_features:
            return self.classifier(features), features.data
        elif self.ret_emb:
            return self.classifier(features), features
        else:
            return self.classifier(features)

    def init_head_withzeroshot(self, classnames, template):
        self.classifier = get_zeroshot_classifier(self.image_encoder_model, clip.tokenize, classnames, template)
        print("Update the head initialization using zero-shot classifier params.")

    def get_embedding_dim(self):
        return self.embed_dim

    def get_preprocess(self, split):
        return self.preprocess_transform


@register_model("clip_vitb16")
def init_clip_viTB16(model_config):
    return CLIPVisionOnly(model_config["num_output"],
                          ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                          pretrain=model_config["pretrain"] if "pretrain" in model_config else True,
                          model_name="ViT-B/16",
                          random_init_visual=model_config.get("random_init_visual", False))


@register_model("clip_vitb32")
def init_clip_viTB32(model_config):
    return CLIPVisionOnly(model_config["num_output"],
                          ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
                          pretrain=model_config["pretrain"] if "pretrain" in model_config else True,
                          model_name="ViT-B/32",
                          random_init_visual=model_config.get("random_init_visual", False))
