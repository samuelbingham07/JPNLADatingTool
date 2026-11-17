import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection

BACKBONE_ID = "openai/clip-vit-base-patch32"
SIZE = 224
_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def load_image(path_or_img) -> Image.Image:
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    # JPEG draft mode decodes at reduced scale; listing photos are ~1600px so this is a big speedup
    img.draft("RGB", (SIZE * 2, SIZE * 2))
    return img.convert("RGB")


def preprocess(img: Image.Image) -> torch.Tensor:
    w, h = img.size
    scale = SIZE / min(w, h)
    img = img.resize((max(SIZE, round(w * scale)), max(SIZE, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(SIZE, SIZE, 3)
    t = t.permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD


def load_backbone():
    model = CLIPVisionModelWithProjection.from_pretrained(BACKBONE_ID)
    model.eval()
    return model


@torch.no_grad()
def embed(model, batch: torch.Tensor) -> torch.Tensor:
    out = model(pixel_values=batch).image_embeds
    return torch.nn.functional.normalize(out, dim=-1)
