from .network import PaperUNet


def create_model(config, pretrained=None):
    cfg = config['model']
    pre = cfg.get('pretrained', True) if pretrained is None else pretrained
    name = cfg['name']
    if name == 'paper_unet':
        return PaperUNet(cfg['backbone'], cfg['attention'], len(config['classes']),
                         config['image_size'], pre)
    if name == 'unet2015':
        from .unet2015 import UNet2015
        return UNet2015(len(config['classes']))
    if name == 'deeplabv3_r101_paper':
        from .deeplabv3_paper import DeepLabV3Paper
        return DeepLabV3Paper(len(config['classes']), pre, cfg.get('weights_path'),
                              cfg.get('weights_sha256'))
    raise ValueError(f'Unknown model: {name}')


def logits_of(model, x):
    output = model(x)
    return output['out'] if isinstance(output, dict) else output
