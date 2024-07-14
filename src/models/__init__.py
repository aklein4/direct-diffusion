
from models.dit import DiTConfig, DiT


CONFIG_DICT = {
    "dit": DiTConfig,
}

MODEL_DICT = {
    "dit": DiT,
}


# from diffusers import DDIMScheduler, UNet2DConditionModel
# from transformers import CLIPTextModel, CLIPTokenizer

# import utils.constants as constants

# def get_components(url):

#     scheduler_config = DDIMScheduler.load_config(url, subfolder='scheduler')
#     scheduler_config['rescale_betas_zero_snr'] = True
#     scheduler_config['prediction_type'] = 'sample'
#     scheduler = DDIMScheduler.from_config(scheduler_config)
#     scheduler.set_timesteps(scheduler.config.num_train_timesteps)

#     unet_config = UNet2DConditionModel.load_config(url, subfolder='unet')
#     unet_config['in_channels'] = constants.LATENT_DEPTH
#     unet_config['out_channels'] = constants.LATENT_DEPTH
#     unet = UNet2DConditionModel.from_config(unet_config)

#     tmp_unet = UNet2DConditionModel.from_pretrained(url, subfolder='unet')
#     state_dict = tmp_unet.state_dict()
#     for k in ['conv_in.weight', 'conv_out.weight', 'conv_out.bias']:
#         state_dict.pop(k)
#     unet.load_state_dict(state_dict, strict=False)

#     tokenizer = CLIPTokenizer.from_pretrained(url, subfolder='tokenizer')
#     text_encoder = CLIPTextModel.from_pretrained(url, subfolder='text_encoder')

#     return (
#         unet,
#         scheduler,
#         tokenizer,
#         text_encoder
#     )
