# VRAM management maps for the retained base modules.

VRAM_MANAGEMENT_MODULE_MAPS = {'uniavc.models.audio_vae.AudioDecoder': {'torch.nn.Conv2d': 'uniavc.core.vram.layers.AutoWrappedModule'},
 'uniavc.models.audio_vae.AudioEncoder': {'torch.nn.Conv2d': 'uniavc.core.vram.layers.AutoWrappedModule'},
 'uniavc.models.audio_vae.AudioVocoder': {'torch.nn.Conv1d': 'uniavc.core.vram.layers.AutoWrappedModule',
                                                 'torch.nn.ConvTranspose1d': 'uniavc.core.vram.layers.AutoWrappedModule'},
 'uniavc.models.dit.DiTModel': {'torch.nn.Linear': 'uniavc.core.vram.layers.AutoWrappedLinear',
                                        'torch.nn.RMSNorm': 'uniavc.core.vram.layers.AutoWrappedModule'},
 'uniavc.models.text_encoder.TextEncoder': {'torch.nn.Linear': 'uniavc.core.vram.layers.AutoWrappedLinear',
                                                        'transformers.models.gemma3.modeling_gemma3.Gemma3MultiModalProjector': 'uniavc.core.vram.layers.AutoWrappedModule',
                                                        'transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm': 'uniavc.core.vram.layers.AutoWrappedModule',
                                                        'transformers.models.gemma3.modeling_gemma3.Gemma3TextScaledWordEmbedding': 'uniavc.core.vram.layers.AutoWrappedModule'},
 'uniavc.models.text_encoder.TextEncoderPostModules': {'uniavc.models.text_encoder.Embeddings1DConnector': 'uniavc.core.vram.layers.AutoWrappedModule',
                                                                   'torch.nn.Linear': 'uniavc.core.vram.layers.AutoWrappedLinear',
                                                                   'torch.nn.RMSNorm': 'uniavc.core.vram.layers.AutoWrappedModule'},
 'uniavc.models.upsampler.LatentUpsampler': {'torch.nn.Conv2d': 'uniavc.core.vram.layers.AutoWrappedModule',
                                                         'torch.nn.Conv3d': 'uniavc.core.vram.layers.AutoWrappedModule',
                                                         'torch.nn.GroupNorm': 'uniavc.core.vram.layers.AutoWrappedModule'},
 'uniavc.models.video_vae.VideoDecoder': {'torch.nn.Conv3d': 'uniavc.core.vram.layers.AutoWrappedModule'},
 'uniavc.models.video_vae.VideoEncoder': {'torch.nn.Conv3d': 'uniavc.core.vram.layers.AutoWrappedModule'}}

VERSION_CHECKER_MAPS = {}
