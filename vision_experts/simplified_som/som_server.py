import gradio as gr
import torch
import argparse

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from inference_semsam_m2m_auto import inference_semsam_m2m_auto
from automatic_mask_generator import prompt_switch

semsam_cfg = "semantic_sam_only_sa-1b_swinL.yaml"
semsam_ckpt = "swinl_only_sam_many2many.pth"
opt_semsam = load_opt_from_config_file(semsam_cfg)

'''
build model
'''
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()

@torch.no_grad()
def inference(image, slider, alpha, label_mode, anno_mode, *args, **kwargs):
    _image = image.convert('RGB')
    
    model_name = 'semantic-sam'
    if slider < 1.5 + 0.14:
        level = [1]
    elif slider < 1.5 + 0.28:
        level = [2]
    elif slider < 1.5 + 0.42:
        level = [3]
    elif slider < 1.5 + 0.56:
        level = [4]
    elif slider < 1.5 + 0.70:
        level = [5]
    elif slider < 1.5 + 0.84:
        level = [6]
    else:
        level = [6, 1, 2, 3, 4, 5]

    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'

    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh = '','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False
        model = model_semsam
        output, mask = inference_semsam_m2m_auto(model, _image, level, text, text_part, text_thresh, 
                                                 text_size, hole_scale, island_scale, semantic, 
                                                 label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, *args, **kwargs)
        
        return output, mask
    

def gradio_interface(image, gradularity, alpha, label_mode, anno_mode):
    return inference(image, gradularity, alpha, label_mode, anno_mode)

demo = gr.Interface(fn=gradio_interface, inputs=[gr.Image(type="pil"),
                                                 gr.Number(value=1.8),
                                                 gr.Number(value=0.1),
                                                 gr.Radio(["Number", "Alphabet"], value="Number"),
                                                 gr.CheckboxGroup(["Mask", "Box", "Mark"], value=["Mask", "Mark"])],
                    outputs=[gr.Image(type="pil"), "json"]
                    )
                    
demo.launch(share=True, server_name="localhost", server_port=8080)