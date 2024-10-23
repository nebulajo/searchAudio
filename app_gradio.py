import os
import json
import torch
import argparse
import transformers
from tqdm import tqdm
from mtrpp.datasets.music_caps import MusicCaps
from mtrpp.datasets.song_describer import SongDescriber
from mtrpp.utils.query_utils import query_processor
from mtrpp.utils.metrics import recall, mean_average_precision, mean_reciprocal_rank, median_rank
from mtrpp.utils.eval_utils import get_query2target_idx, get_task_predictions, load_ttmr_pp, print_model_params
from sklearn import metrics

from mtrpp.utils.audio_utils import float32_to_int16, int16_to_float32, load_audio, STR_CH_FIRST

import gradio as gr
from g2pkk import G2p as G2pk
import pysbd
import numpy as np
import pandas as pd
from googletrans import Translator
from langdetect import detect

##################
##   inference  ##
##################

parser = argparse.ArgumentParser(description="")
parser.add_argument("--data_dir", type=str, default="../../dataset")
parser.add_argument("--data_type", type=str, default="music_caps")
parser.add_argument("--audio_loader", type=str, default="ffmpeg")
parser.add_argument("--eval_query", type=str, default="caption")
parser.add_argument("--device", type=str, default="cuda:0")
# ttmr_pp config
parser.add_argument('--model_type', type=str, default="ttmrpp")
parser.add_argument("--caption_type", type=str, default="meta_tag_caption_sim")
# train confing 
parser.add_argument("--tid", default="base", type=str)
parser.add_argument("--ckpt_type", default="last", type=str)

parser.add_argument("--ckpt", default="ckpt", type=str)
args = parser.parse_args()

g2pk = G2pk()
segmenter = pysbd.Segmenter(language="en", clean=False)  # no korean support

class GradioApp:
    def __init__(self, args):
        self.args = args
        
        self.device = args.device
        self.model, self.sr, self.duration = load_ttmr_pp(args.ckpt, model_types=args.ckpt_type)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.audio_loader = args.audio_loader
        self.n_samples = int(self.sr * self.duration)
        
            
    def _load_audio(self, audio_path):
        
        audio, _ = load_audio(
            path=audio_path,
            ch_format= STR_CH_FIRST,
            resample_by = self.audio_loader,
            sample_rate= self.sr,
            downmix_to_mono= True
        )
        audio = int16_to_float32(float32_to_int16(audio.astype('float32')))

        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        input_size = int(self.n_samples)
        if audio.shape[-1] < input_size:
            pad = np.zeros(input_size)
            pad[:audio.shape[-1]] = audio
            audio = pad
        ceil = int(audio.shape[-1] // self.n_samples) 
        audio_tensor = torch.from_numpy(np.stack(np.split(audio[:ceil * self.n_samples], ceil)).astype('float32'))
        return audio_tensor


    def inference(self, query, audio_li):
        
        print("query: ", query)
        # Text Translation
        query_lan = detect(query)
        if query_lan == "ko":
            translator = Translator()
            query = translator.translate(query, src='ko', dest='en')
            query = query.text        
            print("translated query: ", query)
        elif query_lan == "en":
            pass
        else:
            raise ValueError("Language must be kr or en")
        
        
        # load audio        
        audio_tensor_li = [self._load_audio(audio) for audio in audio_li]
        
        audio_features, query_features = [], []        
        
        # audio forward
        for audio in audio_tensor_li:
            if audio.shape[0] == int(self.sr * self.duration):
                audio = audio.unsqueeze(0) # for pre-batch
            else:
                audio = audio.squeeze(0) # for pre-chunk
            with torch.no_grad():
                audio_embs = self.model.audio_forward(audio.to(self.device)).mean(0, True)
            audio_features.extend(audio_embs.detach().cpu())
        
        # text forward
        with torch.no_grad():
            query_embs = self.model.text_forward([query])
        query_features.extend(query_embs.detach().cpu())

        # get similarity
        query2audio_matrix = get_task_predictions(
            torch.stack(query_features), torch.stack(audio_features)
        )

        desc_indices = np.argsort(query2audio_matrix, axis=-1)[:, ::-1]        
        
        audio_df = [ [idx+1, audio_li[rank].split("/")[-1]] for idx, rank in enumerate(desc_indices[0])]
        print(audio_df)
        audio_df = pd.DataFrame(audio_df, columns = ["rank", "audio"])

        audio_df.iloc[4, audio_df.columns.get_loc('audio')] = gr.Audio(type="filepath", label="The most similar sample")

        return (            
            audio_li[desc_indices[0][0]], 
            audio_df
        )

if __name__ == "__main__":
    app = GradioApp(args)

    with gr.Blocks() as demo:
        
        title= gr.Markdown("<h1 style='text-align: center;'>Find the most similar Audio</h1>")
        
        # interface = app.interface
        with gr.Row():
            textbox = gr.Textbox(
                label="Sample Description",
                value="Write down description about sample",
            )

            audiobox = gr.Files(label="Audio files", file_count="multiple")
                
        tts_button = gr.Button("Run")

        with gr.Row():
            sim_audio1 = gr.Audio(type="filepath", label="The most similar sample")
        
            audio_df = gr.Dataframe(
                headers=["rank", "audio"],
                datatype=["number", "str"],
            )        

        tts_button.click(
            app.inference,
            inputs=[
                textbox,
                audiobox,
            ],
            outputs=[
                sim_audio1,
                audio_df
            ],
        )
        
    demo.launch(share=True, debug=True)
