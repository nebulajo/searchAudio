# Find most similar sound 

본 프로젝트는  [TTMR++](#)(Enriching Music Descriptions with a Finetuned-LLM and Metadata for Text-to-Music Retrieval) 연구의 모델과 가중치를 사용하였습니다.

## DEMO
![KakaoTalk_20241023_152433514](https://github.com/user-attachments/assets/81b4a23d-a859-4cd9-af65-f28066926a26)

## Open Source Material
- [Pre-trained model weights](https://huggingface.co/seungheondoh/ttmr-pp/tree/main) 

## Weight, Dataset dir
* gradio만 실행하는 경우 datasets은 설치하지 않아도 됩니다.

```bash
-- MUSIC-TEXT-REPRESENTATION-PP
---- ckpt
-------- last.pth
-------- hparams.yaml
---- datasets
---- song_describer
-------- song_describer.csv
-------- audio
```

## Install
* Python==3.9
```bash
pip install .
```

## Run Gradio

* Run Gradio
```bash
python app_gradio.py --ckpt {Pre-trained model weights dir}
```


## run inference

* append **sys.path** in `eva_semantic.py`
```python
# mtrpp/evalutaion/eva_semantic.py
sys.path.append("~/music-text-representation-pp")

```

### License
This project is under the CC-BY-NC 4.0 license.

## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.

```
@inproceedings{doh2024enriching,
  title={Enriching Music Descriptions with a Finetuned-LLM and Metadata for Text-to-Music Retrieval},
  author={Doh, SeungHeon and Lee, Minhee and Jeong, Dasaem and Nam, Juhan},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024}.
}
```
