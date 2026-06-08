# Event Burst Trigger: An Availability Backdoor Attack on Event-Based SNN Object Detection

[Jaesun Baek](https://github.com/baekchaesun), Chanwook Lee, and [Eun-kyu Lee](https://sites.google.com/site/inueklee)

[Security Research Lab (SRL)](https://sites.google.com/view/seclabinu/home), Department of Information and Communication Engineering, Incheon National University, Republic of Korea

---

*For repository-related inquiries, please use GitHub Issues or contact `jsbaek@inu.ac.kr`.*

---

## Abstract

Event-based vision and spiking neural networks (SNNs) are increasingly adopted for edge intelligence under strict latency and energy constraints. However, the vulnerability of event-based SNN object detection models to availability backdoor attacks remains insufficiently studied. This paper presents Event Burst Trigger (EBT), an availability backdoor attack targeting SNN-based object detection models.

EBT injects carefully crafted event-based triggers into the training data, which induce temporally concentrated event streams during inference. These burst-like activations increase the number of phantom (i.e., spurious) object candidates, and consequently inflate the computational cost of the post-processing stage, particularly Non-Maximum Suppression (NMS). We evaluate EBT on SpikeYOLO, the state-of-the-art SNN-based object detector, under a poison-only threat model that does not require modifications to the model architecture, loss function, or inference pipeline.

Experimental results show that while detection accuracy remains largely preserved, with mAP@0.5 decreasing by less than 0.099, the latency of the NMS stage increases by up to 38x. This indicates that NMS can become a dominant availability bottleneck in event-based SNN object detection. Experiments on an edge platform further show that the proposed attack elevates baseline resource utilization and reduces scheduling slack without inducing conspicuous peaks in resource usage. In addition, STRIP-based backdoor detection fails to reliably distinguish the proposed attack from benign inputs. These results characterize a previously underexplored availability backdoor threat in event-based SNN object detection systems.

![image](pictures/figure4.png) <br>
![image](pictures/figure3.png)

## Dataset

### Prophesee Gen1 Automotive Detection

The Gen1 Automotive Detection dataset can be downloaded from:

- https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/

**Note**
Event data preprocessing was performed using SpikeYOLO’s `SpikeYOLO_for_Gen1.py`

## Poisoned Dataset Generation

Create a poisoned dataset with:

```bash
python run_poisoning.py
```

The runner creates a separate poisoned dataset directory and keeps the original dataset unchanged. The current version supports the event-noise trigger; single-patch and weighted-patch triggers will be added later.

The following shows the overall poisoning process for the Gen1 dataset.

| Original | Pre-processed | Poisoned |
|---|---|---|
| ![Original](pictures/org_event.gif) | ![Pre-processed](pictures/clean_framed.gif) | ![Poisoned](pictures/poisoned_framed.gif) |

## Train

After generating the poisoned dataset, train SpikeYOLO normally by pointing the dataset configuration to the poisoned dataset path.

```text
original dataset
    -> run_poisoning.py
    -> poisoned dataset
    -> model training without source-code modification
```

## Pre-Trained Model

The pretrained SpikeYOLO weights are hosted in a separate repository. Please refer to the following link for details and downloads:

- https://github.com/BICLab/SpikeYOLO

## Thanks

## Citation

```bibtex
```
