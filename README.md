# SFTrack
Official Repository of "*SFTrack: A Robust Scale and Motion Adaptive Algorithm for Tracking Small and Fast Moving Objects*" (IROS 2024)



## Abstract

![Motion-Aware Heatmap Regression](Figures/Front_Image.png)


This paper addresses the problem of multi-object tracking in Unmanned Aerial Vehicle (UAV) footage.
It plays a critical role in various UAV applications, including traffic monitoring systems and real-time suspect tracking by the police.
However, this task is highly challenging due to the fast motion of UAVs, as well as the small size of target objects in the videos caused by the high-altitude and wide angle views of drones.
In this study, we thus introduce a simple yet more effective method compared to previous work to overcome these challenges. Our approach involves a new tracking strategy, which initiates the tracking of target objects from low-confidence detections commonly encountered in UAV application scenarios. Additionally, we propose revisiting traditional appearance-based matching algorithms to improve the association of low-confidence detections. To evaluate the effectiveness of our method, we conducted benchmark evaluations on two UAV-specific datasets (VisDrone2019, UAVDT) and one general object tracking dataset (MOT17). The results demonstrate that our approach surpasses current state-of-the-art methodologies, highlighting its robustness and adaptability in diverse tracking environments. Furthermore, we have improved the annotation of the UAVDT dataset by rectifying several errors and addressing omissions found in the original annotations. We will provide this refined version of the dataset to facilitate better benchmarking in the field.



## News
- (**Soon**) Codes are available.
- (2024.11) Refined UAVDT dataset annotations are available.
- (2024.06) Our paper is accepted by **IROS 2024** as selected **Oral**!

## Citation
```bibtex
@article{song2024sftrack,
  title={SFTrack: A Robust Scale and Motion Adaptive Algorithm for Tracking Small and Fast Moving Objects},
  author={Song, InPyo and Lee, Jangwon},
  journal={arXiv preprint arXiv:2410.20079},
  year={2024}
}
```