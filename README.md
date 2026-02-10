Multi-scale network based on dynamic graph grouping and attention aggregation for surface point cloud segmentation of parts

## Usage
### Installation
```
pip install -r requirements.txt
cd openpoints/cpp

cd pointnet2_batch
python setup.py install
cd ..

cd pointops
python setup.py install
cd ..

cd chamfer_dist
python setup.py install
cd ..

cd emd
python setup.py install
cd ..

cd grouping
python setup.py install
cd ..

```
## Result
```
The dataset and model is available at https://huggingface.co/zz3503/DGGAA2
```
### shapenetpart
```
python examples/shapenetpart/main.py --cfg=cfgs/shapenetpart/dggaa.yaml mode=test pretrained_path=E
:\毕业论文\DGGAA2\log\shapenetpart\shapenetpart-train-pointnext-s_c64-ngpus4-seed3050-20260108-010351-GG4M5SotyLeHoif2DMoanb20260108-1
32740-GyZhm3hURMd8fAXTT7NxfK\checkpoint\shapenetpart-train-pointnext-s_c64-ngpus4-seed3050-20260108-010351-GG4M5SotyLeHoif2DMoanb_ckpt
_best.pth

Instance mIoU 86.91, Class mIoU 84.87, 
 Class mIoUs tensor([85.7655, 86.6499, 87.7371, 82.2919, 91.8200, 81.4489, 92.2530, 87.7847,
        85.5389, 96.0213, 76.0754, 96.1415, 83.7664, 64.3933, 76.3094, 83.8429],
       device='cuda:0')
```
### indpart
```
python compare/ours.py

times : 1.2753541995952655
{'accuracy': 0.8936597980520125, 'precision': 0.8307798605042584, 'recall': 0.8890691813154992, 'f1': 0.7917245299778042, 'jaccard': 0.7513464407306353}
```