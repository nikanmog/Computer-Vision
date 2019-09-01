# Training Documentation
This folder contains all training related files. The training was performed using the Tensorflow object detection library and an azure virutal machine with 4 K80 GPUs.

1. Download the tensorflow object detection library and save it to C:/
2. Create a virtual environment (here we use the conda environment 'AzureML') and install all required libraries (Tensorflow, CUDA and cuDNN)
3. Follow all steps described in my [jupyter notebook](data_transformation.ipynb) to prepare the data set and conduct the necessary data transformations (all steps are explained there in detail with code & markdown)
4. Run the following code
5. Wait...
```bash
conda activate AzureML
set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
cd C:\tensorflow1\models\research\object_detection
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssdlite_pipeline.config
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssdlite_pipeline.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
tensorboard --logdir=training
}
```
See more detailed set up instructions [here](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#1-install-anaconda-cuda-and-cudnn)
