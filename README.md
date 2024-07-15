# Vitis AI & Versal YOLO Model Compatibility

This guide aims to provide the necessary information and mitigation tactics to setup and run YOLO models on the Versal Platform VCK190. This guide focuses on PyTorch but detailed explanations will be provided so you can use your framework of choice accordingly.

> [!Note]
> Models and Datasets are from our ["Evaluation of Resource-Efficient Crater Detectors on Embedded Systems"](https://github.com/billpsomas/mars_crater_detection/) paper for illustration purposes.

## Table of contents
1. [Issues](#what-is-the-issue)
2. [Layer Fix](#layer-related-fixes)
3. [Model definition fixes](#model-definition-fixes)
4. [Vitis AI](#vitis-ai)
5. [Versal](#versal)

## What is the issue?
Vitis-AI has a limited amount of operations supported detailed here:
https://docs.amd.com/r/3.0-English/ug1414-vitis-ai/Currently-Supported-Operators

(Newer versions of VitisAI may support more features however since we are working with VCK190 board, we must use VAI 3.0)

Due to those limitations, SiLU layers commonly found on YOLO models are unsupported. Moreover, operations like transpose and permute are also unsupported leading to further issues and the inability to quantize and convert the model accordingly. Transpose and permute functions are especially a problem because VitisAI attemts to convert PyTorch to XIR where there is a shape mismatch (BCHW to BHWC).

Finally, YOLO models made with the ultralytics library, feature different functions than models using pure PyTorch. For example, model(input) or model.train have completely different function calls and behaviors where VitisAI cannot get the model trace or adjust accordingly.

We will have  to fix these issues before we are able to move on VitisAI and then Versal where we will make small adjustments.

> [!Important]
> The following fixes might not work on newer ultralytics and VitisAI versions. Make sure you are aware of the version you are using and any further changes required.


## Layer related fixes

Fixing the SiLU layer compatibility issue is rather easy. We can replace all the SiLU layers with LeakyReLU with 0.1 slope due to their extremely similar behavior. Depending on your application you might not need to retrain the model after this change however its highly recommended.

While one layer replacement will not affect the accuracy | precision significantly, YOLO models feature a lot them and the losses are very noticable.

[YOLO Converter.py](Resources/Model_Fixes/YOLO_converter.py) is provided for this exact replacement of layers using PyTorch.

In this script we also fix the "issue" of ultralytics models. If you load the best.pt file of the exported model, you will find its a dictionary containing the model, the training parameters and the training logs. Keeping the model from said dictionary, we find that instead of a YOLO model class, its a DetectionModel class which follows typical PyTorch Model classes and retains the functionality of the YOLO model. Leveraging this feature, we replace the layers (with slope 0.1015625 instead of 0.1 as a requirement for VCK190) and save the new DetectionModel class model.


At this stage you could retrain the model with a conventional pytorch training function.

## Model definition fixes

> [!Caution]
> From this point forward different versions of ultralytics have different implementations. The proposed fix will most likely not work on your version. (Latest working version: 8.2.37,  Latest fully tested: 8.1.47 )

All thats left now is fixing the internal class functions.
We will need to make 2 total changes. First one will be on the DFL forward function layer and the second one on the Detect class forward function where the bounding box generation will have to be removed. The generation will occur on the postprocessing stage where we will also have a configuration file from our model with all the parameters required, generated right before the VitisAI XIR conversion.

> [!Warning]
> The following changes have been tested only on ultralytics package version: 8.1.47


Specific class definition files are not provided due to the amount of dependencies and functions used in the ultralytics package. Instead, you could temporarily modify the installed package with the following changes.

Further details on where to apply those changes are provided in the next stage where a docker container is used for VitisAI and any changes within it are removed when exiting.


1st change: DFL Layer (ultralytics/nn/modules/block.py)

Old DFL forward function:
```
def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

```

Updated function:
```
def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(torch.cat([x.view(b,4,self.c1,a).split(1,dim=2)[i].squeeze(2).unsqueeze(1) for i in range(x.view(b,4,self.c1,a).size(2))], dim=1).softmax(1)).view(b,4,a)
```

> [!Note]
> The exact same functionality and result are retained while omitting transpose to fix our issue.


2nd change: Detection Class (ultralytics/nn/modules/head.py)

Old forward function:
```
def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
```


New forward function:
```
def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.shape = shape
        return x
```

Here we fully removed the bbox generation along with training path checks, anchor generation and export functionality.

Anchors and Bounding Boxes will be transferred to our post-processing stage while training and exporting will never be used since the sole purpose of our changes is to quantize and compile our model into a Versal DPU ready state.


## Vitis AI 

> [!Important]
> We are using VitisAI 3.0. This guide might not work on newer versions.

### Setup

1. Install [Docker](https://docs.docker.com/engine/install/): 
2. Verify its installation:
    * docker run hello-world <br />  
    * docker --version

3. Pull the [Vitis-AI container](https://xilinx.github.io/Vitis-AI/3.0/html/docs/install/install.html). Note: The VitisAI container is independent to the Vitis-AI 3.0 repo. 
    > docker pull xilinx/vitis-ai-pytorch-cpu:latest <br />
4. Now we are ready to pull the github repo:
    > git clone --branch 3.0 https://github.com/Xilinx/Vitis-AI.git


After downloading the repository you can start Vitis-AI:
> ./docker_run.sh xilinx/vitis-ai-python-cpu:latest

At this point Vitis AI is up and running.

### Implementation

First enable the pytorch environment with
``conda activate vitis-ai-pytorch``.

Now install the ultralytics package and make the changes mentioned in the previous section.\
``pip install ultralytics==8.1.47``

Once the package is installed you can edit the files with your terminal text editor (eg. vim) at /home/vitis-ai-user/.local/lib/python3.8/site-packages/ultralytics/


The DFL changes should be done at ``/home/vitis-ai-user/.local/lib/python3.8/site-packages/ultralytics/nn/modules/block.py``

While the class changes should happen at ``/home/vitis-ai-user/.local/lib/python3.8/site-packages/ultralytics/nn/modules/head.py``


Move [vai_q_yolo.py](resources/vitis-ai/vai_q_yolo.py) to your Vitis-AI installation folder along with your model and dataset.

Let's breakdown its usage:

* You can run it as follows:\
``python vai_q_yolo.py --model_name <file_name.file_ext> --batch_size <batch_num> --quant_mode <calib|test> --target <DPU_ARCH> --deploy``\
where:\
**model_name**: is the file name (eg. yolov8n.pt) or including the path name if its in a different folder\
**quant_mode**: should be "calib" first to generate the necessary structure and "test" for the finalized step.\
**batch_size**: your preferred batch size. Keep in mind the RAM resources used since if they are exceeded the program will get killed. When in test mode, the batch size ***must*** always be 1.\
**target**: your DPU architecure. In our case for Versal AI Core Series VCK190 Evaluation Kit, its "DPUCVDX8G_ISA3_C32B6". It's extremely important to define the correct target.\
**deploy**: Should only be used with test mode to export the finalized xmodel which is used up next.

The full commands in our case would be:
``python vai_q_yolo.py --model_name yolov8n.pt --batch_size 32 --quant_mode calib --target DPUCVDX8G_ISA3_C32B6``\
and ``python vai_q_yolo.py --model_name yolov8n.pt --batch_size 1 --quant_mode test --target DPUCVDX8G_ISA3_C32B6 --deploy``

This script loads the model and validation dataset, exports a configuration file for post-processing and quantizes the model using internal implementations. Calibration and Testing are achieved by simply running inference on the validation set. There is no further processing of the results to save time and resources. For the dataset definition we use the LoadImagesAndVideos dataset class from ultralytics with a txt file containing the path to our validation images.

When running the script a quantized_result folder is created containing all necessary calibration and export files along with the configuration one.

#### What should you change to run your own model?

In line 258 you should change the input shape to your corresponding dataset size.\
In line 226 you also need to change the filepath of your validation files txt for them to load.

> [!Note]
> When running calibration there is a chance you get an error about parameters or DetectionModel.py missing. Simply rerun the command and there should be no issues.

If for any reason you choose to run the script with the model and dataset provided as is, don't forget to extact the [val.tar](resources/vitis-ai/data/val.tar).

Assuming now that calibration and testing are done, there should be 2 important files in the quantized_result folder:

* \<model_name>_config_no_srd_reg_nc_dfl.pkl
* \<model_name>.xmodel

So now all that's needed is to compile the model using vai_c_xir.
Run:
> /workspace/board_setup/vck190/host_cross_compiler_setup.sh

and after the automated installation, run: 
``unset LD_LIBRARY_PATH`` and\
``source $install_path/environment-setup-cortexa72-cortexa53-xilinx-linux``

where *$install_path* would be the installation path of the script. Just check the final output, it should have the full command including the path.

Finally run ``vai_c_xir -x quantize_result/<model_name>.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o ./ -n <new_model_name>``

This will finalize the model to run on our Versal Board generating a new .xmodel file named *\<new_model_name>*. If you have a different board, you must absolutely change the architecture json with your corresponding one.

## Versal

Moving to our board, we transfer all files in the [versal_board](resources/versal_board/) folder.
These now include our finalized model and the test set.

Running inference here is quite easy. Simply run ``./run_per_model.sh yolov8n`` and wait for a results folder to be created.

The results however are unusable at this stage since we care about the bounding boxes which aren't calculated. Generally, we can't calculate the boxes from the DPU rather we choose to "export" the raw tensors in a text file to then be processed since we know they have fixed shapes of [65, 32, 32], [65, 16, 16] and [65, 8, 8]

> [!Note]
> This process is done by iterating over the tensors and writing them to a file which generally inefficient. There might be improvements of this process at a later date drastically reducing the latency.


## Postprocessing

Now we are able to use our configuration file along with the predicted raw tensors to calculate the boxes and evaluate the accuracy and precision. We provide 2 scripts for this reason, 1 for the bbox calculation and the second to convert into coco format for easy evaluation. We also include our coco format evaluator and the ground truths for our used dataset. 


> [!Note]
> Move your configuration file from VitisAI (eg. yolov8n_config_no_srd_reg_nc_dfl.pkl) in the results folder before continuing.

* [postproc.py](resources/postproc/postproc.py) runs as: ``python postproc.py <model_name>_results/ <model_name> <iou>``
Eg. python postproc.py yolov8n_results/ yolov8n 0.1

where iou is the IOU for the non-max suppression. It should be adjusted according tou your desired precision metrics. For example if you need a mAP 40 you have to use an IOU under 0.4.

It processes all the tensors and creates an output file *output_boxes_\<model_name>.txt* with all boxes in XYWH format. 

* [coco_prep.py](resources/postproc/coco_prep.py) runs as ``python coco_prep.py <model_name>`` Eg. python coco_prep.py yolov8n

It loads all calculated boxes and fuses them in a json dictionary in XYWH for coco evaluation in an exported file *coco-preped-\<model_name>.json*.


Finally, [evaluate.py](resources/postproc/evaluate.py) loads the ground truths (gt_eval.json) and our file as an argument (``python evaluate.py coco-preped-yolov8n.json``) calculating our corresponding metrics.