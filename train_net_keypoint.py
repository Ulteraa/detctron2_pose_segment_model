#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import os
from collections import OrderedDict
import torch
# import wandb
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)


from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params

from swint import add_swint_config

# import json
# json_address = '/home/fariborz_taherkhani/keypint_train/annotations/person_keypoints_train2017.json'
# with open(json_address, 'r') as file:
#     data = json.load(file)  # Load JSON data

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            overrides={
                "absolute_pos_embed": {"lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.0},
                "relative_position_bias_table": {"lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.0},
            }
        )

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optimizer_type == "AdamW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, betas=(0.9, 0.999),
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog
    address_test = '/home/fariborz_taherkhani/keypint_train/datasets/coco'
    address_train = '/home/fariborz_taherkhani/keypint_train/datasets/coco'
    #
    #
    register_coco_instances("experiment", {}, os.path.join(address_train, "annotations/person_keypoints_train2017.json"),
                            os.path.join(address_train, "train2017"))
    # address_train = '/home/fariborz_taherkhani/Combined_Dataset/'
    #
    # register_coco_instances("experiment", {}, os.path.join(address_train, "combine_dataset_train.json"),
    #                         os.path.join(address_train, "images"))

    sample_metadata = MetadataCatalog.get("experiment")
    dataset_dicts = DatasetCatalog.get("experiment")
    #####################################################################################################
    register_coco_instances("experiment_test", {}, os.path.join(address_test, "annotations/person_keypoints_val2017.json"),
                            os.path.join(address_test, "val2017"))
    sample_metadata = MetadataCatalog.get("experiment_test")
    dataset_dicts = DatasetCatalog.get("experiment_test")

    MetadataCatalog.get("experiment").set(
        keypoint_names=[
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ],
        keypoint_flip_map=[
            ("left_eye", "right_eye"), ("left_ear", "right_ear"),
            ("left_shoulder", "right_shoulder"), ("left_elbow", "right_elbow"),
            ("left_wrist", "right_wrist"), ("left_hip", "right_hip"),
            ("left_knee", "right_knee"), ("left_ankle", "right_ankle")
        ],
        evaluator_type="coco",  # Since you are using the COCO dataset
        thing_classes=["person"]  # Assuming you're detecting persons with keypoints
    )

    # MetadataCatalog.get("experiment").set(
    #     # thing_classes=["bad", "good"],
    #     # keypoint_names=["grab_point"],
    #     keypoint_flip_map=[],
    #     evaluator_type='coco',  # Assuming COCO evaluator is appropriate
    #     json_file=os.path.join(address_test, "annotations/person_keypoints_train2017.json"),
    #     image_root=os.path.join(address_test, "train2017")
    # )

    cfg = get_cfg()
    # cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 80
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    add_swint_config(cfg)
    # cfg.SOLVER.OPTIMIZER = "AdamW"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.MODEL.WEIGHTS = "/home/fariborz_taherkhani/SwinT_detectron2-main_one_package/output/model_final.pth"
    #cfg.MODEL.WEIGHTS = '/home/fariborz_taherkhani/SwinT_detectron2-main_one_package_freeze_second_v/output_before_ck_modificataion/Before_PTO/model_0854999.pth'
    cfg.MODEL.WEIGHTS = '/home/fariborz_taherkhani/keypint_train/output/model_final.pth'
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17
    # cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.DEVICE = "cuda:1"
   




    # cfg.TEST.EVAL_PERIOD = 100

    # print(cfg.ROI_BOX_HEAD)

    # cfg.SOLVER.AMP.ENABLE = False
    # cfg.freeze()
    # default_setup(cfg, args)

    return cfg

def main(args):
    cfg = setup(args)
    print(cfg.TEST.KEYPOINT_OKS_SIGMAS)
    args.eval_only = True
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        print(cfg.TEST.KEYPOINT_OKS_SIGMAS)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        print(cfg.TEST.KEYPOINT_OKS_SIGMAS)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    model_ = trainer.model
    print(cfg.MODEL)
    # print('here we are going to print parameters')
    # print(cfg.MODEL.FPN.OUT_CHANNELS)
    # print(cfg.MODEL.ROI_HEADexitS.NUM_CLASSES)
    # print(cfg.MODEL.RPN.POST_NMS_TOPK_TEST)
    # print(cfg.MODEL.RPN.NMS_THRESH)
    #
    # # these are prev ones
    # print(cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION)
    # print(cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO)
    # print(cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE)
    # print(cfg.TEST.DETECTIONS_PER_IMAGE)
    # print(cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
    # print(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    # print(cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION)
    # print(cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO)
    # print(cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE)
    #
    # print('here we are going to print parameters')

    # for name, param in model_.named_parameters():
    #     if ('backbone.bottom_up' in name):
    #         param.requires_grad = False

    print('here we are gonna test name ')

    for name, param in model_.named_parameters():
        if ('backbone.bottom_up' in name):
            param.requires_grad = False
    for name, param in model_.named_parameters():
         if ('backbone.bottom_up.norm' in name):
             param.requires_grad = True
    print('here we are gonna test name ')
    for name, param in model_.named_parameters():
          if (param.requires_grad):
              print(name)
                           # print('the parametrs that we need for tensor RT')

    # print(cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION)
    # print(cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO)
    # print(cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE)
    # print(cfg.TEST.DETECTIONS_PER_IMAGE)
    # print(cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
    # print(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    # print(cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION)
    # print(cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO)
    # print(cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE)

    # print('the parametrs that we need for tensor RT')

    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.config_file = 'configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml'
    #wandb.init(project="detecron_wandb_wp", sync_tensorboard=True)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
