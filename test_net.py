import cv2
import itertools
import logging
import os
import numpy as np
from collections import OrderedDict
import torch
from detectron2.engine import DefaultPredictor
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
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

# class Trainer(DefaultTrainer):
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#         evaluator_list = []
#         evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
#         if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
#             evaluator_list.append(
#                 SemSegEvaluator(
#                     dataset_name,
#                     distributed=True,
#                     num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
#                     ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
#                     output_dir=output_folder,
#                 )
#             )
#         if evaluator_type in ["coco", "coco_panoptic_seg"]:
#             evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
#         if evaluator_type == "coco_panoptic_seg":
#             evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
#         if evaluator_type == "cityscapes_instance":
#             assert (
#                 torch.cuda.device_count() >= comm.get_rank()
#             ), "CityscapesEvaluator currently do not work with multiple machines."
#             return CityscapesInstanceEvaluator(dataset_name)
#         if evaluator_type == "cityscapes_sem_seg":
#             assert (
#                 torch.cuda.device_count() >= comm.get_rank()
#             ), "CityscapesEvaluator currently do not work with multiple machines."
#             return CityscapesSemSegEvaluator(dataset_name)
#         elif evaluator_type == "pascal_voc":
#             return PascalVOCDetectionEvaluator(dataset_name)
#         elif evaluator_type == "lvis":
#             return LVISEvaluator(dataset_name, cfg, True, output_folder)
#         if len(evaluator_list) == 0:
#             raise NotImplementedError(
#                 "no Evaluator for the dataset {} with the type {}".format(
#                     dataset_name, evaluator_type
#                 )
#             )
#         elif len(evaluator_list) == 1:
#             return evaluator_list[0]
#         return DatasetEvaluators(evaluator_list)
#
#     @classmethod
#     def test_with_TTA(cls, cfg, model):
#         logger = logging.getLogger("detectron2.trainer")
#         # In the end of training, run an evaluation with TTA
#         # Only support some R-CNN models.
#         logger.info("Running inference with test-time augmentation ...")
#         model = GeneralizedRCNNWithTTA(cfg, model)
#         evaluators = [
#             cls.build_evaluator(
#                 cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
#             )
#             for name in cfg.DATASETS.TEST
#         ]
#         res = cls.test(cfg, model, evaluators)
#         res = OrderedDict({k + "_TTA": v for k, v in res.items()})
#         return res
#
#     @classmethod
#     def build_optimizer(cls, cfg, model):
#         params = get_default_optimizer_params(
#             model,
#             base_lr=cfg.SOLVER.BASE_LR,
#             weight_decay=cfg.SOLVER.WEIGHT_DECAY,
#             weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
#             bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
#             weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
#             overrides={
#                 "absolute_pos_embed": {"lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.0},
#                 "relative_position_bias_table": {"lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.0},
#             }
#         )
#
#         def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
#             # detectron2 doesn't have full model gradient clipping now
#             clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
#             enable = (
#                 cfg.SOLVER.CLIP_GRADIENTS.ENABLED
#                 and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
#                 and clip_norm_val > 0.0
#             )
#
#             class FullModelGradientClippingOptimizer(optim):
#                 def step(self, closure=None):
#                     all_params = itertools.chain(*[x["params"] for x in self.param_groups])
#                     torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
#                     super().step(closure=closure)
#
#             return FullModelGradientClippingOptimizer if enable else optim
#
#         optimizer_type = cfg.SOLVER.OPTIMIZER
#         if optimizer_type == "SGD":
#             optimizer = maybe_add_gradient_clipping(torch.optim.SGD)(
#                 params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
#                 nesterov=cfg.SOLVER.NESTEROV,
#                 weight_decay=cfg.SOLVER.WEIGHT_DECAY,
#             )
#         elif optimizer_type == "AdamW":
#             optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
#                 params, cfg.SOLVER.BASE_LR, betas=(0.9, 0.999),
#                 weight_decay=cfg.SOLVER.WEIGHT_DECAY,
#             )
#         else:
#             raise NotImplementedError(f"no optimizer type {optimizer_type}")
#         return optimizer

# def visualize_all_keypoints(image, outputs, metadata):
#     """
#     Visualize image with all keypoints, regardless of their 'visibility' flag.
#     """
#     vis = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
#     instances = outputs["instances"].to("cpu")
#
#     if instances.has("pred_keypoints"):
#         keypoints = instances.pred_keypoints
#         for keypoint_instance in keypoints:
#             for idx, (x, y, v) in enumerate(keypoint_instance):
#                 # Draw all keypoints regardless of visibility flag
#                 color = (255, 0, 0)
#                 cv2.circle(image, (int(x), int(y)), 5, color, thickness=2)
#                 #cv2.putText(image, f"{idx}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#
#     # Draw the rest of the predictions
#     output_vis = vis.draw_instance_predictions(instances)
#     return output_vis.get_image()[:, :, ::-1]


def setup(args):
    """
    Create configs and perform basic setups.
    """
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog
    address_test = 'datasets/coco/'
    address_train = 'datasets/coco/'
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
    register_coco_instances("experiment_test", {}, os.path.join(address_test, "annotations/person_keypoints_train2017.json"),
                            os.path.join(address_test, "train2017"))
    sample_metadata = MetadataCatalog.get("experiment_test")
    dataset_dicts = DatasetCatalog.get("experiment_test")

    MetadataCatalog.get("experiment").set(
        # thing_classes=["bad", "good"],
        # keypoint_names=["grab_point"],
        keypoint_flip_map=[],
        evaluator_type='coco',  # Assuming COCO evaluator is appropriate
        json_file=os.path.join(address_test, "annotations/person_keypoints_train2017.json"),
        image_root=os.path.join(address_test, "train2017")
    )

    cfg = get_cfg()
    # cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    add_swint_config(cfg)
    # cfg.SOLVER.OPTIMIZER = "AdamW"

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.MODEL.WEIGHTS = "/home/fariborz_taherkhani/SwinT_detectron2-main_one_package/output/model_final.pth"
    # cfg.MODEL.WEIGHTS = '/home/fariborz_taherkhani/SwinT_detectron2-main_one_package_freeze_second_v/output_before_ck_modificataion/Before_PTO/model_0854999.pth'
   # cfg.MODEL.WEIGHTS = '/home/fariborz_taherkhani/SwinT_detectron2-main_one_package_freeze_second_v/output/model_0028999.pth'

    # cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.1]

    # print(cfg.ROI_BOX_HEAD)

    # cfg.SOLVER.AMP.ENABLE = False
    # cfg.freeze()
    # default_setup(cfg, args)

    cfg.merge_from_file("configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml")  # replace with your model's config path
    #cfg.MODEL.WEIGHTS = "/home/fariborz_taherkhani/SwinT_detectron2-main_one_package_freeze_second_v/output/model_0139999.pth"
    #cfg.MODEL.WEIGHTS = 'output/model_0024999.pth'
    #cfg.MODEL.WEIGHTS='output/model_0254999.pth'
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17
    cfg.MODEL.WEIGHTS = '/home/fariborz_taherkhani/keypint_train/output/model_final.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17
    # cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.DEVICE = "cuda:1"
    # set threshold for this model
    # cfg.freeze()
    predictor = DefaultPredictor(cfg)
    # save_ad = '/home/fariborz_taherkhani/SwinT_detectron2-main/saved_imgs'
    # folder_path = '/home/fariborz_taherkhani/SwinT_detectron2-main_one_package_freeze_second_v/commerce_test_imgs'
    folder_path = '/home/fariborz_taherkhani/keypint_train/good_samples'
    # folder_path = '/home/fariborz_taherkhani/Downloads/copty'
    #folder_path ='/home/fariborz_taherkhani/new_data_29_march/z_camera_data_updated_name'
    #folder_path = '/home/fariborz_taherkhani/test/images'
    #folder_path = '/home/fariborz_taherkhani/left_confirm_images'
    #folder_path = '/home/fariborz_taherkhani/left_confirm_images'
    #folder_path ='/home/fariborz_taherkhani/Bereket_S_data'
    #image_ad = '/home/fariborz_taherkhani/edge/'
    counter = 0
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.endswith('.jpg') or filename.endswith('.png'):
            #filename ='left_1680712522965841424.jpg'
            print(filename)
            # Read the image using OpenCV
            #filename ='left_1680801979379691428.jpg'
            image_path = os.path.join(folder_path, filename)
            # image_path ='img/crop_b.jpg'
            im = cv2.imread(image_path)
            if im is not None:
                if im.shape[0] > 0 and im.shape[1] > 0:
                    #im = cv2.resize(im, (1008, 624), interpolation=cv2.INTER_CUBIC)
                    # brightness_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
                    # beta_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
                    # for brightness_factor in brightness_factors :
                    #   for beta_factor in beta_factors:
                    # # brightness_factor = 0.5  # Adjust this value to change brightness
                    #     im = cv2.convertScaleAbs(im_original, alpha=brightness_factor, beta=beta_factor)
                    # # plt.imshow(im)
                    # # plt.show()
                    outputs = predictor(im)

                    # print(outputs['pre'])
                    # instances = outputs['instances']
                    # predicted_masks = instances.pred_masks
                    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
                                       )
                    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    #image_saved = image_ad + str(counter)+filename
                    counter +=1
                    #cv2.imwrite(image_saved, v.get_image())
                    # image_saved = image_ad_ + str(counter) + 'original'+ filename
                    # cv2.imwrite(image_saved, im)
                    #
                    # # cv2_imshow(v.get_image()[:, :, ::-1])
                    plt.imshow(v.get_image())

                    # instances = outputs["instances"].to("cpu")
                    #
                    # if instances.has("pred_keypoints"):
                    #     keypoints = instances.pred_keypoints
                    #     for keypoint_instance in keypoints:
                    #         x, y = keypoint_instance[0][0], keypoint_instance[0][1]
                    #         # for idx, (x, y, v) in enumerate(keypoint_instance):
                    #         plt.plot(x, y, 'ro')  # This plots the keypoints as red circles
                    #             # Draw all keypoints regardless of visibility flag
                    #             # color = (255, 0, 0)
                    #             # cv2.circle(image, (int(x), int(y)), 5, color, thickness=2)
                    #             # # cv2.putText(image, f"{idx}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    plt.show()

    # for filename in os.listdir(folder_path):
    #     # Check if the file is an image
    #     if filename.endswith('.jpg') or filename.endswith('.png'):
    #         # Read the image using OpenCV
    #         image_path = os.path.join(folder_path, filename)
    #         im = cv2.imread(image_path)
    #         im = cv2.resize(im, (640, 480), interpolation=cv2.INTER_LINEAR)
    #         # # plt.imshow(im)
    #         # # plt.show()
    #         # outputs = predictor(im)
    #         # masks = outputs["instances"].pred_masks.cpu().numpy()
    #         # shape = masks.shape
    #         # for i in range(shape[0]):
    #         #     mask = np.uint8(masks[i, :, :])  # Ensure the mask is of type uint8
    #         #     binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    #         #
    #         #     # Find contours in the binary mask
    #         #     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         #
    #         #     # Draw boundaries around each contour
    #         #     for contour in contours:
    #         #         cv2.drawContours(im, [contour], -1, (0, 255, 0), thickness=2)
    #         # path_ = os.path.join(save_ad, filename)
    #         # cv2.imwrite(path_, im)
    #         # plt.imshow(im)
    #         # plt.show()
    #         v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    #                        )
    #         v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #         image_test = v.get_image()
    #         mask_only_visualization = v.get_image()[:, :, ::-1]  # Get the RGB visualization
    #         mask_only_visualization[v.get_image()[:, :, 0] == 0] = 0  # Set bounding box pixels to black (remove them)
    #         # cv2_imshow(v.get_image()[:, :, ::-1])
    #         plt.imshow(mask_only_visualization)
    #         plt.show()

    return cfg


def main(args):
    cfg = setup(args)

    # if args.eval_only:
    #     model = Trainer.build_model(cfg)
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     res = Trainer.test(cfg, model)
    #     if cfg.TEST.AUG.ENABLED:
    #         res.update(Trainer.test_with_TTA(cfg, model))
    #     if comm.is_main_process():
    #         verify_results(cfg, res)
    #     return res
    #
    # """
    # If you'd like to do anything fancier than the standard training logic,
    # consider writing your own training loop (see plain_train_net.py) or
    # subclassing the trainer.
    # """
    # trainer = Trainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    # if cfg.TEST.AUG.ENABLED:
    #     trainer.register_hooks(
    #         [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
    #     )
    # return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.config_file = 'configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
