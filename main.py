# Base class following the Template Design patter
import argparse
import json
import os

from dataset.AttributeAlterationDataset import get_data_loader
from dataset.FaceReIdDataset import get_reid_data_loader
from environment_setup import PROJECT_ROOT_DIR
from use_case.FaceReIdentification import FaceReIdentificationTargetedCriterion, FaceReIdentificationUntargetedCriterion
from wrapper.pytorch_to_foolbox import PyToFool
from model.pytorch_model.dummy_network import DummyNetwork
from tqdm import tqdm
import torch
from attacks.attack_list import get_attacks
from model.pytorch_model.inception_resnet_v1 import InceptionResnetV1
from use_case.AttributeAlteration import AttributeAlterationCriteria

FIXED_EPSILON = 8. / 255


class AttackExecuter:
    def __init__(self):
        pass

    def compute_original_accuracy(self, device, dataloader, foolbox_model, target_label):
        """
        Computes the accuracy of model before any perturbations
        :param device: gpu/cpu
        :param dataloader: dataloader for the task
        :param foolbox_model: model wrapped in foolbox framework
        :param target_label: The label to be used in targeted attack. If None -> Untargeted. Default: None
        :return: accuracy value
        """
        # Compute the initial accuracy
        total = 0
        correct = 0
        for idx, (_, images, labels) in enumerate(tqdm(dataloader)):
            total += images.size(0)
            images, labels = images.to(device), labels.to(device)
            if target_label is not None:
                labels = torch.ones_like(labels) * int(target_label)
            correct += dataloader.dataset.accuracy(foolbox_model, images, labels)
        acc = correct / total
        return acc

    def load_model(self, checkpoint_dir, task_type, model_numer=8):
        """
        Loads a pretrained model.
        :param checkpoint_dir: folder to check model weights from
        :param task_type: ReID/Attr
        :param model_numer: model identifier. We give a base name to model and identify different instances using `model_number`.(Similar to SAFAIR Contest nomenclature)
        :return: (model, bounds) model and bound of values in input
        """
        # Load the pretrained model
        model, bounds = self.select_model_type(task_type=task_type, checkpoint_dir=checkpoint_dir)
        model.load(model_number=model_numer, checkpoint_dir=checkpoint_dir)
        model.eval()
        return model, bounds

    def select_model_type(self, task_type, checkpoint_dir):
        """
        Helper function for loading model weights. Please change this function when defining new networks accordingly.
        :param task_type: attr/reid
        :param checkpoint_dir: folder to check model weights from
        :return: (model, bounds) model and bound of values in input
        """
        if task_type == 'attr':
            attr_network = DummyNetwork(name="step_")
            bounds = (0, 1)
            return attr_network, bounds
        else:
            num_classes = 5304
            dropout_prob = 0.6
            classify = True
            bounds = (-1, 1)
            name = 'facenet'
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pretrained = 'vggface2'
            model = InceptionResnetV1(name=name, pretrained=pretrained, classify=classify, num_classes=num_classes,
                                      dropout_prob=dropout_prob, device=device, pre_train_wt_folder=checkpoint_dir)
            return model, bounds

    def convert_model(self, model, bounds):
        """
        Function to help in conversion of model to Foolbox.
        :param model: model defined in PyTorch (can be extended to Tensorflow in future)
        :return: foolbox model
        """
        # Initialize the converter
        converter = PyToFool(model=model, bounds=bounds)
        fmodel = converter()
        return fmodel

    def create_task_criterion(self, task_type='attr', target_label=None):
        """
        A task specific metric selector. This needs to be done to work properly with Foolbox
        :param task_type: attr/reid
        :return: Misclassification/MultiLabelMisclassification class
        """
        if task_type == 'attr':
            return AttributeAlterationCriteria.get_criterion()
        if target_label is not None:
            return FaceReIdentificationTargetedCriterion.get_criterion()
        return FaceReIdentificationUntargetedCriterion.get_criterion()

    def create_attack(self, task_type):
        """
        Function to load all the attacks defined.
        :param task_type: reid/attr
        :return: a list of attacks as defined in `attaks/attack_types` folder
        """
        return get_attacks(task_type=task_type)

    def compute_adv_acc(self, dataloader, device, task_criterion, foolbox_model, attack, target_label=None):
        """
        Computes the adversarial accuracy of model
        :param dataloader: dataloader for the task
        :param device: gpu/cpu
        :param task_criterion: Misclassification/MultiLabelMisclassification
        :param foolbox_model: model wrapped in foolbox framework
        :param attack: Specific attack defined using foolbox framework
        :param target_label: The label to be used in targeted attack. If None -> Untargeted. Default: None
        :return: adversarial accuracy of the model
        """
        # Now run it for the new set of input
        total = 0
        adv_correct = 0
        for idx, (_, images, labels) in enumerate(tqdm(dataloader)):
            total += images.size(0)
            images, labels = images.to(device), labels.to(device)
            if target_label is not None:
                labels = torch.ones_like(labels) * int(target_label)
            criterion = task_criterion(labels)
            raw, clipped, is_adv = attack(foolbox_model, images, criterion=criterion, epsilons=FIXED_EPSILON)
            adv_correct += dataloader.dataset.accuracy(foolbox_model, raw, labels)
        adv_acc = adv_correct / total
        return adv_acc

    def create_dataloader(self, task_type, batch_size):
        """
        Creates the dataloader.
        :param task_type: reid/attr
        :param batch_size: size of batch used for loading the values
        :return: dataloader
        """
        if task_type == 'attr':
            return get_data_loader(batch_size=batch_size, split="test")
        return get_reid_data_loader(batch_size=batch_size, split="test", num_workers=0)

    def execute_attack(self, args):
        """
        The main execution method
        :param args: argparse object
        :return: None
        """
        json_data = {}
        dataloader = self.create_dataloader(task_type=args.task_type, batch_size=args.batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = os.path.join(PROJECT_ROOT_DIR, args.checkpoint_dir)
        model, bounds = self.load_model(checkpoint_dir=checkpoint_dir, task_type=args.task_type,
                                        model_numer=args.model_number)
        foolbox_model = self.convert_model(model=model, bounds=bounds)
        initial_acc = self.compute_original_accuracy(device=device, dataloader=dataloader, foolbox_model=foolbox_model,
                                                     target_label=args.target_label)
        json_data['orig_acc'] = initial_acc
        attack_class_list = self.create_attack(task_type=args.task_type)

        for attack_class in attack_class_list:
            try:
                task_criterion = self.create_task_criterion(task_type=args.task_type, target_label=args.target_label)
                attack = attack_class.instantiate_attack()
                adv_acc = self.compute_adv_acc(dataloader=dataloader, device=device, task_criterion=task_criterion,
                                               foolbox_model=foolbox_model, attack=attack,
                                               target_label=args.target_label)
                json_data[attack_class.attack_description()] = adv_acc
            except TypeError as a:
                print(a)
            except ValueError as v:
                print(f"Targeted Attack not defined for {attack_class.attack_description()}. Skipping!!")
        print(json.dumps(json_data, indent=4))


def parse_args():
    """parse input arguments
    :return: args
    """

    parser = argparse.ArgumentParser(description='adversarial ml')
    parser.add_argument('--batch_size', help='Batch size. Default: 32', type=int, default='32')
    parser.add_argument('--task_type', help='reid/attr. Default:attr', type=str, default='attr')
    parser.add_argument('--checkpoint_dir', help='Directory to load model from. Default: saved_models/', type=str,
                        default='saved_models/')
    parser.add_argument('--model_number',
                        help='Model to test final results on. Needed for test and adv mode. Default: 0', type=int,
                        default='0')
    parser.add_argument('--target_label', help='Target label to use in targeted attack. An integer value. Default:None',
                        type=str, default=None)
    # Now parse all the values
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.target_label is not None:
        print(f"Using TARGETED attack with target label {args.target_label}")
        assert args.task_type != 'attr', "Targeted Attack only defined for Re-identification task"
    else:
        print("Using UN-TARGETED attack")
    executor = AttackExecuter()
    executor.execute_attack(args)
