import os

import torch

from foolbox.attacks import LinfPGD
import eagerpy as ep

from environment_setup import PROJECT_ROOT_DIR


class Attack:
    def __init__(self):
        pass

    def instantiate_attack(self):
        """
        Attack the model
        :return: NotImplementedError
        """
        raise NotImplementedError

    def attack_description(self):
        """
        String description for the attack
        :return: NotImplementedError
        """
        raise NotImplementedError

    def get_use_case_loss_fn(self, model, labels):
        """
        Selected between reid/attr tasks for proper loss computation. We can switch between cross entropy and bce
        depending upon the task (reid/attr respectively). One can extend this class to accommodate more loss
        functions.
        :param model: Foolbox model :param labels: labels for the inputs
        :return: cross_entropy/bce_with_logits loss
        """

        # can be overridden by users
        def loss_fn(inputs):
            logits = model(inputs)
            if self.task_type == 'reid':
                return ep.crossentropy(logits, labels).sum()
            else:
                # binary cross entropy case in here
                return ep.astensor(
                    torch.nn.functional.binary_cross_entropy_with_logits(logits.raw, labels.raw.to(torch.float),
                                                                         reduction="sum"))

        return loss_fn
