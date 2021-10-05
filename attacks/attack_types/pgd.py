import torch

from attacks.base import Attack
from foolbox.attacks import LinfPGD
import eagerpy as ep


class LinfPGDAttack(Attack):
    """
    Creates an Linf Projected Gradient Attack
    """
    def __init__(self, task_type):
        super(LinfPGDAttack, self).__init__()
        self.task_type = task_type

    def instantiate_attack(self):
        """
        Create an instance of the Foolbox LinfPGD attack
        :return: foolbox attack instance
        """
        self.attack = LinfPGD()
        # monkey patching
        self.attack.get_loss_fn = self.get_use_case_loss_fn
        return self.attack

    def attack_description(self):
        """
        String description for the attack. Return `self.attack` for more details. Edit the function as per use case
        :return: string
        """
        # return str(self.attack)
        return "pgd"


if __name__ == '__main__':
    task = LinfPGDAttack(task_type='attr')
    task.instantiate_attack()
    print(task)

