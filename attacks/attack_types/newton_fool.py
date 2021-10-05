from attacks.base import Attack
from foolbox.attacks import NewtonFoolAttack


class NewtonAttack(Attack):
    """
    Creates an LinfAdditiveUniformNoise attack
    """

    def __init__(self, task_type):
        """
        :param task_type: attr/reid
        """
        super(NewtonAttack, self).__init__()
        self.task_type = task_type

    def instantiate_attack(self):
        """
        Create an instance of the Foolbox LinfAdditiveUniformNoise attack
        :return: foolbox attack instance
        """
        self.attack = NewtonFoolAttack()
        if self.task_type == 'attr':
            raise TypeError('Newton Method for Attribute Alteration not implemented. Skipping!!!')
        return self.attack

    def attack_description(self):
        """
        String description for the attack. Return `self.attack` for more details. Edit the function as per use case
        :return: string
        """
        # return str(self.attack)
        return "newton"
