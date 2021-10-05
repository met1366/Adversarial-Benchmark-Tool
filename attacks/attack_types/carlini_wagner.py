from attacks.base import Attack
from foolbox.attacks import L2CarliniWagnerAttack


class CarliniWagnerL2Attack(Attack):
    """
    Creates an CarliniWagnerL2A attack
    """
    def __init__(self, task_type):
        """
        :param task_type: attr/reid
        """
        super(CarliniWagnerL2Attack, self).__init__()
        self.task_type = task_type

    def instantiate_attack(self):
        """
        Create an instance of the Foolbox L2CarliniWagner attack
        :return: foolbox attack instance
        """
        if self.task_type == 'attr':
            raise TypeError('C&W for Attribute Alteration not implemented. Skipping!!!')
        self.attack = L2CarliniWagnerAttack()
        return self.attack

    def attack_description(self):
        """
        String description for the attack. Return `self.attack` for more details. Edit the function as per use case
        :return: string
        """
        # return str(self.attack)
        return "carlini_wagner"
