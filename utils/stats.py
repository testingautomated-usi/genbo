from typing import List, Tuple


class Stats:
    __instance: "Stats" = None

    @staticmethod
    def get_instance():
        if Stats.__instance is None:
            Stats()
        return Stats.__instance

    def __init__(self):

        if Stats.__instance is not None:
            raise Exception("This class is a singleton!")

        self.num_mutate_position = 0
        self.num_mutate_velocity = 0
        self.num_mutate_orientation = 0
        self.num_mutate_position_velocity = 0
        self.num_mutate_position_orientation = 0
        self.num_mutate_velocity_orientation = 0
        self.num_mutate_all = 0
        Stats.__instance = self

    def get_num_mutations(self) -> Tuple[List[int], List[int], int]:
        return (
            [
                self.num_mutate_position,
                self.num_mutate_velocity,
                self.num_mutate_orientation,
            ],
            [
                self.num_mutate_position_velocity,
                self.num_mutate_position_orientation,
                self.num_mutate_velocity_orientation,
            ],
            self.num_mutate_all,
        )
