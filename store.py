from product import Product
from shelf import Shelf
from typing import List
import random


class Store:
    def __init__(self, rows=15, cols=9):
        self.rows = rows  # Number of rows
        self.cols = cols  # Number of columns
        self.shelves = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.populate_shelves()

    def populate_shelves(self):
        shelf_width = 5.0
        shelf_length = 10.0

        def get_spacing():
            return 5  # random.randint(2, 4)

        for i in range(self.rows):
            for j in range(self.cols):
                position = f"{i+1}{chr(65+j)}"
                cx = j * (shelf_width + get_spacing()) + shelf_width / 2
                cy = i * (shelf_length + get_spacing()) + shelf_length / 2
                random_width = shelf_width + random.uniform(-1, 1)
                random_length = shelf_length + random.uniform(-1, 1)
                products = [
                    Product(
                        f"Product{i*self.cols+j+k}",
                        random.randint(100000000, 999999999),
                        random.randint(10000, 99999),
                        position,
                    )
                    for k in range(25)
                ]
                self.shelves[i][j]  = Shelf(
                    position, cx, cy, random_length, random_width, products
                )

    def get_random_shelves(self, n: int) -> List[Shelf]:
        flat_shelves = [shelf for row in self.shelves for shelf in row]
        return random.sample(flat_shelves, n)

    def __str__(self) -> str:
        return "\n".join(
            ["\n".join([str(shelf) for shelf in row]) for row in self.shelves]
        )


if __name__ == "__main__":
    store = Store()
    random_shelves = store.get_random_shelves(5)
    for shelf in random_shelves:
        print(shelf)
