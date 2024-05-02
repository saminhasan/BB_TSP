from product import Product
from shelf import Shelf
from typing import List, Optional
import random


class Store:
    def __init__(self, rows: int = 15, cols: int = 9):
        self.rows = rows  # Number of rows
        self.cols = cols  # Number of columns
        self.shelves: List[List[Optional[Shelf]]] = [
            [None for _ in range(cols)] for _ in range(rows)
        ]
        self.populate_shelves()

    def populate_shelves(self):
        shelf_width: float = 5.0
        shelf_length: float = 10.0

        def get_spacing() -> int:
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
                self.shelves[i][j] = Shelf(
                    position, cx, cy, random_length, random_width, products
                )

    def get_random_shelves(self, n: int) -> List[Shelf]:
        flat_shelves: List[Shelf] = [
            shelf for row in self.shelves for shelf in row if shelf is not None
        ]
        if n > len(flat_shelves):
            raise ValueError(
                f"Requested {n} shelves, but only {len(flat_shelves)} are available."
            )
        return random.sample(flat_shelves, n)

    def __str__(self) -> str:
        return "\n".join(
            ["\n".join([str(shelf) for shelf in row]) for row in self.shelves]
        )


if __name__ == "__main__":
    store: Store = Store()
    random_shelves = store.get_random_shelves(5)
    for shelf in random_shelves:
        print(shelf)
