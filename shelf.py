from typing import List
from product import Product

class Shelf:
    def __init__(self, position: str, cx: float, cy: float, length: float, width: float, products: List[Product]):
        self.position = position
        self.cx = cx
        self.cy = cy
        self.length = length
        self.width = width
        self.products = products

    def __str__(self) -> str:
        product_details = '\n'.join([f"\t{product.name} : {product.upc} " for product in self.products])
        return (f"Shelf Position: {self.position}\nProducts: \n{product_details}")

if __name__ == "__main__":
    product1 = Product("LG TV", 123456789, 98765, "1A", "Electronics", 799.99)
    product2 = Product("Samsung Monitor", 987654321, 12345, "1A", "Electronics", 399.99)
    
    shelf_test = Shelf("1A", 5.0, 5.0, 10.0, 4.0, [product1, product2])
    print(shelf_test)
