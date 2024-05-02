class Product:
    def __init__(
        self,
        name: str,
        upc: int,
        sku: int,
        location: str,
        category: str = "",
        price: float = 0,
    ):
        self.name = name
        self.upc = upc
        self.sku = sku
        self.location = location
        self.category = category
        self.price = price

    def __str__(self) -> str:
        return (
            f"Product Name: {self.name}, UPC: {self.upc}, SKU: {self.sku}, "
            f"Location: {self.location}, Category: {self.category}, "
            f"Price: {self.price} $"
        )


if __name__ == "__main__":
    product_test = Product("LG TV", 123456789, 98765, "1A", "Electronics", 799.99)
    print(product_test)
