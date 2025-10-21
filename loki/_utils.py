


class lokiError(Exception):

    def __init__(self, message, error_code):
        self.message = message
        self.error_code = error_code

        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Error Code: {self.error_code})"

    