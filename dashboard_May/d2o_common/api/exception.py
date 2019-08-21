class APIError(Exception):
    def to_json(self):
        return {
            'ok': False,
            'message': str(self.message)
        }


class APIMessageError(APIError):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
