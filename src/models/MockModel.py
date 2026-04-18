from .Model import Model


class MockModel(Model):
    """Deterministic local model used for offline smoke tests."""

    def set_API_key(self):
        return None

    def query(self, msg):
        if "Contexts:" not in msg:
            return "I don't know"

        try:
            context_block = msg.split("Contexts:", 1)[1].split("Query:", 1)[0].strip()
        except Exception:
            return "I don't know"

        for line in context_block.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:300]

        return "I don't know"

